#include "shaderio.h"             // Shared definitions with CPU
#include "utils/common.slang"     // Shared definitions with shaders
#include "utils/bbox.slang"       // AABB helpers
#include "utils/sdf.slang"        // SDF functions, operations and primitives

//----------------------------
// External memory definition
//----------------------------
[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

[[vk::binding(BindingPoints::aabbs)]] StructuredBuffer<Bbox> aabbs;
[[vk::binding(BindingPoints::objects)]] StructuredBuffer<SceneObject> objects;
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;

[[vk::binding(BindingPoints::clipMap)]] RWTexture3D<uint32_t> clipMap;

[[vk::binding(BindingPoints::buildJobQ)]] StructuredBuffer<BuildJob> build_job_queue;
[[vk::binding(BindingPoints::brickJobQ)]] RWStructuredBuffer<BrickJob> brick_job_queue;
[[vk::binding(BindingPoints::counters)]] RWStructuredBuffer<uint32_t> counters;
[[vk::binding(BindingPoints::indirectCommands)]] RWStructuredBuffer<DispatchIndirectCommand> indirect_commands;
[[vk::binding(BindingPoints::freeList)]] RWStructuredBuffer<uint32_t> free_list;


float map(float3 point){
  const float iniD = 10000.0f;
  float result = iniD;

  for(int obIdx = 0; obIdx < pushConst.numObjects; obIdx++) {
    float3 p = float3(point);
    Bbox bbox = aabbs[obIdx];

    // If not inside bbox primitive continue with next 
    if(!insideBbox(p,bbox))
      continue;

    SceneObject params = objects[obIdx];

    p = mul(params.tInv, float4(p, 1.0)).xyz;

    p = applyRepOp(params.repOp, p, params.spacing.xyz, params.limit.xyz);

    p = applyDefOp(params.defOp, p, params.defP.xyz);

    p /= params.scale;

    float d = evalPrimitive(params.type, p) - params.roundness;

    d *= params.scale;

    result = evalCombOp(params.combOp, d, result, params.smoothness);
  }

  return result;
}

int allocPointer(){
  // TODO: Maybe add a max value check
  int index;
  InterlockedAdd(counters[int(Counters::allocCounter)],1,index);
  index = index%NUM_BRICKS_IN_ATLAS;
  return free_list[index];
}

void freePointer(int pointer){
  // Out of range pointer
  if(pointer >= NUM_BRICKS_IN_ATLAS)
    return;

  int index;
  InterlockedAdd(counters[int(Counters::freeCounter)],1,index);
  index = index%NUM_BRICKS_IN_ATLAS;
  free_list[index] = pointer;
}

[shader("compute")]
[numthreads(MAX_BUILD_JOB_SIZE, MAX_BUILD_JOB_SIZE, MAX_BUILD_JOB_SIZE)]
void computeMain(uint3 tId : SV_DispatchThreadID){
  const int3 hole_min = int3(NUM_BRICKS_PER_AXIS/4);
  const int3 hole_max = int3(NUM_BRICKS_PER_AXIS*3/4-2);

  const int jobIdx = tId.z/MAX_BUILD_JOB_SIZE;
  const BuildJob job = build_job_queue[jobIdx]; // TODO: Group shared memory might be faster
  const int3 brickOffset = int3(tId.x, tId.y, tId.z%MAX_BUILD_JOB_SIZE);

  // Out of bounds thread for this build job => Discard
  if(any(brickOffset >= job.num_b.xyz))
    return;

  const int level = job.min_id_level.w;
  const int3 brickId = job.min_id_level.xyz + brickOffset;  

  // Transform global id to a relative id to the camera id
  const int3 camId = sceneInfo.cameraId0.xyz>>level;
  const int3 relBrickId = brickId - camId + (NUM_BRICKS_PER_AXIS/2);

  // Might not be necesary
  // Out of bounds brick for this level
  if(any(relBrickId < 0) || any(relBrickId > NUM_BRICKS_PER_AXIS-1))
    return;
 
  // Brick inside hole in levels > 0 check
  if(level > 0 && all(relBrickId > hole_min) && all(relBrickId < hole_max))
    return;

  int3 brickIndex = id2index(brickId);
  int3 clipIndex = brickIndex;
  clipIndex.z += NUM_BRICKS_PER_AXIS * level;
  
  // Atomic OR of dirty bit into clip pointer. To prevent repeted brick jobs
  int prevClipPointer;
  InterlockedOr(clipMap[clipIndex],DIRTY_BIT,prevClipPointer);

  if((prevClipPointer & DIRTY_BIT) == 0){
    // Brickjob will be executed and the dirty bit cleaned
    BrickJob brickJob = BrickJob(int4(brickId,level));

    const float brick_size = BRICK_SIZES[level];
    const float max_brick_value = MAX_BRICK_VALUES[level];

    // Evaluate scene at brick center
    float3 p = (float3(brickId)+0.5)*brick_size;
    float centerSDF = map(p);
    
    // Uniform value brick check
    if(abs(centerSDF) > max_brick_value){
      freePointer(prevClipPointer);
      int newPointer = centerSDF<0 ? UNIFORM_NEGATIVE_BRICK_POINTER : UNIFORM_POSITIVE_BRICK_POINTER;
      clipMap[clipIndex] = newPointer | DIRTY_BIT;
    }else if(prevClipPointer >= NUM_BRICKS_IN_ATLAS){
      clipMap[clipIndex] = allocPointer() | DIRTY_BIT;
    }

    // Add to brick job queue with the nextBrickJob counter
    uint brickJobIndex;
    InterlockedAdd(counters[int(Counters::nextBrickJob)],1,brickJobIndex);
    brick_job_queue[brickJobIndex] = brickJob; 

    // Add +1 to brick job groups.y if necesary
    if(brickJobIndex % BRICK_JOB_GROUP_X_DISPATCH_SIZE == 0){
      InterlockedAdd(indirect_commands[0].y,1);
    }
  }

}
