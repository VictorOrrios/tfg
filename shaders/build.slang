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

[[vk::binding(BindingPoints::clipMap)]] RWTexture3D<uint32_t> clipMap;

[[vk::binding(BindingPoints::buildJobQ)]] StructuredBuffer<BuildJob> build_job_queue;
[[vk::binding(BindingPoints::brickJobQ)]] RWStructuredBuffer<BrickJob> brick_job_queue;
[[vk::binding(BindingPoints::freeListCounter)]] RWStructuredBuffer<uint32_t> free_list_counter;


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
  int prevValue;
  InterlockedAdd(free_list_counter[0],1,prevValue);
  // TODO: Maybe add a max value check
  return prevValue;
}

void freePointer(int pointer){
  // Out of range pointer
  if(pointer >= BRICK_PER_ATLAS_AXIS*BRICK_PER_ATLAS_AXIS)
    return;

  // DO NOTHING. TODO FOR LATER
}

[shader("compute")]
[numthreads(MAX_BUILD_JOB_SIZE, MAX_BUILD_JOB_SIZE, MAX_BUILD_JOB_SIZE)]
void computeMain(uint3 tId : SV_DispatchThreadID){
  const int3 hole_min = int3(NUM_BRICKS_PER_AXIS/4-1);
  const int3 hole_max = int3(NUM_BRICKS_PER_AXIS*3/4-1);

  const int jobIdx = tId.z/MAX_BUILD_JOB_SIZE;
  const BuildJob job = build_job_queue[jobIdx]; // TODO: Group shared memory might be faster
  const int3 brickOffset = int3(tId.x, tId.y, tId.z%MAX_BUILD_JOB_SIZE);

  // Out of bounds thread for this build job => Discard
  if(any(brickOffset >= job.num_b_level.xyz)){
    return;
  }

  const int level = job.num_b_level.w;
  const int brickJobOffset = brickOffset.x + brickOffset.y*job.num_b_level.x + brickOffset.z*job.num_b_level.x*job.num_b_level.y;
  const int brickJobIndex = brickJobOffset + job.min_b_Q_offset.w;

  int3 brickIndex = brickOffset + job.min_b_Q_offset.xyz;
  int3 clipIndex = brickIndex;
  clipIndex.z += NUM_BRICKS_PER_AXIS * level;
  
  int prevClipPointer;
  BrickJob brickJob = BrickJob(int4(0),int4(0)); // Initialize as non valid job

  InterlockedOr(clipMap[clipIndex],DIRTY_BIT,prevClipPointer);
  if((prevClipPointer & DIRTY_BIT) == 0){
    // Brick outside hole in levels > 0 check
    if(level == 0 || any(brickIndex <= hole_min) ||  any(brickIndex >= hole_max)){
      brickJob.id = int4(brickIndex,0);
      brickJob.valid_level = int4(1,level,0,0);

      const float axis_size = AXIS_SIZES[level];
      const float brick_size = BRICK_SIZES[level];
      const float max_brick_value = MAX_BRICK_VALUES[level];

      float3 center_offset = float3(0.5*axis_size);
      float3 p = (float3(brickIndex)+0.5)*brick_size-center_offset;
      float centerSDF = map(p);
      
      if(abs(centerSDF) > max_brick_value){
        freePointer(prevClipPointer);
        int newPointer = centerSDF<0 ? UNIFORM_NEGATIVE_BRICK_POINTER : UNIFORM_POSITIVE_BRICK_POINTER;
        clipMap[clipIndex] = newPointer | DIRTY_BIT;
      }else if(prevClipPointer >= BRICK_PER_ATLAS_AXIS*BRICK_PER_ATLAS_AXIS){
        clipMap[clipIndex] = allocPointer() | DIRTY_BIT;
      }
    }
  }

  brick_job_queue[brickJobIndex] = brickJob; 
}
