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

  BrickJob brickJob = BrickJob(int4(0));
  int prevClipPointer;

  const int jobIdx = tId.z/MAX_BUILD_JOB_SIZE;
  const BuildJob job = build_job_queue[jobIdx]; // TODO: Group shared memory might be faster
  const int3 brickOffset = int3(tId.x, tId.y, tId.z%MAX_BUILD_JOB_SIZE);

  // Out of bounds thread for this build job => Discard
  if(any(brickOffset >= job.num_b.xyz)){
    return;
  }

  const int3 brickIndex = brickOffset + job.min_b_Q_offset.xyz;
  const int brickJobOffset = brickOffset.x + brickOffset.y*job.num_b.x + brickOffset.z*job.num_b.x*job.num_b.y;
  const int brickJobIndex = brickJobOffset + job.min_b_Q_offset.w;

  InterlockedOr(clipMap[brickIndex],DIRTY_BIT,prevClipPointer);
  if((prevClipPointer & DIRTY_BIT) == 0){
    brickJob.id_valid = int4(brickIndex,1);
    float3 p = (float3(brickIndex)+0.5)*BRICK_AXIS_SIZE-float3(0.5*L0_AXIS_WORLD_SIZE);
    float centerSDF = map(p);
    if(abs(centerSDF) > MAX_BRICK_CENTER_VALUE){
      freePointer(prevClipPointer);
      int newPointer = centerSDF<0 ? UNIFORM_NEGATIVE_BRICK_POINTER : UNIFORM_POSITIVE_BRICK_POINTER;
      clipMap[brickIndex] = newPointer | DIRTY_BIT;
    }else if(prevClipPointer >= BRICK_PER_ATLAS_AXIS*BRICK_PER_ATLAS_AXIS){
      clipMap[brickIndex] = allocPointer() | DIRTY_BIT;
    }
  }

  brick_job_queue[brickJobIndex] = brickJob; 
}
