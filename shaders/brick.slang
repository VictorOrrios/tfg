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
[[vk::binding(BindingPoints::brickAtlas)]] RWTexture3D<float> brickAtlas;

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

[shader("compute")]
[numthreads(BRICK_SIZE, BRICK_SIZE, BRICK_SIZE)]
void computeMain(uint3 tId : SV_DispatchThreadID){
  const int brickJobIdx = tId.x/BRICK_SIZE + (tId.y/BRICK_SIZE)*BRICK_JOB_GROUP_X_DISPATCH_SIZE;

  if(brickJobIdx>=pushConst.numBrickJobs){
    return;
  }

  const BrickJob brickJob = brick_job_queue[brickJobIdx];

  // Invalid job check
  if(brickJob.valid_level.x == 0){
    return;
  }

  int3 id = brickJob.id.xyz;
  int level = brickJob.valid_level.y;
  int3 clipId = id;
  clipId.z += NUM_BRICKS_PER_AXIS * level;

  int clipPointer = clipMap[clipId];
  clipPointer = clipPointer & NOT_DIRTY_BIT;
  clipMap[clipId] = clipPointer;

  if(clipPointer < BRICK_PER_ATLAS_AXIS*BRICK_PER_ATLAS_AXIS){
    int3 brickPointer = int3(clipPointer%BRICK_PER_ATLAS_AXIS,clipPointer/BRICK_PER_ATLAS_AXIS,0)*BRICK_SIZE;
    int3 voxelId = int3(tId.x%BRICK_SIZE,tId.y%BRICK_SIZE,tId.z%BRICK_SIZE);

    const float axis_size = AXIS_SIZES[level];
    const float brick_size = BRICK_SIZES[level];
    const float voxel_size = VOXEL_SIZES[level];
    const float max_voxel_value = MAX_VOXEL_VALUES[level];

    float3 center_offset = float3(0.5*axis_size);
    float3 brickPos = float3(id)*brick_size - center_offset;
    float3 p = brickPos + float3(voxelId)*voxel_size;

    float sdfValue = clamp(map(p)/max_voxel_value,-1.0,1.0);

    brickAtlas[brickPointer+voxelId] = sdfValue;
  }
}
