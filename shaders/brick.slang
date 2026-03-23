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
[[vk::binding(BindingPoints::counters)]] StructuredBuffer<uint32_t> counters;


[shader("compute")]
[numthreads(BRICK_SIZE, BRICK_SIZE, BRICK_SIZE)]
void computeMain(uint3 tId : SV_DispatchThreadID){
  const int brickJobIdx = tId.x/BRICK_SIZE + (tId.y/BRICK_SIZE)*BRICK_JOB_GROUP_X_DISPATCH_SIZE;

  // Outside dispatch range check
  if(brickJobIdx >= counters[int(Counters::nextBrickJob)])
    return;

  const BrickJob brickJob = brick_job_queue[brickJobIdx];

  const int3 id = brickJob.id_level.xyz;
  const int level = brickJob.id_level.w;

  int3 brickIndex = id2index(id);
  int3 clipId = brickIndex;
  clipId.z += NUM_BRICKS_PER_AXIS * level;

  // Clean dirty bit from pointer. TODO: This executes BRICK_SIZE³ times per brick
  int clipPointer = clipMap[clipId];
  clipPointer = clipPointer & NOT_DIRTY_BIT;
  clipMap[clipId] = clipPointer;

  // If not a uniform value brick
  if(clipPointer < NUM_BRICKS_IN_ATLAS){
    int3 brickPointer = int3(clipPointer%BRICK_PER_ATLAS_AXIS,clipPointer/BRICK_PER_ATLAS_AXIS,0)*BRICK_SIZE;
    int3 voxelId = int3(tId.x%BRICK_SIZE,tId.y%BRICK_SIZE,tId.z%BRICK_SIZE);

    const float brick_size = BRICK_SIZES[level];
    const float voxel_size = VOXEL_SIZES[level];
    const float max_voxel_value = MAX_VOXEL_VALUES[level];

    float3 brickPos = float3(id)*brick_size;
    float3 p = brickPos + float3(voxelId)*voxel_size;

    float mapValue = map(p,MAX_BRICK_VALUES[level],pushConst.numObjects,aabbs,objects);
    float sdfValue = clamp(mapValue/max_voxel_value,-1.0,1.0);

    brickAtlas[brickPointer+voxelId] = sdfValue;
  }
}
