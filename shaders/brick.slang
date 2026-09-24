#include "shaderio.h"               // Shared definitions with CPU
#include "utils/common.slang"       // Shared definitions with shaders
#include "utils/bbox.slang"         // AABB helpers
#include "utils/sdf.slang"          // SDF functions, operations and primitives
#include "utils/descriptors.slang"  // External memory definitions

float brickJobMap(float3 point, int3 nolIdx){
  const float iniD = 10000.0f;
  float result = iniD;

  for(int i = 0; i < NEAR_OBJECT_LIST_SIZE; i++) {
    int3 idx = int3(nolIdx.x,nolIdx.y,i);
    int obIdx = nearObjectListAtlas[idx];
    if(obIdx >= pushConst.numObjects)
      break;
    float3 p = float3(point);
    SceneObject params = objects[obIdx];
    result = mapSceneObject(p,params,result);
  }

  return result;
}

//groupshared int4 g_brickIdLevel;
//groupshared int  g_clipPointer;

[shader("compute")]
[numthreads(BRICK_SIZE, BRICK_SIZE, BRICK_SIZE)]
void computeMain(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID){
  const uint brickJobIdx = groupId.x + groupId.y*BRICK_JOB_GROUP_X_DISPATCH_SIZE;

  // Outside dispatch range check
  if(brickJobIdx >= counters[int(Counters::nextBrickJob)])
    return;

  // Groupshared memory version
  /*
  // Read the job and clear the dirty bit once per brick. Doing it from every
  // thread of the group is a non atomic read-modify-write of the same clipMap
  // entry by BRICK_SIZE^3 threads, which races.
  if(groupThreadId.x == 0 && groupThreadId.y == 0 && groupThreadId.z == 0){
    const BrickJob brickJob = brick_job_queue[brickJobIdx];
    g_brickIdLevel = brickJob.id_level;

    int3 brickIndex = id2index(g_brickIdLevel.xyz);
    int3 clipIndex = brickIdx2clipIdx(brickIndex,g_brickIdLevel.w);

    g_clipPointer = clipMap[clipIndex] & NOT_DIRTY_BIT;
    clipMap[clipIndex] = g_clipPointer;
  }

  GroupMemoryBarrierWithGroupSync();

  const int3 brickId = g_brickIdLevel.xyz;
  const int level = g_brickIdLevel.w;
  const int clipPointer = g_clipPointer;

  */

  // Non groupshared memory version
  int4 brickIdLevel;
  int  clipPointer;
  const BrickJob brickJob = brick_job_queue[brickJobIdx];
  brickIdLevel = brickJob.id_level;
  int3 brickIndex = id2index(brickIdLevel.xyz);
  int3 clipIndex = brickIdx2clipIdx(brickIndex,brickIdLevel.w);
  clipPointer = clipMap[clipIndex] & NOT_DIRTY_BIT;
  clipMap[clipIndex] = clipPointer;
  const int3 brickId = brickIdLevel.xyz;
  const int level = brickIdLevel.w;

  // If not a uniform value brick
  if(clipPointer < NUM_BRICKS_IN_ATLAS){
    const float brick_size = BRICK_SIZES[level];
    const float voxel_size = VOXEL_SIZES[level];
    const float max_voxel_value = MAX_VOXEL_VALUES[level];

    int3 brixelOffset = int3(groupThreadId);
    int3 brickPointer = pointer2atlasIdx(clipPointer);
    int3 atlasIndex = brickPointer + brixelOffset;
    int3 nolIndex = pointer2nearObejctListIdx(clipPointer);

    float3 brickPos = float3(brickId)*brick_size;
    float3 p = brickPos + float3(brixelOffset)*voxel_size;

    float sdfValue = brickJobMap(p,nolIndex);
    sdfValue = clamp(sdfValue/max_voxel_value,-1.0,1.0);
    brickAtlas[atlasIndex] = sdfValue;

    // If it needs to generate a material
    if(MAT_PER_BRICK_AXIS > 1 && all(brixelOffset < MAT_PER_BRICK_AXIS)){
      float mat_voxel_size = brick_size / (MAT_PER_BRICK_AXIS-1);
      p = brickPos + float3(brixelOffset)*mat_voxel_size;

      int3 matBrickIdx = pointer2matAtlasIdx(clipPointer);
      int3 matBrixelOffset = brixelOffset;
      int3 matAtlasIndex = matBrickIdx + matBrixelOffset;

      Material mat;
      map(p,mat);
      mat.albedo_shininess.w /= MAX_SHININESS;

      matAtlas[matAtlasIndex] = mat.albedo_shininess;
    }
  }
}
