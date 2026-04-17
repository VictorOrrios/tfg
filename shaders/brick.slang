#include "shaderio.h"               // Shared definitions with CPU
#include "utils/common.slang"       // Shared definitions with shaders
#include "utils/bbox.slang"         // AABB helpers
#include "utils/sdf.slang"          // SDF functions, operations and primitives
#include "utils/descriptors.slang"  // External memory definitions


[shader("compute")]
[numthreads(BRICK_SIZE, BRICK_SIZE, BRICK_SIZE)]
void computeMain(uint3 tId : SV_DispatchThreadID){
  const int brickJobIdx = tId.x/BRICK_SIZE + (tId.y/BRICK_SIZE)*BRICK_JOB_GROUP_X_DISPATCH_SIZE;

  // Outside dispatch range check
  if(brickJobIdx >= counters[int(Counters::nextBrickJob)])
    return;

  const BrickJob brickJob = brick_job_queue[brickJobIdx];

  const int3 brickId = brickJob.id_level.xyz;
  const int level = brickJob.id_level.w;

  int3 brickIndex = id2index(brickId);
  int3 clipIndex = brickIdx2clipIdx(brickIndex,level);

  // Clean dirty bit from pointer. TODO: This executes BRICK_SIZE³ times per brick
  int clipPointer = clipMap[clipIndex];
  clipPointer = clipPointer & NOT_DIRTY_BIT;
  clipMap[clipIndex] = clipPointer;

  // If not a uniform value brick
  if(clipPointer < NUM_BRICKS_IN_ATLAS){
    const float brick_size = BRICK_SIZES[level];
    const float voxel_size = VOXEL_SIZES[level];
    const float max_voxel_value = MAX_VOXEL_VALUES[level];

    int3 brixelOffset = tId % BRICK_SIZE;
    int3 brickPointer = pointer2atlasIdx(clipPointer);
    int3 atlasIndex = brickPointer + brixelOffset;

    float3 brickPos = float3(brickId)*brick_size;
    float3 p = brickPos + float3(brixelOffset)*voxel_size;

    float sdfValue = map(p,level);
    sdfValue = clamp(sdfValue/max_voxel_value,-1.0,1.0);
    brickAtlas[atlasIndex] = sdfValue;

    // If it needs to generate a material 
    if(all(brixelOffset < MAT_PER_BRICK_AXIS)){
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
