#include "shaderio.h"               // Shared definitions with CPU
#include "utils/common.slang"       // Shared definitions with shaders
#include "utils/bbox.slang"         // AABB helpers
#include "utils/sdf.slang"          // SDF functions, operations and primitives
#include "utils/descriptors.slang"  // External memory definitions


int allocPointer(){
  // TODO: Maybe add a max value check
  int index;
  InterlockedAdd(counters[int(Counters::allocCounter)],1,index);
  index = index%NUM_BRICKS_IN_ATLAS;
  
  // Out of space in atlas warning
  //if(counters[int(Counters::allocCounter)] - counters[int(Counters::freeCounter)] > NUM_BRICKS_IN_ATLAS)
  //  printf("WARN: ATLAS IS FULL\n");
  
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

inline float3x4 createTransform(int3 id, int level){
  const float brick_size = BRICK_SIZES[level];
  float s = brick_size;
  float3 pos = id*brick_size;
  return float3x4(
    s, 0, 0, pos.x,
    0, s, 0, pos.y,
    0, 0, s, pos.z
  );
}

void updateInstance(int instIdx, int3 id, int level){
  instances[instIdx].mask = 0xFF;
  instances[instIdx].transform = createTransform(id,level);
  instances[instIdx].instanceCustomIndex = level;
}

void maskInstance(int instIdx){
  instances[instIdx].mask = 0x00;
}

[shader("compute")]
[numthreads(MAX_BUILD_JOB_SIZE, MAX_BUILD_JOB_SIZE, MAX_BUILD_JOB_SIZE)]
void computeMain(uint3 tId : SV_DispatchThreadID){
  const int3 hole_min = int3(NUM_BRICKS_PER_AXIS/4 + 1);
  const int3 hole_max = int3(NUM_BRICKS_PER_AXIS*3/4);

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
  
  int3 brickIndex = id2index(brickId);
  int3 clipIndex = brickIdx2clipIdx(brickIndex,level);
  
  // Atomic OR of dirty bit into clip pointer. To prevent repeted brick jobs
  int prevClipPointer;
  InterlockedOr(clipMap[clipIndex],DIRTY_BIT,prevClipPointer);

  if((prevClipPointer & DIRTY_BIT) == 0){
    // Brickjob will be executed and the dirty bit cleaned
    BrickJob brickJob = BrickJob(int4(brickId,level));

    const bool brick_inside_hole = level > 0 && all(relBrickId >= hole_min) && all(relBrickId < hole_max);
    const bool brick_allocated = prevClipPointer < NUM_BRICKS_IN_ATLAS;

    // If brick inside hole => Free brick
    if(brick_inside_hole){
      if(brick_allocated){
        freePointer(prevClipPointer);
        maskInstance(prevClipPointer);
        clipMap[clipIndex] = UNIFORM_POSITIVE_BRICK_POINTER | DIRTY_BIT;
      }
    }else{
      const float brick_size = BRICK_SIZES[level];
      const float max_brick_value = MAX_BRICK_VALUES[level];

      // Evaluate scene at brick center
      float3 p = (float3(brickId)+0.5)*brick_size;
      uint mat = 0;
      float centerSDF = map(p,level);
      
      // Uniform value brick check
      if(abs(centerSDF) > max_brick_value){
        if(brick_allocated){
          freePointer(prevClipPointer);
          maskInstance(prevClipPointer);
        }
        int newPointer = centerSDF<0 ? UNIFORM_NEGATIVE_BRICK_POINTER : UNIFORM_POSITIVE_BRICK_POINTER;
        clipMap[clipIndex] = newPointer | DIRTY_BIT;
      }else if(!brick_allocated){
        int newPointer = allocPointer();
        updateInstance(newPointer,brickId,level);
        clipMap[clipIndex] = newPointer | DIRTY_BIT;
      }else{
        updateInstance(prevClipPointer,brickId,level);
      }
    }

    // Add to brick job queue with the nextBrickJob counter
    uint brickJobIndex;
    InterlockedAdd(counters[int(Counters::nextBrickJob)],1,brickJobIndex);
    brick_job_queue[brickJobIndex] = brickJob;
    
    // Out of space in brick job queue warning
    //if(brickJobIndex >= MAX_NUM_BRICK_JOBS)
    //  printf("Out off space for brick job\n");

    // Add +1 to brick job groups.y if necesary
    if(brickJobIndex % BRICK_JOB_GROUP_X_DISPATCH_SIZE == 0){
      InterlockedAdd(indirect_commands[0].y,1);
    }
  }

}
