#include "../shaderio.h"     // Shared definitions with CPU
#include "common.slang"      // Shared definitions with shaders
#include "descriptors.slang" // External memory definitions

/*
uint32_t [NEAR_OBJECT_LIST_SIZE] objASQuery(float3 point, int level){
  const float nearRange = MAX_BRICK_VALUES[level];

  uint32_t [NEAR_OBJECT_LIST_SIZE] out;

  RayQuery<RAY_FLAG_NONE> query;

  RayDesc ray;
  ray.Origin = point;
  ray.Direction = float3(0,0,1);
  ray.TMin = 0.0;
  ray.TMax = 0.001;

  query.TraceRayInline(
    tlasObj[level],
    RAY_FLAG_NONE,
    0xFF,
    ray
  );

  int i = 0;
  while (query.Proceed() && i<NEAR_OBJECT_LIST_SIZE){
    uint32_t objIdx = (uint32_t)query.CandidateInstanceIndex();
    //if (!nearBbox(point, aabbs[objIdx], nearRange)) continue;
    out[i] = objIdx;
    i++;
  }

  if(i<NEAR_OBJECT_LIST_SIZE){
    out[i] = MAX_SCENE_OBJECTS + 1;
  }else{
    //printf("RayQ: List full\n");
    out[1] = MAX_SCENE_OBJECTS;
  }

  return out;
}
*/
