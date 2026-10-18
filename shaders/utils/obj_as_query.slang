#include "../shaderio.h"     // Shared definitions with CPU
#include "common.slang"      // Shared definitions with shaders
#include "descriptors.slang" // External memory definitions

uint32_t [NEAR_OBJECT_LIST_SIZE] objASQuery(float3 point, int level){
  const float nearRange = MAX_BRICK_VALUES[level];

  uint32_t [NEAR_OBJECT_LIST_SIZE] out;
  for(int i = 0; i<NEAR_OBJECT_LIST_SIZE; i++){
    out[i]=0;
  }

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

  while (query.Proceed()){
    uint32_t objIdx = (uint32_t)query.CandidateInstanceIndex();
    setBit(out[objIdx/32],objIdx%32);
  }

  return out;
}
