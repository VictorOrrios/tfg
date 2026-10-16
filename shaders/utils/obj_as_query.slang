#include "../shaderio.h"     // Shared definitions with CPU
#include "common.slang"      // Shared definitions with shaders
#include "descriptors.slang" // External memory definitions

uint16_t [NEAR_OBJECT_LIST_SIZE] objASQuery(float3 point, int level){
  uint16_t [NEAR_OBJECT_LIST_SIZE] out;

  RayQuery<RAY_FLAG_NONE> query;

  RayDesc ray;
  ray.Origin = point;
  ray.Direction = float3(1,0,0);
  ray.TMin = 0.001;
  ray.TMax = 0.002;

  query.TraceRayInline(
    tlasObj[level],
    RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
    0xFF,
    ray
  );

  int i = 0;
  while (query.Proceed() && i<NEAR_OBJECT_LIST_SIZE){
    out[i] = (uint16_t)query.CandidateInstanceID();
    i++;
  }

  if(i<NEAR_OBJECT_LIST_SIZE){
    out[i] = -1;
  }else{
    out[0] = -1;
  }

  return out;
}
