#include "../shaderio.h"     // Shared definitions with CPU
#include "common.slang"      // Shared definitions with shaders
#include "descriptors.slang" // External memory definitions

void radixSort(inout uint list[]) {
  uint count = list[0];

  if (count <= 1)
    return;

  uint temp[NEAR_OBJECT_LIST_SIZE]; // scratch
  uint histogram[256];

  // 4 pases: 8, 16, 24 y 32 bits.
  for (uint shift = 0; shift < 24; shift += 8) {
    // Init histogram to 0
    for (uint i = 0; i < 256; ++i)
      histogram[i] = 0;

    // Fill with frecuency
    for (uint i = 1; i <= count; ++i) {
      uint digit = (list[i] >> shift) & 0xFF;
      histogram[digit]++;
    }

    // Turn to acummulative sum
    uint sum = 0;
    for (uint i = 0; i < 256; ++i) {
      uint frequency = histogram[i];
      histogram[i] = sum;
      sum += frequency;
    }

    // Fill scratch using acumm f
    for (uint i = 1; i <= count; ++i) {
      uint value = list[i];
      uint digit = (value >> shift) & 0xFF;

      temp[histogram[digit] + 1] = value;
      histogram[digit]++;
    }

    // Copy to output
    for (uint i = 1; i <= count; ++i)
      list[i] = temp[i];
  }
}

uint32_t[NEAR_OBJECT_LIST_SIZE] objASQuery(float3 point, int level) {
  const float nearRange = MAX_BRICK_VALUES[level];

  uint32_t[NEAR_OBJECT_LIST_SIZE] out;

  RayQuery<RAY_FLAG_NONE> query;

  RayDesc ray;
  ray.Origin = point;
  ray.Direction = float3(0, 0, 1);
  ray.TMin = 0.0;
  ray.TMax = 0.001;

  query.TraceRayInline(tlasObj[level], RAY_FLAG_NONE, 0xFF, ray);

  int i = 1;
  while (query.Proceed() && i < NEAR_OBJECT_LIST_SIZE) {
    uint32_t objIdx = (uint32_t)query.CandidateInstanceIndex();
    // if (!nearBbox(point, aabbs[objIdx], nearRange)) continue;
    out[i] = objIdx;
    i++;
  }

  out[0] = i;

  radixSort(out);

  return out;
}
