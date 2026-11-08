#include "../shaderio.h"     // Shared definitions with CPU
#include "common.slang"      // Shared definitions with shaders
#include "descriptors.slang" // External memory definitions

#define INSERTION_SORT_THRESHOLD 32

// Object indices fit in 16 bits, so two 8-bit passes are enough.
#if MAX_SCENE_OBJECTS > 65536
#error "radixSort assumes object indices fit in 16 bits"
#endif

void insertionSort(inout uint list[], uint count) {
  for (uint i = 2; i <= count; ++i) {
    uint key = list[i];
    int j = int(i) - 1;

    while (j >= 1 && list[j] > key) {
      list[j + 1] = list[j];
      --j;
    }

    list[j + 1] = key;
  }
}

// Radix sort: one 8-bit pass if every key is below 256, otherwise two.
// No helper function: passing the arrays to one used too much local memory.
void radixSort(inout uint list[]) {
  uint count = list[0];

  if (count <= 1)
    return;

  uint temp[NEAR_OBJECT_LIST_SIZE]; // scratch
  uint histogram[256];

  uint maxKey = 0;
  for (uint i = 1; i <= count; ++i)
    maxKey = max(maxKey, list[i]);

  // Pass 1: low byte, list -> temp
  for (uint i = 0; i < 256; ++i)
    histogram[i] = 0;
  for (uint i = 1; i <= count; ++i)
    histogram[list[i] & 0xFF]++;
  uint sum = 0;
  for (uint i = 0; i < 256; ++i) {
    uint frequency = histogram[i];
    histogram[i] = sum;
    sum += frequency;
  }
  for (uint i = 1; i <= count; ++i) {
    uint value = list[i];
    uint digit = value & 0xFF;
    temp[histogram[digit] + 1] = value;
    histogram[digit]++;
  }

  if (maxKey < 256) {
    for (uint i = 1; i <= count; ++i)
      list[i] = temp[i];
    return;
  }

  // Pass 2: high byte, temp -> list
  for (uint i = 0; i < 256; ++i)
    histogram[i] = 0;
  for (uint i = 1; i <= count; ++i)
    histogram[(temp[i] >> 8) & 0xFF]++;
  sum = 0;
  for (uint i = 0; i < 256; ++i) {
    uint frequency = histogram[i];
    histogram[i] = sum;
    sum += frequency;
  }
  for (uint i = 1; i <= count; ++i) {
    uint value = temp[i];
    uint digit = (value >> 8) & 0xFF;
    list[histogram[digit] + 1] = value;
    histogram[digit]++;
  }
}

// Sorted indices of the objects at point, with the count in list[0].
// Filled in place: returning the array by value broke the pipeline at 512*8.
void objASQuery(float3 point, int level, out uint32_t list[NEAR_OBJECT_LIST_SIZE]) {
  RayQuery<RAY_FLAG_NONE> query;

  RayDesc ray;
  ray.Origin = point;
  ray.Direction = float3(0, 0, 1);
  ray.TMin = 0.0;
  ray.TMax = 0.0000001;

  query.TraceRayInline(tlasObj[level], RAY_FLAG_NONE, 0xFF, ray);

  int i = 1;
  while (query.Proceed() && i < NEAR_OBJECT_LIST_SIZE) {
    list[i] = (uint32_t)query.CandidateInstanceIndex();
    i++;
  }

  list[0] = i - 1;

  if ((i-1) > INSERTION_SORT_THRESHOLD) {
    radixSort(list);
  } else {
    insertionSort(list, i-1);
  }
}
