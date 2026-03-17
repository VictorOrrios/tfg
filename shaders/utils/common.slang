#pragma once
#include "../shaderio.h"             // Shared definitions with CPU

//--------------------------------------
// Structs, types and enums definitions
//--------------------------------------

struct Ray {
  float3 orig;
  float3 dir;
};

// Axis aligned Bounding Box 
struct Bbox{
  float3 bMin;
  float3 bMax;
};

//--------------------------------------
// Inline macros
//--------------------------------------
inline int3 id2index(int3 x){
  return ((x % NUM_BRICKS_PER_AXIS) + NUM_BRICKS_PER_AXIS) % NUM_BRICKS_PER_AXIS;
}

inline int3 id2voxelIndex(int3 x){
  return ((x % NUM_VOXELS_PER_AXIS) + NUM_VOXELS_PER_AXIS) % NUM_VOXELS_PER_AXIS;
}

inline int3 voxel2offset(int3 x){
  return ((x % (BRICK_SIZE-1)) + (BRICK_SIZE-1)) % (BRICK_SIZE-1);
}