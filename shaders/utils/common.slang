#pragma once

#include "../shaderio.h"             // Shared definitions with CPU

//--------------------------------------
// Math const
//--------------------------------------
#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309
#define E_NUMBER 2.71828182845


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

inline int3 id2voxelIdx(int3 x){
  return ((x % NUM_VOXELS_PER_AXIS) + NUM_VOXELS_PER_AXIS) % NUM_VOXELS_PER_AXIS;
}

inline int3 voxel2offset(int3 x){
  return ((x % (BRICK_SIZE-1)) + (BRICK_SIZE-1)) % (BRICK_SIZE-1);
}

inline int3 brickIdx2clipIdx(int3 id, int level){
  id.z += NUM_BRICKS_PER_AXIS * level;
  return id;
}

inline int3 pointer2atlasIdx(int pointer){
  return int3(pointer % BRICK_PER_ATLAS_AXIS, pointer / BRICK_PER_ATLAS_AXIS, 0) * BRICK_SIZE;
}

inline int3 pointer2matAtlasIdx(int pointer){
  return int3(pointer % BRICK_PER_ATLAS_AXIS, pointer / BRICK_PER_ATLAS_AXIS, 0) * MAT_PER_BRICK_AXIS;
}