#pragma once

#include "../shaderio.h"        // Shared definitions with CPU
#include "common.slang"         // Shared definitions with shaders
#include "descriptors.slang"    // External memory definitions

// Samples a 2x2x2 voxel (voxel2) and outputs values into two float4 vectors:
// sBottom: r=s000 g=100 b=101 a=001
// sTop:    r=s010 g=110 b=111 a=011
// Left hand coordinate system (x=right, y=up, z=forwards)
// TODO: This looks slow, might be interesting to make more efficient
void sampleVoxel2(int3 index, int level, out float4 sBottom, out float4 sTop){
  const int NUM_VOXELS_PER_BRICK = BRICK_SIZE-1;
  const float max_voxel_value = MAX_VOXEL_VALUES[level];

  float3 indexNorm = float3(index)/NUM_VOXELS_PER_BRICK;
  int3 brickIndex = int3(floor(indexNorm));
  int3 clipIndex = brickIdx2clipIdx(brickIndex,level);
  uint pointer = clipMap[clipIndex] & NOT_DIRTY_BIT;

  if(pointer == UNIFORM_POSITIVE_BRICK_POINTER){
    sBottom = float4(max_voxel_value);
    sTop    = float4(max_voxel_value);
  }else if(pointer == UNIFORM_NEGATIVE_BRICK_POINTER){
    sBottom = float4(-max_voxel_value);
    sTop    = float4(-max_voxel_value);
  }else{

    int3 brickAtlasIndex = int3(pointer%BRICK_PER_ATLAS_AXIS,pointer/BRICK_PER_ATLAS_AXIS,0);
    int3 voxelId = int3(index.x%NUM_VOXELS_PER_BRICK,index.y%NUM_VOXELS_PER_BRICK,index.z%NUM_VOXELS_PER_BRICK);
    int3 voxelIndex = brickAtlasIndex*BRICK_SIZE + voxelId;

    sBottom = float4(
      brickAtlas[voxelIndex + int3(0,0,0)],  // s000
      brickAtlas[voxelIndex + int3(1,0,0)],  // s100
      brickAtlas[voxelIndex + int3(1,0,1)],  // s101
      brickAtlas[voxelIndex + int3(0,0,1)]   // s001
    ) * max_voxel_value;

    sTop = float4(
      brickAtlas[voxelIndex + int3(0,1,0)],  // s010
      brickAtlas[voxelIndex + int3(1,1,0)],  // s110
      brickAtlas[voxelIndex + int3(1,1,1)],  // s111
      brickAtlas[voxelIndex + int3(0,1,1)]   // s011
    ) * max_voxel_value;
  }
}

void fetchBrickValue(int pointer, int3 voxel_offset, int level, out float4 sBottom, out float4 sTop){
  const float max_voxel_value = MAX_VOXEL_VALUES[level];

  if(pointer == UNIFORM_POSITIVE_BRICK_POINTER){
    sBottom = float4(max_voxel_value);
    sTop    = float4(max_voxel_value);
  }else if(pointer == UNIFORM_NEGATIVE_BRICK_POINTER){
    sBottom = float4(-max_voxel_value);
    sTop    = float4(-max_voxel_value);
  }else{
    int3 atlasIndex = pointer2atlasIdx(pointer) + voxel_offset;
    sBottom = float4(
      brickAtlas[atlasIndex + int3(0,0,0)],  // s000
      brickAtlas[atlasIndex + int3(1,0,0)],  // s100
      brickAtlas[atlasIndex + int3(1,0,1)],  // s101
      brickAtlas[atlasIndex + int3(0,0,1)]   // s001
    ) * max_voxel_value;

    sTop = float4(
      brickAtlas[atlasIndex + int3(0,1,0)],  // s010
      brickAtlas[atlasIndex + int3(1,1,0)],  // s110
      brickAtlas[atlasIndex + int3(1,1,1)],  // s111
      brickAtlas[atlasIndex + int3(0,1,1)]   // s011
    ) * max_voxel_value;
  }
}

float3 interpolateMat(int pointer, int3 voxel_offset, float3 coor){

  if(pointer < NUM_BRICKS_IN_ATLAS){
    int3 base = pointer2matAtlasIdx(pointer);
    base += voxel_offset;
    return lerp(
      lerp(
        lerp(
          matAtlas[base + int3(0,0,0)],
          matAtlas[base + int3(0,0,1)],
          coor.z
        ),
        lerp(
          matAtlas[base + int3(1,0,0)],
          matAtlas[base + int3(1,0,1)],
          coor.z
        ),
        coor.x
      ),
      lerp(
        lerp(
          matAtlas[base + int3(0,1,0)],
          matAtlas[base + int3(0,1,1)],
          coor.z
        ),
        lerp(
          matAtlas[base + int3(1,1,0)],
          matAtlas[base + int3(1,1,1)],
          coor.z
        ),
        coor.x
      ),
      coor.y
    ).xyz;
  }else{
    return float3(1.0);
  }
}

void getkConstants(float4 sBottom, float4 sTop, out float4 k0123, out float4 k4567){
  float aux1 = sBottom.a - sBottom.r; // s001 - s000
  float aux2 = sTop.a - sTop.r;       // s011 - s010
  
  // Bottom bilinear coeff
  k0123 = float4(
    sBottom.r,                // k0 = s000
    aux1,                     // k1 = aux1
    sBottom.g-sBottom.r,      // k2 = s100 - s000
    sBottom.b-sBottom.g-aux1  // k3 = s101 - s100 - aux1
  );

  // Top bilinear coeff
  k4567 = float4(
    sTop.r - k0123.x,                 // k4 = s010 - k0
    aux2 - k0123.y,                   // k5 = aux2 - k1
    sTop.g - sTop.r - k0123.z,        // k6 = s110 - s010 - k2
    sTop.b - sTop.g - aux2 - k0123.w  // k7 = s111 - s110 - aux2 - k3
  );
}

void getkConstants(int3 index, out float4 k0123, out float4 k4567){
  float4 sBottom, sTop;
  sampleVoxel2(index,0, sBottom,sTop);
  getkConstants(sBottom,sTop,k0123,k4567);
}

// Interpolates at p using the bilinear coefficients
float triLerpK(float3 p, float4 k0123, float4 k4567){
  // f(x,y,z) = y(k5·z + k6·x + k7·z·x + k4) + (k1·z + k2·x + k3·z·x + k0)
  return fma(p.y,
    fma(k4567.y, p.z,
      fma(k4567.z, p.x,
        fma(k4567.w, p.z*p.x, k4567.x))),
    fma(k0123.y, p.z,
      fma(k0123.z, p.x,
        fma(k0123.w, p.z*p.x, k0123.x)))
    );
}

float triLerp(float3 p, float4 sBottom, float4 sTop){
  //p = saturate(p); // Ensures it is in [0,1]³
  return lerp(
    lerp(
      lerp(sBottom.r,sBottom.a,p.z),
      lerp(sBottom.g,sBottom.b,p.z),
      p.x
    ),
    lerp(
      lerp(sTop.r,sTop.a,p.z),
      lerp(sTop.g,sTop.b,p.z),
      p.x
    ),
    p.y
  );
}

// Calculates analytically the normal of a voxel2 at point p
// Pre: p must be within voxel range in [0,1]³
float3 calcNormalAnalytic(float3 p, float4 sBottom, float4 sTop){
  //p = saturate(p);
  float dx = lerp(
    lerp(
      sBottom.g - sBottom.r,  // s100 - s000
      sTop.g - sTop.r,        // s110 - s010
      p.y),
    lerp(
      sBottom.b - sBottom.a,  // s101 - s001
      sTop.b - sTop.a,        // s111 - s011
      p.y),
    p.z
  );

  float dy = lerp(
    lerp(
      sTop.r - sBottom.r,     // s010 - s000
      sTop.g - sBottom.g,     // s110 - s100
      p.x),
    lerp(
      sTop.a - sBottom.a,     // s011 - s001
      sTop.b - sBottom.b,     // s111 - s101
      p.x),
    p.z
  );

  float dz = lerp(
    lerp(
      sBottom.a - sBottom.r,  // s001 - s000
      sBottom.b - sBottom.g,  // s101 - s100
      p.x),
    lerp(
      sTop.a - sTop.r,        // s011 - s010
      sTop.b - sTop.g,        // s111 - s110
      p.x),
    p.y
  );

  return normalize(float3(dx,dy,dz));
}

float3 analyticNormal(float3 p){
  int level = getLevelOfPoint(p);
  float voxel_size = VOXEL_SIZES[level];
  // Scale normalization
  float3 pVoxelNorm = p/voxel_size;
  // Truncate normalized coordinates to get the id
  int3 voxelId = int3(floor(pVoxelNorm));
  // Get the grid indices with floor division modulus (WARNING: Normal modulus % will not give correct result)
  int3 voxelIndex = id2voxelIdx(voxelId);
  // Get grid index and coordinates inside the voxel
  int3 currentIndex = max(int3(0),min(int3(NUM_VOXELS_PER_AXIS-1),
    voxelIndex));
  float3 voxelCoor = frac(pVoxelNorm);
  // Sample voxel
  float4 sBottom,sTop;
  sampleVoxel2(currentIndex,level,sBottom,sTop);
  // Calc normal
  return calcNormalAnalytic(voxelCoor,sBottom,sTop);
}

// Returns the coefficients for: f(t) = c_0 + c_1·t + c_2·t² + c_3·t³
// return: x=c_0 y=c_1 z=c_2 w=c_3
// This was based from the formula in "Ray Tracing of Signed Distance Function Grids" NVIDIA 2022
// The paper uses a different coordinate system: xyz (Paper) => zxy (This code)
// TODO: Check if it works, if not throw it and start again
float4 getCubeCoeff(Ray r, float4 k0123, float4 k4567){
  float3 o = r.orig; float3 d = r.dir;

  float3 m012 = float3(
    o.z*o.x,
    d.z*d.x,
    fma(o.z,d.x, o.x*d.z)
  );
  float3 m345 = float3(
    fma(k4567.y,o.y, - k0123.y),
    fma(k4567.z,o.y, - k0123.z),
    fma(k4567.w,o.y, - k0123.w),
  );

  // c_0 = oz·m3 + ox·m4 + m0·m5 + (k4·oy - k0)
  // c_1 = dz·m3 + dx·m4 + m2·m5 + dy(k4 + k5·oz + k6·ox + k7·m0)
  // c_2 = m1·m5 + dy(k5·dz + k6·dx + k7·m2)
  // c_3 = k7·m1·dy
  return float4(
    fma(o.z, m345.x, 
      fma(o.x, m345.y, 
        fma(m012.x, m345.z, 
          fma(k4567.x, o.y, -k0123.x)))),
    
    fma(d.z, m345.x, 
      fma(d.x, m345.y, 
        fma(m012.z, m345.z, 
          d.y * fma(k4567.y, o.z, 
                  fma(k4567.z, o.x, 
                    fma(k4567.w, m012.x, k4567.x)))))),
    
    fma(m012.y, m345.z, 
      d.y * fma(k4567.y, d.z, 
              fma(k4567.z, d.x, k4567.w * m012.z))),
    
    k4567.w*m012.y*d.y
  );
}

// Evaluates a cubic polinomial
// return = f(t) = c.x + c.y·t + c.z·t² + c.w·t³
float evalCubic(float t, float4 c){
  // Horner form: ((c3 * t + c2) * t + c1) * t + c0
  return fma(t, fma(t, fma(t, c.w, c.z), c.y), c.x);
}

int getLevelOfPoint(float3 p) {
  int3 camId0 = sceneInfo.cameraId0.xyz;
  int3 pId0 = int3(floor(p / BRICK_SIZES[0]));
  
  int3 diff = pId0 - camId0;
  //diff += max(0,sign(diff));
  diff = abs(diff);
  int maxDiff = max(diff.x,max(diff.y,diff.z));
  
  const int HALF = NUM_BRICKS_PER_AXIS/2;
  const float GRID_OFFSET = 2.0 / NUM_BRICKS_PER_AXIS;
  float logVal = log2((float)(maxDiff) * GRID_OFFSET);
  int level = int(ceil(max(0,logVal)));

  int3 camId = camId0 >> level;
  int3 pId = pId0 >> level;
  int3 rel_id = pId - camId + HALF;
  if(any(rel_id < 0) || any(rel_id >= NUM_BRICKS_PER_AXIS)){
    level++;
  }

  return clamp(level,0,CLIPMAP_LEVELS - 1);
}

float3 sampleInterpolatedMat(float3 p){
  float4 sBottom, sTop;
  int level = getLevelOfPoint(p);
  const float brick_size = BRICK_SIZES[level];

  // Scale normalization
  float3 pBrickNorm = p/brick_size;
  // Truncate normalized coordinates to get the id
  int3 brickId = int3(floor(pBrickNorm));
  // Get coordinates inside brick
  float3 brickCoor = frac(pBrickNorm);

  // Get de mat voxel params
  float3 relMatVoxelCoor = brickCoor * (MAT_PER_BRICK_AXIS-1);
  int3 matVoxelOffset = int3(relMatVoxelCoor); // No floor needed, brickCoor ∈ [0,1]
  float3 matVoxelCoor = fract(relMatVoxelCoor);

  // Get pointer of allocated brick
  int3 brickIndex = id2index(brickId);
  int3 clipIndex = brickIdx2clipIdx(brickIndex, level);
  int clipPointer = clipMap[clipIndex] & NOT_DIRTY_BIT;

  return interpolateMat(clipPointer,matVoxelOffset,matVoxelCoor);
}

float sampleInterpolatedPoint(float3 p){
  float4 sBottom, sTop;
  int level = getLevelOfPoint(p);
  const float brick_size = BRICK_SIZES[level];
  const float voxel_size = VOXEL_SIZES[level];

  // Scale normalization
  float3 pBrickNorm = p/brick_size;
  // Truncate normalized coordinates to get the id
  int3 brickId = int3(floor(pBrickNorm));
  // Get coordinates inside brick
  float3 brickCoor = frac(pBrickNorm);

  // Get de voxel params
  float3 relVoxelCoor = brickCoor * (BRICK_SIZE-1);
  int3 voxelOffset = int3(relVoxelCoor); // No floor needed, brickCoor ∈ [0,1]
  float3 voxelCoor = fract(relVoxelCoor);

  // Get pointer of allocated brick
  int3 brickIndex = id2index(brickId);
  int3 clipIndex = brickIdx2clipIdx(brickIndex, level);
  int clipPointer = clipMap[clipIndex] & NOT_DIRTY_BIT;

  // Update current voxel data and index
  fetchBrickValue(clipPointer,voxelOffset,level,sBottom,sTop);
  // Interpolate current sdf value of the voxel2
  return triLerp(voxelCoor,sBottom,sTop);
}