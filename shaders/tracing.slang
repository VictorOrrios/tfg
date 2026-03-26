/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
// TODO: Change banner

#include "shaderio.h"       // Shared definitions with CPU
#include "utils/common.slang"     // Shared definitions with shaders
#include "utils/bbox.slang"       // AABB helpers
#include "utils/sdf.slang"        // SDF functions, operations and primitives
#include "utils/palettes.slang"   // Debug color palette

//----------------------------
// External memory definition
//----------------------------

[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

[[vk::binding(BindingPoints::normalBuffer)]] RWTexture2D<float4> normalImage;
[[vk::binding(BindingPoints::albedoBuffer)]] RWTexture2D<float4> albedoImage;
[[vk::binding(BindingPoints::depthBuffer)]] RWTexture2D<float> depthImage;
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;
[[vk::binding(BindingPoints::aabbs)]] StructuredBuffer<Bbox> aabbs;
[[vk::binding(BindingPoints::objects)]] StructuredBuffer<SceneObject> objects;

[[vk::binding(BindingPoints::clipMap)]] RWTexture3D<uint32_t> clipMap;
[[vk::binding(BindingPoints::brickAtlas)]] RWTexture3D<float> brickAtlas;

#ifdef NDEBUGSHADER
// TODO: Change to false for final release and boost performance
const static bool debugColor = pushConst.debug.mode == 1;
const static bool bboxColor = pushConst.debug.mode == 5;
#else
const static bool debugColor = pushConst.debug.mode == 1;
const static bool bboxColor = pushConst.debug.mode == 5;
#endif


static float3 debug = float3(0.0);

float3 scalarToColor(float x){
  return scalarToColor(x, pushConst.debug.palette);
}

// Evaluates the whole sdf scene in a point in space
float map(float3 p){
  const float maxD = 100.0;
  float mapValue = map(p,maxD,pushConst.numObjects,aabbs,objects);
  return min(mapValue,maxD);
}

// True numerical
float3 evalNormalTrue(float3 p){
  const float eps = 0.0001;
  const float2 h = float2(eps,0.0);
  return normalize(float3(
    map(p+h.xyy)-map(p-h.xyy),
    map(p+h.yxy)-map(p-h.yxy),
    map(p+h.yyx)-map(p-h.yyx)
  ));
}

// Tetrahedron aproximation
float3 evalNormal(float3 p){
  const float h = 0.0001;
  const float2 k = float2(1,-1);
  return normalize(
    k.xyy*map(p + k.xyy*h) +
    k.yyx*map(p + k.yyx*h) + 
    k.yxy*map(p + k.yxy*h) + 
    k.xxx*map(p + k.xxx*h)
  );
}

// Sphere traces the scene evaluating the scene sdf at every step
// Pre: r.orig must be inside bbox
float sphereTraceMap(Ray ray, float maxDepth){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS * int(CLIPMAP_LEVELS*0.5);
  const float MIN_DIST = 0.0001;
  
  float depth = 0.0;

  for(int i = 0; i < MAX_ITERATIONS; i++){
    float3 p = ray.orig + ray.dir * depth;
    float dist = map(p);
    
    if(abs(dist) < MIN_DIST){
      return depth;  // Hit
    }
    
    if(depth >= maxDepth) return -1.0;  // No hit

    depth += dist;
  }
  
  return -1.0;
}



// Samples a 2x2x2 voxel (voxel2) and outputs values into two float4 vectors:
// sBottom: r=s000 g=100 b=101 a=001
// sTop:    r=s010 g=110 b=111 a=011
// Left hand coordinate system (x=right, y=up, z=forwards)
// TODO: This looks slow, might be interesting to make more efficient
void sampleVoxel2(int3 index, int level, out float4 sBottom, out float4 sTop){
  const int NUM_VOXELS_PER_BRICK = BRICK_SIZE-1;
  const float max_voxel_value = MAX_VOXEL_VALUES[level];

  int3 brickIndex = index/NUM_VOXELS_PER_BRICK;
  brickIndex.z += NUM_BRICKS_PER_AXIS*level;
  uint pointer = clipMap[brickIndex] & NOT_DIRTY_BIT;

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

void sampleVoxel2New(int3 voxelOffset, int3 brickIndex, int level, out float4 sBottom, out float4 sTop){
  const float max_voxel_value = MAX_VOXEL_VALUES[level];

  int3 clipIndex = brickIndex;
  clipIndex.z += NUM_BRICKS_PER_AXIS*level;
  uint pointer = clipMap[clipIndex] & NOT_DIRTY_BIT;

  if(pointer == UNIFORM_POSITIVE_BRICK_POINTER){
    sBottom = float4(max_voxel_value);
    sTop    = float4(max_voxel_value);
  }else if(pointer == UNIFORM_NEGATIVE_BRICK_POINTER){
    sBottom = float4(-max_voxel_value);
    sTop    = float4(-max_voxel_value);
  }else{

    int3 brickAtlasIndex = int3(pointer%BRICK_PER_ATLAS_AXIS,pointer/BRICK_PER_ATLAS_AXIS,0);
    int3 voxelIndex = brickAtlasIndex*BRICK_SIZE + voxelOffset;

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

void fetchBrickValue(int pointer, int3 voxel_offset, out float4 sBottom, out float4 sTop){
  int3 brickAtlasIndex = int3(pointer%BRICK_PER_ATLAS_AXIS,pointer/BRICK_PER_ATLAS_AXIS,0);
  int3 voxelIndex = brickAtlasIndex*BRICK_SIZE + voxel_offset;
  sBottom = float4(
    brickAtlas[voxelIndex + int3(0,0,0)],  // s000
    brickAtlas[voxelIndex + int3(1,0,0)],  // s100
    brickAtlas[voxelIndex + int3(1,0,1)],  // s101
    brickAtlas[voxelIndex + int3(0,0,1)]   // s001
  );

  sTop = float4(
    brickAtlas[voxelIndex + int3(0,1,0)],  // s010
    brickAtlas[voxelIndex + int3(1,1,0)],  // s110
    brickAtlas[voxelIndex + int3(1,1,1)],  // s111
    brickAtlas[voxelIndex + int3(0,1,1)]   // s011
  );
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
  diff += max(0,sign(diff));
  diff = abs(diff);
  int maxDiff = max(diff.x,max(diff.y,diff.z));
  
  const int HALF = NUM_BRICKS_PER_AXIS / 2;
  float logVal = log2((float)(maxDiff) / (float)HALF);
  int level = clamp((int)ceil(logVal),0,CLIPMAP_LEVELS - 1);
  
  int3 camId = id0LevelTransform(camId0,level);
  int3 pId = id0LevelTransform(pId0,level);
  int3 rel_id = pId - camId;
  if (all(rel_id < HALF)) {
    return level;
  }else{
    return min(level+1,CLIPMAP_LEVELS - 1);
  }
}

bool sphereTraceBrick(Ray r, int3 id, int level, out float depth){
  const int MAX_ITERATIONS = BRICK_SIZE * 5;
  depth = 0.0;

  int3 clipId = id;
  clipId.z += NUM_BRICKS_PER_AXIS * level;
  const int pointer = clipMap[clipId] & NOT_DIRTY_BIT;
  if(pointer > NUM_BRICKS_IN_ATLAS)
    return pointer == UNIFORM_NEGATIVE_BRICK_POINTER;

  const float brick_size = BRICK_SIZES[level];
  const float voxel_size = VOXEL_SIZES[level];
  const float3 brickPos = id*brick_size;
  float3 p = r.orig;
  float3 diff = p - brickPos;
  float4 sBottom,sTop;
  int3 prev_voxel_offset;

  for(int step = 0; step<MAX_ITERATIONS; step++){
    if(any(diff<0.0) || any(diff>brick_size))
      break;

    float3 diffNorm = diff/voxel_size;
    int3 voxel_offset = int3(floor(diffNorm));
    float3 voxel_coor = frac(diffNorm);
    if(step == 0 || any(prev_voxel_offset != voxel_offset)){
      fetchBrickValue(pointer,voxel_offset,sBottom,sTop);
      prev_voxel_offset = voxel_offset;
    }

    float t = triLerp(voxel_coor,sBottom,sTop);

    if(t < voxel_size/10){
      depth = length(r.orig - p);
      return true;
    }

    p += r.dir*t;

    diff = p - brickPos;
  }

  return false;
}

// Pre: r.orig must be inside bbox
float sphereTraceBbox(Ray r, Bbox bbox){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS * int(CLIPMAP_LEVELS*0.5);

  // Debug visualizers
  const bool drawSilhouette   = debugColor && true;   // Draw the silhouette of the grid
  const bool drawSteps        = debugColor && false;  // Draw the numer of steps taken
  const bool drawLevel        = debugColor && false;  // Draw the clip map level at hit point
  const bool drawHitIndex     = debugColor && false;  // Draw the grid brick index at the hit point
  const bool drawBrickPointer = debugColor && true;   // Draw the pointer value of the brick at hit point

  float3 p = r.orig, dir = r.dir;
  int3 currentIndex, lastIndex;
  float4 sBottom, sTop;
  int level;

  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    level = getLevelOfPoint(p);

    const float voxel_size = VOXEL_SIZES[level];

    // Scale normalization
    float3 pVoxelNorm = p/voxel_size;
    // Truncate normalized coordinates to get the id
    int3 voxelId = int3(floor(pVoxelNorm));
    // Get the grid indices with floor division modulus (WARNING: Normal modulus % will not give correct result)
    int3 voxelIndex = id2voxelIndex(voxelId);
    // Get grid index and coordinates inside the voxel
    currentIndex = voxelIndex;
    float3 voxelCoor = frac(pVoxelNorm);

    // If this voxel hasn't been visited yet
    if(steps == 1 || any(currentIndex != lastIndex)){
    //if(true){
      // Update current voxel data and index
      sampleVoxel2(currentIndex,level,sBottom,sTop);
      //sampleVoxel2New(voxelOffset,currentIndex,level,sBottom,sTop);
      //getkConstants(sBottom,sTop,k0123,k4567);
      lastIndex = currentIndex;
    }

    // Interpolate current sdf value of the voxel2
    float t = triLerp(voxelCoor,sBottom,sTop);


    // Cutoff condition: A distance of 1/10 voxels
    // TODO: Fine tune to better performance/quality
    if(t < voxel_size/10){

      // Debug visualizer of step count
      if(drawSteps) debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));

      // Level indicator
      if(drawLevel) debug = scalarToColor(float(level)/float(CLIPMAP_LEVELS-1));

      // Debug visualizer of hit brick index
      if(drawHitIndex) debug = float3(currentIndex/(BRICK_SIZE-1))/float3(NUM_BRICKS_PER_AXIS);

      // Brick pointer visualizer
      if(drawBrickPointer){
        int3 brickIndex = currentIndex/(BRICK_SIZE-1);
        brickIndex.z += NUM_BRICKS_PER_AXIS*level;
        uint pointer = clipMap[brickIndex] & NOT_DIRTY_BIT;
        if((pointer & DIRTY_BIT) != 0){
          debug = scalarToColor(-1);
        }else{
          debug = scalarToColor(float(pointer)/float(NUM_BRICKS_IN_ATLAS));
        }
      }

      // Returns depth of hit
      return length(p - r.orig);
    }
    
    // Update p to move the estimated length to surface
    p += r.dir*t;
  
    // Check if ray has left the bbox
    if(!insideBbox(p,bbox)){
      // Outside grid
      if(drawSteps){
        debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
        return 1;
      }else{
        // No hit
        return -1;
      }
    }
  }

  // Over iterated 
  if(drawSteps){
    // Debug visualizer of step count
    debug = scalarToColor(1.1);
    return 1;
  }else{
    // No hit
    return -1;
  }
}

// WARNING: Not working
// Experiment of hybrid traversal
float sphereTraceBbox2(Ray r, Bbox bbox){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS * int(CLIPMAP_LEVELS*0.5);
  const int3 hole_min = int3(NUM_BRICKS_PER_AXIS/4);
  const int3 hole_max = int3(NUM_BRICKS_PER_AXIS*3/4-2);

  float depth = -1;

  // Initial boundig box hit test
  float3 p, far, dir = r.dir;
  bool hit = intersectCube(r, bbox, p, far);
  if(!hit) return -1;

  const int3 step = sign(dir);

  int level = getLevelOfPoint(p);
  float brickSize = BRICK_SIZES[level];

  float3 pBrickNorm = p/brickSize;
  int3 brickId = int3(floor(pBrickNorm));
  int3 brickIndex = id2index(brickId);
  float3 brickCoor = frac(pBrickNorm);
  
  float3 delta = brickSize/abs(dir);
  float3 deltaOffset = select(step == 1,1.0-brickCoor,brickCoor);
  float3 maxD = delta*deltaOffset;

  Ray brickRay;
  brickRay.dir = dir;
  float3 origP = p;

  for(int steps = 1; steps <= MAX_ITERATIONS; steps++){

    if(any(brickIndex)<0 || any(brickIndex)>(NUM_BRICKS_PER_AXIS-1)){
      // No more levels checks
      if(level == CLIPMAP_LEVELS-1)
        return -1;

      level++;
      delta *= 2;
      brickId = id0LevelTransform(brickId,1);
      brickIndex = id2index(brickId);
    }else if(level>0 && all(brickIndex > hole_min) && all(brickIndex < hole_max)){
      // -1 level
      level--;
      delta /= 2;
      brickId *= 2;
      brickIndex = id2index(brickId);
      // In case the index is NUM_BRICKS_PER_AXIS-2 add 1 to it and the id
      int3 holeIndexFix = select(brickIndex == NUM_BRICKS_PER_AXIS-2,1,0);
      brickIndex += holeIndexFix;
      brickId += holeIndexFix;
    }
    

    // Brick hit check
    brickRay.orig = p;
    if(!intersectCube(brickRay, bbox, p, far))
      return -1;
    if(sphereTraceBrick(brickRay,brickId,level,depth)){
      return length(brickRay.orig+(dir*depth) - r.orig);
    }

    // Add delta to lowest d
    if (maxD.x < maxD.y && maxD.x < maxD.z) {
      // X-axis traversal.
      p.x = origP.x + maxD.x*step.x;
      maxD.x += delta.x;
      brickIndex.x += step.x;
      brickId.x += step.x;
    } else if (maxD.y < maxD.z) {
      // Y-axis traversal.
      p.y = origP.y + maxD.y*step.y;
      maxD.y += delta.y;
      brickIndex.y += step.y;
      brickId.y += step.y;
    } else {
      // Z-axis traversal.
      p.z = origP.z + maxD.z*step.z;
      maxD.z += delta.z;
      brickIndex.z += step.z;
      brickId.z += step.z;
    }
  }

  return -1;
}

float3 analyticNormal(float3 p){
  int level = getLevelOfPoint(p);
  float voxel_size = VOXEL_SIZES[level];
  // Scale normalization
  float3 pVoxelNorm = p/voxel_size;
  // Truncate normalized coordinates to get the id
  int3 voxelId = int3(floor(pVoxelNorm));
  // Get the grid indices with floor division modulus (WARNING: Normal modulus % will not give correct result)
  int3 voxelIndex = id2voxelIndex(voxelId);
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


float traceScene(Ray r, out float3 normal){
  float depth = -1.0f;
  normal = float3(0.0);
  float3 lastDebug = debug;

  // Iterated in reverse order because user placed objects are usually nearest to screen and at the bottom of the stack
  for(int i = pushConst.numObjects; i>=0; i--){
    Bbox bbox = aabbs[i];

    // Peprare the ray that will trace the object
    Ray bboxRay = r;
    float3 far, dir = r.dir;
    if(!intersectCube(r, bbox, bboxRay.orig, far)) continue;

    // Cull if bbox is too far
    float orig2P = length(bboxRay.orig - r.orig);
    if(depth>0 && depth<orig2P)
      continue;

    // Trace bbox using the method defined by user
    float objDepth;
    objDepth = sphereTraceBbox(bboxRay,bbox);
    //objDepth = sphereTraceMap(bboxRay,length(far - bboxRay.orig));

    // Update min depth
    if(objDepth >= 0){
      objDepth += orig2P;
      if(depth < 0 || (objDepth>=0 && objDepth<depth)){
        depth = objDepth;
        lastDebug = debug;
      }
    }
  }

  // Use the nearest debug value that counted as the hit
  debug = lastDebug;

  // Calc normal using user defined method
  if(depth >= 0){
    float3 p = r.orig + r.dir*depth;
    normal = analyticNormal(p);
    //normal = evalNormal(p);
  }

  return depth;
}

// PRE: r.dir must be normalized
float rayMarchAmaWoo(Ray r, Bbox bbox, out float3 normal){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS*3;

  // Debug visualizers
  const bool drawSilhouette = debugColor && true;   // Draw the silhouette of the grid 
  const bool drawSteps      = debugColor && true;   // Draw the numer of steps taken
  const bool drawHitIndex   = debugColor && false;  // Draw the grid voxel index at the hit point
  const bool drawFirstStep  = debugColor && false;  // Draw the axis of the step taken by the first iteration

  normal = float3(0);

  // Initial boundig box hit test
  float3 orig, far, dir = r.dir;
  bool hit = intersectCube(r, bbox, orig, far);
  if(!hit) return -1;

  // Silhouette highlighter
  if(drawSilhouette && length(far-orig)<0.02){
    debug = scalarToColor(-1);
    return 1;
  }

  const int numVoxelsPerAxis = NUM_BRICKS_PER_AXIS;       // How many voxels are per axis the grid
  const float3 gridSize = bbox.bMax - bbox.bMin;          // World space grid size
  const int3 step = sign(dir);                            // How many indices to jump per axis traversal
  const float3 voxelDensity = numVoxelsPerAxis/gridSize;  // How many voxels per unit of world space
  const float3 voxelSize = gridSize/numVoxelsPerAxis;     // Size of a voxel
  const float3 delta = voxelSize/abs(dir);                // How much to travel in each direction till we meet the next voxel of that axis
  
  float3 hitCoorPos = (orig-bbox.bMin)*voxelDensity;
  float3 hitCoorNeg = max(bbox.bMax-orig,float3(0))*voxelDensity;
  float3 hitCoor = lerp(hitCoorNeg,hitCoorPos,step*0.5+0.5);

  int3 origIndex = min(int3(numVoxelsPerAxis-1),max(int3(0),int3(
    trunc(hitCoorPos)
  )));
  int3 index = origIndex;
  float3 dOffset = 1.0-fract(hitCoor);
  float3 iniDelta = delta * dOffset;
  float3 maxD = iniDelta;
  float3 lastTraversal;

  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    // Check current voxel for hit
    int pointer = clipMap[index];
    if(pointer < NUM_BRICKS_IN_ATLAS){
      // Debug visualizer of step count
      //if(drawSteps) debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
      if(drawSteps){
        if((pointer & DIRTY_BIT) != 0){
          debug = float3(1,1,0);
        }else{
          debug = scalarToColor(float(pointer)/float(NUM_BRICKS_IN_ATLAS*0.2));
        }
      }

      // Calculate normal based on last step
      normal = -step*lastTraversal;

      // Distance = size of step * numer of steps
      float3 deltas = voxelSize*abs(index-origIndex);
      return length(deltas*step + orig - r.orig);
    }
    
    if (maxD.x < maxD.y && maxD.x < maxD.z) {
      // X-axis traversal.
      maxD.x += delta.x;
      index.x += step.x;
      lastTraversal = float3(1,0,0);
    } else if (maxD.y < maxD.z) {
      // Y-axis traversal.
      maxD.y += delta.y;
      index.y += step.y;
      lastTraversal = float3(0,1,0);
    } else {
      // Z-axis traversal.
      maxD.z += delta.z;
      index.z += step.z;
      lastTraversal = float3(0,0,1);
    }

    // Debug visualizer of first step
    if(drawFirstStep){
      debug = lastTraversal;
      return 1;
    }

    if (any(index < 0) || any(index >= NUM_VOXELS_PER_AXIS+1)){
      // Outside grid
      if(drawSteps){
        debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
        return 1;
      }else{
        // No hit
        return -1;
      }
    }
  };

  // Over iterated 
  if(drawSteps){
    // Debug visualizer of step count
    debug = scalarToColor(1.1);
    return 1;
  }else{
    // No hit
    return -1;
  }
}


Ray get_ray(float2 uv){
  // Calculate offsets
  float2 ndc = uv * 2.0 - 1.0;
  float4 rayClip = float4(ndc.x, ndc.y, -1.0, 1.0);

  float4 view = mul(rayClip,sceneInfo.projMatrix);
  float3 world = mul(float4(normalize(view.xyz),0.0),sceneInfo.viewMatrix).xyz;
  float3 dir = normalize(world);

  Ray ray;
  ray.orig = sceneInfo.cameraPosition.xyz;
  ray.dir = dir;

  return ray;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID){
  float depth; float3 albedo, normal;

  float2 fragCoord = threadIdx.xy;
  uint2 iResolution;
  normalImage.GetDimensions(iResolution.x, iResolution.y);
  float2 uv = fragCoord/iResolution;
  
  Ray r = get_ray(uv);

  float3 p1, p2;
  float3 hitPoint;
  // Scene bounding box
  float bboxSize = AXIS_SIZES[CLIPMAP_LEVELS-1];
  float3 bboxPos = float3(0.0,0.0,0.0);
  Bbox bbox = unitBBox;
  bbox.bMin = bbox.bMin*bboxSize+bboxPos;
  bbox.bMax = bbox.bMax*bboxSize+bboxPos;
  
  //depth = rayMarchAmaWoo(r,bbox,normal);
  depth = traceScene(r,normal);

  bool hit = depth >= 0.0;
  if(hit){
    albedo = float3(1.0);
  }else{
    albedo = float3(0);
    debug = r.dir;
  }

  if(debugColor){
    albedo = debug;
  }else if(bboxColor){
    for(int i = 0; i<pushConst.numObjects; i++){
      const float eps = 0.05;
      float3 near, far, dir = r.dir;
      Bbox bboxObj = aabbs[i];
      bool hit = intersectCube(r, bboxObj, near, far);
      if(!hit) continue;
      float nearD = length(near-r.orig);
      if((depth < 0 || depth+eps > nearD) && length(near-far) < eps){
        albedo = scalarToColor(float(i)/float(pushConst.numObjects));
        depth = nearD;
      }
    }
  }

  albedoImage[int2(fragCoord)] = float4(albedo,1.0F);
  normalImage[int2(fragCoord)] = float4(normal, 0.0F);
  depthImage[int2(fragCoord)] = depth;
}
