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

const static int NUM_VOXELS_PER_AXIS = 99; // Numer of voxel2 per axis of globalGrids
[[vk::binding(BindingPoints::globalGrid)]] RWTexture3D<float> globalGrid;

#ifdef NDEBUG
const static bool debugColor = false;
#else
const static bool debugColor = pushConst.debug.mode == 1;
#endif

static float3 debug = float3(0.0);

float3 scalarToColor(float x){
  return scalarToColor(x, pushConst.debug.palette);
}

// Evaluates the whole sdf scene in a point in space
float map(float3 p){
  float t = sin(pushConst.time)*0.5+0.5;
  float r = 100000.0;
  r = min(r,lerp(sdSnowMan(p),sdSphere(p,1.0),t*t));
  r = min(r,sdPlane(p,float3(0,1,0),1.0));
  return r;
}

// True numerical
float3 calcNormalMapSlow(float3 p){
  const float eps = 0.0001;
  const float2 h = float2(eps,0.0);
  return normalize(float3(
    map(p+h.xyy)-map(p-h.xyy),
    map(p+h.yxy)-map(p-h.yxy),
    map(p+h.yyx)-map(p-h.yyx)
  ));
}

// Tetrahedron aproximation
float3 calcNormalMap(float3 p){
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
float sphereTraceMap(Ray ray){
  const int MAX_ITERATIONS = 100;
  const float MAX_DIST = 100.0;
  const float MIN_DIST = 0.0001;
  
  float depth = 0.0;
  
  for(int i = 0; i < MAX_ITERATIONS; i++){
    float3 p = ray.orig + ray.dir * depth;
    float dist = map(p);
    
    if(abs(dist) < MIN_DIST) return depth;  // Hit
    
    if(depth > MAX_DIST) return -1.0;  // No hit

    depth += dist;
  }
  
  return -1.0;
}



// Samples a 2x2x2 voxel (voxel2) and outputs values into two float4 vectors:
// sBottom: r=s000 g=100 b=101 a=001
// sTop:    r=s010 g=110 b=111 a=011
// Left hand coordinate system (x=right, y=up, z=forwards)
// TODO: This looks slow, might be interesting to make more efficient
void sampleVoxel2(int3 index, out float4 sBottom, out float4 sTop){
  sBottom = float4(
    globalGrid[index + int3(0,0,0)],  // s000
    globalGrid[index + int3(1,0,0)],  // s100
    globalGrid[index + int3(1,0,1)],  // s101
    globalGrid[index + int3(0,0,1)]   // s001
  );

  sTop = float4(
    globalGrid[index + int3(0,1,0)],  // s010
    globalGrid[index + int3(1,1,0)],  // s110
    globalGrid[index + int3(1,1,1)],  // s111
    globalGrid[index + int3(0,1,1)]   // s011
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
  sampleVoxel2(index, sBottom,sTop);
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

// TODO: Fix X axis artifacts
float sphereTraceGrid(Ray r, Bbox bbox, out float3 normal){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS;

  // Debug visualizers
  const bool drawSilhouette = debugColor && true;  // Draw the silhouette of the grid
  const bool drawSteps      = debugColor && true;   // Draw the numer of steps taken
  const bool drawHitIndex   = debugColor && false;  // Draw the grid voxel index at the hit point

  normal = float3(0);

  // Initial boundig box hit test
  float3 p, far, dir = r.dir;
  bool hit = intersectCube(r, bbox, p, far);
  if(!hit) return -1;

  // Silhouette highlighter
  if(drawSilhouette && length(far-p)<0.02){
    debug = scalarToColor(-1);
    return 1;
  }

  const float3 gridSize = bbox.bMax - bbox.bMin;            // World space grid size
  const float3 voxelSize = gridSize/NUM_VOXELS_PER_AXIS;    // Size of a voxel
  const float3 voxelDensity = NUM_VOXELS_PER_AXIS/gridSize; // How many voxels per unit of world space
  
  float3 hitCoor, voxelCoor;
  int3 currentIndex, lastIndex;
  float4 coeff, sBottom, sTop;

  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    hitCoor = (p-bbox.bMin)*voxelDensity; // P coordinates relative to grid, in [0,1]³

    // TODO: Check if it needs clamping
    currentIndex = int3(trunc(hitCoor));  // Grid index of p
    voxelCoor = frac(hitCoor);            // P Coordinates relative to voxel, in [0,1]³

    // If this voxel hasn't been visited yet
    if(steps == 1 || any(currentIndex != lastIndex)){
      // Update current voxel data and index
      sampleVoxel2(currentIndex,sBottom,sTop);
      //getkConstants(sBottom,sTop,k0123,k4567);
      lastIndex = currentIndex;
    }

    // Interpolate current sdf value of the voxel2
    //float t = triLerp(voxelCoor,sBottom,sTop);
    float t = triLerp(voxelCoor,sBottom, sTop);


    // Cutoff condition: A distance of 1/10 voxels
    // TODO: Fine tune to better performance/quality
    if(t < voxelSize.x/10){

      // Debug visualizer of step count
      if(drawSteps) debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
      // Debug visualizer of hit grid index
      if(drawHitIndex) debug = float3(currentIndex)/float3(NUM_VOXELS_PER_AXIS);
      
      normal = calcNormalAnalytic(voxelCoor,sBottom,sTop);
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

// Intersects 
// Pre: r.orig must be in canonical voxel space [0,1]³
//      r.dir must be normalized
//      index must be in range off grid
bool intersectVoxel2(Ray r, int3 index){
  // TODO
  return false;
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

  const float3 gridSize = bbox.bMax - bbox.bMin;                // World space grid size
  const int3 step = sign(dir);                                  // How many indices to jump per axis traversal
  const float3 voxelDensity = (NUM_VOXELS_PER_AXIS+1)/gridSize; // How many voxels per unit of world space
  const float3 voxelSize = gridSize/(NUM_VOXELS_PER_AXIS+1);    // Size of a voxel
  const float3 delta = voxelSize/abs(dir);                      // How much to travel in each direction till we meet the next voxel of that axis
  
  float3 hitCoorPos = (orig-bbox.bMin)*voxelDensity;
  float3 hitCoorNeg = max(bbox.bMax-orig,float3(0))*voxelDensity;
  float3 hitCoor = lerp(hitCoorNeg,hitCoorPos,step*0.5+0.5);

  int3 origIndex = min(int3(NUM_VOXELS_PER_AXIS),max(int3(0),int3(
    trunc(hitCoorPos)
  )));
  int3 index = origIndex;
  float3 dOffset = 1.0-fract(hitCoor);
  float3 iniDelta = delta * dOffset;
  float3 maxD = iniDelta;
  float3 lastTraversal;

  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    // Check current voxel for hit
    if(globalGrid[index] < voxelSize.x/10){

      // Debug visualizer of step count
      if(drawSteps) debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
      
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
  ray.orig = sceneInfo.cameraPosition;
  ray.dir = dir;

  return ray;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
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
  float bboxSize = 1.5;
  float3 bboxPos = float3(0.0,0.0,0.0);
  Bbox bbox = unitBBox;
  bbox.bMin = bbox.bMin*bboxSize+bboxPos;
  bbox.bMax = bbox.bMax*bboxSize+bboxPos;
  
  //depth = rayMarchAmaWoo(r,bbox,normal);
  depth = sphereTraceGrid(r,bbox,normal);

  bool hit = depth >= 0.0;
  if(hit){
    albedo = float3(1.0);
  }else{
    albedo = float3(0);
    debug = r.dir;
  }

  if(debugColor){
    albedo = debug;
  }

  normalImage[int2(fragCoord)] = float4(normal, 0.0F);
  albedoImage[int2(fragCoord)] = float4(albedo,1.0F);
  depthImage[int2(fragCoord)] = depth;
}
