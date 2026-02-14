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

#include "shaderio.h"

[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

[[vk::binding(BindingPoints::normalBuffer)]] RWTexture2D<float4> normalImage;
[[vk::binding(BindingPoints::albedoBuffer)]] RWTexture2D<float4> albedoImage;
[[vk::binding(BindingPoints::depthBuffer)]] RWTexture2D<float> depthImage;
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;
[[vk::binding(BindingPoints::globalGrid)]] RWTexture3D<float> globalGrid;

struct Ray {
  float3 orig;
  float3 dir;
};

// Taken from vk_mini_samples texture_3d
struct Bbox{
  float3 bMin;
  float3 bMax;
};
const static Bbox unitBBox =  {float3(-0.5, -0.5, -0.5),float3(0.5, 0.5, 0.5)};
// Determines whether a ray intersects with a bounding box (bbox) representing a cube.
// If an intersection occurs, it calculates the two points of intersection and returns
// them as output.
// The function returns a boolean value indicating whether an intersection took place.
bool intersectCube(in Ray ray, in Bbox bbox, out float3 p1, out float3 p2)
{
  p1 = float3(0.0, 0.0, 0.0);
  p2 = float3(0.0, 0.0, 0.0);

  float3 invDir = 1.0 / ray.dir;

  float3 tMin = (bbox.bMin - ray.orig) * invDir;
  float3 tMax = (bbox.bMax - ray.orig) * invDir;

  float3 t1 = min(tMin, tMax);
  float3 t2 = max(tMin, tMax);

  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar  = min(min(t2.x, t2.y), t2.z);


  if(tNear > tFar || tFar < 0.0)
  {
    return false;
  }

  p1 = ray.orig + ray.dir * max(tNear, 0.0);
  p2 = ray.orig + ray.dir * tFar;

  return true;
}


//---------------------------------------
// Operations from Iñigo Quilez
//---------------------------------------

float opUnion(float a, float b){
    return min(a,b);
}
float opSubtraction(float a, float b){
    return max(-a,b);
}
float opIntersection(float a, float b){
    return max(a,b);
}
float opXor(float a, float b){
    return max(min(a,b),-max(a,b));
}
float opSmoothUnion(float a, float b, float k){
    k *= 4.0;
    float h = max(k-abs(a-b),0.0);
    return min(a, b) - h*h*0.25/k;
}
float opSmoothSubtraction(float a, float b, float k){
    return -opSmoothUnion(a,-b,k);

    // k *= 4.0;
    // float h = max(k-abs(-a-b),0.0);
    // return max(-a, b) + h*h*0.25/k;
}
float opSmoothIntersection(float a, float b, float k){
    return -opSmoothUnion(-a,-b,k);

    // k *= 4.0;
    // float h = max(k-abs(a-b),0.0);
    // return max(a, b) + h*h*0.25/k;
}

//---------------------------------------
// 3D SDF Primitives from Iñigo Quilez
//---------------------------------------

float sdSphere(float3 p, float s){
  return length(p) - s;
}
// Pre: n must be normalized
float sdPlane(float3 p, float3 n, float h ){
  return dot(p,n) + h;
}
float sdCapsule(float3 p, float3 a, float3 b, float r){
  float3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}
float sdRoundedCylinder(float3 p, float ra, float rb, float h){
  float2 d = float2( length(p.xz)-ra+rb, abs(p.y) - h + rb );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float sdSnowMan(float3 p){
  float r = sdSphere(p,1.0);
  r = opSmoothUnion(r,sdSphere(p-float3(0,1.5,0),0.6),0.1);
  r = opSmoothUnion(r,sdSphere(p-float3(0.3,1.6,0.5),0.1),0.01);
  r = opSmoothUnion(r,sdSphere(p-float3(-0.3,1.6,0.5),0.1),0.01);
  r = opSmoothUnion(r,sdCapsule(p,float3(0.0),float3(1.6,0.8,0.0),0.15),0.05);
  r = opSmoothUnion(r,sdCapsule(p,float3(0.0),float3(-1.6,0.8,0.0),0.15),0.05);
  r = opSmoothUnion(r,sdCapsule(p,float3(0.0,1.4,0.0),float3(0.0,1.3,0.8),0.05),0.01);
  r = opUnion(r,sdRoundedCylinder(p-float3(0.0,2.1,0.0),0.7,0.05,0.1));
  r = opUnion(r,sdRoundedCylinder(p-float3(0.0,2.5,0.0),0.4,0.05,0.5));
  return r;
}

float map(float3 p){
  float t = sin(pushConst.time)*0.5+0.5;
  float r = 100000.0;
  r = min(r,lerp(sdSnowMan(p),sdSphere(p,1.0),t*t));
  r = min(r,sdPlane(p,float3(0,1,0),1.0));
  return r;
}

// True numerical
float3 calcNormalSlow(float3 p){
  const float eps = 0.0001;
  const float2 h = float2(eps,0.0);
  return normalize(float3(
    map(p+h.xyy)-map(p-h.xyy),
    map(p+h.yxy)-map(p-h.yxy),
    map(p+h.yyx)-map(p-h.yyx)
  ));
}

// Tetrahedron aproximation
float3 calcNormal(float3 p){
  const float h = 0.0001;
  const float2 k = float2(1,-1);
  return normalize(
    k.xyy*map(p + k.xyy*h) +
    k.yyx*map(p + k.yyx*h) + 
    k.yxy*map(p + k.yxy*h) + 
    k.xxx*map(p + k.xxx*h)
  );
}

float sphereTrace(Ray ray){
  const float MAX_DIST = 100.0;
  const float MIN_DIST = 0.0001;
  
  float depth = 0.0;
  
  for(int i = 0; i < 500; i++){
      float3 p = ray.orig + ray.dir * depth;
      float dist = map(p);
      
      if(abs(dist) < MIN_DIST){
          return depth;  // Hit
      }
      
      if(depth > MAX_DIST){
          return -1.0;  // No hit
      }
      
      depth += dist;
  }
  
  return -1.0;
}

// PRE: r.dir must be normalized
float rayMarchAmaWoo(Ray r, Bbox bbox, out int3 index){
  const int NUM_VOXELS_PER_AXIS = 100;
  const float MIN_SDF_DIST = 0.0001;
  const int MAX_ITERATIONS = 1000;

  // Initial boundig box hit test
  float3 orig, far, dir = r.dir;
  bool hit = intersectCube(r, bbox, orig, far);
  if(!hit) return -1;

  const float3 gridSize = bbox.bMax - bbox.bMin;              // World space grid size
  const int3 step = sign(dir);                                // How many indices to jump per axis traversal
  const float3 delta = gridSize/abs(dir)/NUM_VOXELS_PER_AXIS; // How much to travel in each direction till we meet the next voxel of that axis
  
  float3 hitCoorPos = (orig-bbox.bMin)*NUM_VOXELS_PER_AXIS/gridSize;
  float3 hitCoorNeg = max(bbox.bMax-orig,float3(0))*NUM_VOXELS_PER_AXIS/gridSize;
  float3 hitCoor = lerp(hitCoorNeg,hitCoorPos,step*0.5+0.5);

  index = min(int3(NUM_VOXELS_PER_AXIS-1),max(int3(0),int3(
    trunc(hitCoorPos)
  )));
  float3 dOffset = 1.0-fract(hitCoor);
  float3 maxD = delta * dOffset;

  // Debug visualizers
  // if(length(far-orig)<0.01) return float3(1.0);      // Silhouette highlighter
  // return float3(index)/float3(NUM_VOXELS_PER_AXIS);  // Index UVS 


  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    // Check current voxel for hit
    if(globalGrid[index] < MIN_SDF_DIST){
      // TODO: Figure out a way to interpolate with neighbouring voxels
      // TODO: This depth is wrong
      return length(orig - r.orig) + min(maxD.x,min(maxD.y,maxD.z));
    }

    if (maxD.x < maxD.y && maxD.x < maxD.z) {
      // X-axis traversal.
      index.x += step.x;
      maxD.x += delta.x;
    } else if (maxD.y < maxD.z) {
      // Y-axis traversal.
      index.y += step.y;
      maxD.y += delta.y;
    } else {
      // Z-axis traversal.
      index.z += step.z;
      maxD.z += delta.z;
    }

    // Outside grid
    if (any(index < 0) || any(index >= NUM_VOXELS_PER_AXIS)) return -1;
  };

  return -1; // Over iterated  
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
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float depth; float3 albedo, normal;

  float2 fragCoord = threadIdx.xy;
  uint2 iResolution;
  normalImage.GetDimensions(iResolution.x, iResolution.y);
  float2 uv = fragCoord/iResolution;
  
  Ray r = get_ray(uv);

/* 
  // Sphere tracing with sdf calculations
  depth = sphereTrace(r);
  float3 p = r.orig+r.dir*depth;
  normal = calcNormal(p);

  float3 groundColor = float3(0.294, 0.49, 0.024);
  float3 snowColor = float3(0.8F);
  albedo = p.y > -0.95F? snowColor : groundColor;
 */

  float3 p1, p2;
  float3 hitPoint;
  // Scene bounding box
  float bboxSize = 1.0;
  float3 bboxPos = float3(0.0,0.0,0.0);
  Bbox bbox = unitBBox;
  bbox.bMin = bbox.bMin*bboxSize+bboxPos;
  bbox.bMax = bbox.bMax*bboxSize+bboxPos;
  float3 p;
  int3 index;
  depth = rayMarchAmaWoo(r,bbox,index);

  bool hit = depth > 0.0;

  if(hit){
    albedo = float3(1.0);
    normal = normalize(float3(0.2,0.2,1.0));
  }else{
    albedo = float3(r.dir);
  }


/* 
  float read = globalGrid[int3((int)(uv.x*100.0),(int)(uv.y*100.0),50)].r;
  if(read > 0.001){
    albedo = float3(1.0f,0,0);
    depth = 1.0;
    normal = normalize(float3(0.2,0.2,1.0));
  }
 */
  normalImage[int2(fragCoord)] = float4(normal, 0.0F);
  albedoImage[int2(fragCoord)] = float4(albedo,1.0F);
  depthImage[int2(fragCoord)] = depth;
}
