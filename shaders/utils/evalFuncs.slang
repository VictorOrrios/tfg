#pragma once

#include "../shaderio.h"  // Shared definitions with CPU
#include "common.slang"   // Shared definitions with shaders
#include "sdf.slang"      // SDF functions, operations and primitives
#include "palettes.slang" // Debug color palette


// Evaluates the whole sdf scene in a point in space
float map(float3 p){
  const float maxD = BRICK_SIZES[CLIPMAP_LEVELS-1];
  float mapValue = map(p,CLIPMAP_LEVELS-1);
  return min(mapValue,maxD);
}

// Evaluates a dynamic node in world spcace
float evalDynamic(float3 point, DynamicObject dyn){
  float3 p = world2Local(dyn,point)/dyn.scale;
  return evalPrimitive(dyn.type, p)*dyn.scale;
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

float3 evalNormalTrueStatic(float3 p, float nearRange){
  const float eps = 0.0001;
  const float2 h = float2(eps,0.0);
  return normalize(float3(
    mapStatic(p+h.xyy,nearRange)-mapStatic(p-h.xyy,nearRange),
    mapStatic(p+h.yxy,nearRange)-mapStatic(p-h.yxy,nearRange),
    mapStatic(p+h.yyx,nearRange)-mapStatic(p-h.yyx,nearRange)
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

float3 evalNormalStatic(float3 p,float nearRange){
  const float h = 0.0001;
  const float2 k = float2(1,-1);
  return normalize(
    k.xyy*mapStatic(p + k.xyy*h,nearRange) +
    k.yyx*mapStatic(p + k.yyx*h,nearRange) + 
    k.yxy*mapStatic(p + k.yxy*h,nearRange) + 
    k.xxx*mapStatic(p + k.xxx*h,nearRange)
  );
}

float3 evalNormalDynamic(float3 p, DynamicObject dyn){
  float3 p = world2Local(dyn,p)/dyn.scale;
  return normalize(rotateByQuat(dyn.rotation,evalPrimitiveNormal(dyn.type, p)));
}

float3 evalGradStatic(float3 p, float h, float nearRange){
  const float2 k = float2(1,-1);
  return (
    k.xyy*mapStatic(p + k.xyy*h,nearRange) +
    k.yyx*mapStatic(p + k.yyx*h,nearRange) + 
    k.yxy*mapStatic(p + k.yxy*h,nearRange) + 
    k.xxx*mapStatic(p + k.xxx*h,nearRange)
  )/(2*h);
}

// Sphere traces the scene evaluating the scene sdf at every step
// Pre: r.orig must be inside bbox
float evalTraceMap(Ray ray, float maxDepth){
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

float evalShadow(float3 p, float k){
  const int3 camIdN = sceneInfo.cameraId0.xyz >> (CLIPMAP_LEVELS-1);
  const float3 camIdNPos = camIdN * BRICK_SIZES[CLIPMAP_LEVELS-1];
  const float3 worldMin = camIdNPos - AXIS_SIZES[CLIPMAP_LEVELS-1]/2;
  const float3 worldMax = camIdNPos + AXIS_SIZES[CLIPMAP_LEVELS-1]/2;

  float shadow = 1.0f;
  float depth = 0.01;

  Ray r;
  r.orig = p;
  r.dir = pushConst.lp.lightDir;

  float3 near,far;
  Bbox bbox = Bbox(worldMin,worldMax);
  if(!intersectCube(r, bbox, near, far)) return 0.0;
  float maxDepth = length(r.orig-far);

  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS * int(CLIPMAP_LEVELS*0.5);
  const float MIN_DIST = 0.0001;
  
  for(int i = 0; i < MAX_ITERATIONS; i++){
    float3 p = r.orig + r.dir * depth;
    float dist = map(p);    

    if(abs(dist) < MIN_DIST){
      shadow = 0.0;
      break;  // Hit
    }
    
    if(depth >= maxDepth) break;  // No hit

    shadow = min(shadow,k*dist/depth);
    depth += dist;
  }
  
  return shadow;
}

Material evalMat(float3 p, float3 n){
  Material mat;
  map(p,mat);
  
  if(mat.type == MaterialType::Debug){
    const float debug_scale = 1.0;
    float debug_value = mapDebugColor(p,BRICK_SIZES[CLIPMAP_LEVELS-1])*debug_scale;
    mat.albedo_shininess.xyz = scalarToStepedColor(debug_value);
  }else if(mat.type == MaterialType::Terrain){
    float slope = 1.0-dot(n,float3(0,1,0));
    mat.albedo_shininess.xyz = getTerrainColor(p,slope);
  }

  return mat;
}