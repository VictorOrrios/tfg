#pragma once
//---------------------------------------------------------
// Definition and functions of Axis aligned Bounding Boxes
//---------------------------------------------------------

#include "common.slang"

// Axis aligned Bounding Box 
struct Bbox{
  float3 bMin;
  float3 bMax;
};

// BBox woth 1x1x1 units in size with the center at 0,0,0
const static Bbox unitBBox =  {float3(-0.5, -0.5, -0.5),float3(0.5, 0.5, 0.5)};

// Rturns if a points fits inside a bbox
bool insideBbox(float3 p, Bbox bbox){
  return all(p >= bbox.bMin) && all(p <= bbox.bMax);
}

// Taken from vk_mini_samples texture_3d
// Determines whether a ray intersects with a bounding box (bbox) representing a cube.
// If an intersection occurs, it calculates the two points of intersection and returns
// them as output. If the ray origin is within the box p1 will be the same as the ray origin.
// The function returns a boolean value indicating whether an intersection took place.
bool intersectCube(in Ray ray, in Bbox bbox, out float3 p1, out float3 p2){
  p1 = float3(0.0, 0.0, 0.0);
  p2 = float3(0.0, 0.0, 0.0);

  float3 invDir = 1.0 / ray.dir;

  float3 tMin = (bbox.bMin - ray.orig) * invDir;
  float3 tMax = (bbox.bMax - ray.orig) * invDir;

  float3 t1 = min(tMin, tMax);
  float3 t2 = max(tMin, tMax);

  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar  = min(min(t2.x, t2.y), t2.z);

  if(tNear > tFar || tFar < 0.0){
    return false;
  }

  p1 = ray.orig + ray.dir * max(tNear, 0.0);
  p2 = ray.orig + ray.dir * tFar;

  return true;
}