#pragma once
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