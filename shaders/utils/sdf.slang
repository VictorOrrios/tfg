// * Most of the code is from Iñigo Quilez (https://iquilezles.org/)

#pragma once

#include "../shaderio.h"      // Shared definitions with CPU
#include "bbox.slang"         // AABB helpers
#include "descriptors.slang"  // External memory definitions
#include "materials.slang"    // material functions
#include "rng.slang"          // Random numbers and hashes

//---------------------------------------
// Operations
//---------------------------------------

float opUnion(float a, float b){
  return min(a,b);
}
float opUnionMat(float a, float b, Material matA, inout Material matB){
  if(a<b) matB = matA;
  return min(a,b);
}
float opSmoothUnion(float a, float b, float k){
  k *= 4.0;
  float h = max(k-abs(a-b),0.0);
  return min(a, b) - h*h*0.25/k;
}
float opSmoothUnionMat(float a, float b, float k, Material matA, inout Material matB){
  float blend = a/(a+b);
  matB = lerpMaterial(matA,matB,blend);
  k *= 4.0;
  float h = max(k-abs(a-b),0.0);
  return min(a, b) - h*h*0.25/k;
}

float opSubtraction(float a, float b){
  return max(-a,b);
}
float opSmoothSubtraction(float a, float b, float k){
  return -opSmoothUnion(a,-b,k);
}

float opIntersection(float a, float b){
  return max(a,b);
}
float opSmoothIntersection(float a, float b, float k){
  return -opSmoothUnion(-a,-b,k);
}

float opXor(float a, float b){
  return max(min(a,b),-max(a,b));
}

float3 opRepetition(float3 p, float3 spacing) {
  return p - spacing * round(p / max(spacing, 0.00001f));
}
float3 opLimRepetition(float3 p, float3 spacing, float3 limit) {
  return p - spacing * clamp(round(p / spacing),-limit, limit);
}

float3 opElongate(float3 p, float3 defP) {
  return p - clamp(p, -defP, defP);
}

float smin(float a, float b, float k){
  float h = max(k-abs(a-b),0.0);
  return min(a, b) - h*h*0.25/k;
}

float smax(float a, float b, float k){
  float h = max(k-abs(a-b),0.0);
  return max(a, b) + h*h*0.25/k;
}

//---------------------------------------
// 3D SDF Primitives
//---------------------------------------

float sdSphere(float3 p, float s){
  return length(p) - s;
}
float sdSphere(float3 p){
  return length(p) - 0.5f;
}

float sdBox(float3 p, float3 b) {
  float3 q = abs(p) - b;
  return length(max(q, float3(0.0f))) +
         min(max(q.x, max(q.y, q.z)), 0.0f);
}
float sdBox(float3 p) {
  return sdBox(p,float3(0.5));
}

// Pre: n must be normalized
float sdPlane(float3 p, float3 n, float h ){
  return dot(p,n) + h;
}
float sdPlane(float3 p){
  return dot(p,float3(0,1,0));
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

float sdTorus(float3 p, float2 t){
  float2 q = float2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float sdTorus(float3 p){
  return sdTorus(p,float2(0.275,0.15));
}

float sdHexPrism(float3 p, float2 h){
  const float3 k = float3(-0.8660254, 0.5, 0.57735);
  p = abs(p);
  p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
  float2 d = float2(
       length(p.xy-float2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdOctahedron(float3 p, float s){
  p = abs(p);
  float m = p.x+p.y+p.z-s;
  float3 q;
       if( 3.0*p.x < m ) q = p.xyz;
  else if( 3.0*p.y < m ) q = p.yzx;
  else if( 3.0*p.z < m ) q = p.zxy;
  else return m*0.57735027;
    
  float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
  return length(float3(q.x,q.y-s+k,q.z-k)); 
}

float sdEmpty(float3 p) { return 1000000.0f; }

// Made by Victor :p
float sdSnowMan(float3 point){
  const float scale = 0.23f;
  const float3 pos = float3(0.0, -0.25, 0.0);
  float3 p = (point - pos) / scale;
  float r = sdSphere(p,1.0);
  r = opSmoothUnion(r,sdSphere(p-float3(0,1.5,0),0.6),0.1);
  r = opSmoothUnion(r,sdSphere(p-float3(0.3,1.6,0.5),0.1),0.01);
  r = opSmoothUnion(r,sdSphere(p-float3(-0.3,1.6,0.5),0.1),0.01);
  r = opSmoothUnion(r,sdCapsule(p,float3(0.0),float3(1.6,0.8,0.0),0.15),0.05);
  r = opSmoothUnion(r,sdCapsule(p,float3(0.0),float3(-1.6,0.8,0.0),0.15),0.05);
  r = opSmoothUnion(r,sdCapsule(p,float3(0.0,1.4,0.0),float3(0.0,1.3,0.8),0.05),0.01);
  r = opUnion(r,sdRoundedCylinder(p-float3(0.0,2.1,0.0),0.7,0.05,0.1));
  r = opUnion(r,sdRoundedCylinder(p-float3(0.0,2.5,0.0),0.4,0.05,0.5));
  return r * scale;
}

float sphSphere(float3 i,float3 f,float3 c){
  float3 p = 17.0*fract(hash3(i+c)+float3(0.11,0.17,0.13));
  float w = fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
  float r = 0.7*w*w;
  return length(f-c) - r; 
}

float sphOctahedron(float3 i,float3 f,float3 c){
  float3 p = 17.0*fract(hash3(i+c)+float3(0.11,0.17,0.13));
  float w = fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
  float r = 0.7*w*w;
  return sdOctahedron(f-c,r);
}

float sdBaseSphere(float3 p){
  int3 i = int3(floor(p));
  float3 f = fract(p);
  return min(min(min(sphSphere(i,f,int3(0,0,0)),
                    sphSphere(i,f,int3(0,0,1))),
                min(sphSphere(i,f,int3(0,1,0)),
                    sphSphere(i,f,int3(0,1,1)))),
            min(min(sphSphere(i,f,int3(1,0,0)),
                    sphSphere(i,f,int3(1,0,1))),
                min(sphSphere(i,f,int3(1,1,0)),
                    sphSphere(i,f,int3(1,1,1)))));
}

float sdBaseOctahedron(float3 p){
  int3 i = int3(floor(p));
  float3 f = fract(p);
  return min(min(min(sphOctahedron(i,f,int3(0,0,0)),
                    sphOctahedron(i,f,int3(0,0,1))),
                min(sphOctahedron(i,f,int3(0,1,0)),
                    sphOctahedron(i,f,int3(0,1,1)))),
            min(min(sphOctahedron(i,f,int3(1,0,0)),
                    sphOctahedron(i,f,int3(1,0,1))),
                min(sphOctahedron(i,f,int3(1,1,0)),
                    sphOctahedron(i,f,int3(1,1,1)))));
}


//---------------------------------------
// Int to sdf op/primitive
//---------------------------------------

float evalPrimitive(int primType, float3 p){
  switch(primType)
  {
    default:
    case 0: return sdEmpty(p);
    case 1: return sdBox(p);
    case 2: return sdSphere(p);
    case 3: return sdTorus(p);
    case 4: return sdSnowMan(p);
    case 5: return sdPlane(p);
  }
}

float evalCombOp(int opIndex, float d, float result, float smoothness){
  switch(opIndex)
  {
    default:
    case 0: return opUnion(d,result);
    case 1: return opSubtraction(d,result);
    case 2: return opIntersection(d,result);
    case 3: return opSmoothUnion(d,result,smoothness);
    case 4: return opSmoothSubtraction(d,result,smoothness);
    case 5: return opSmoothIntersection(d,result,smoothness);
  }
}
float evalCombOpMat(int opIndex, float d, float result, float smoothness, Material matA, inout Material matB){
  switch(opIndex)
  {
    default:
    case 0: return opUnionMat(d,result,matA,matB);
    case 1: return opSubtraction(d,result);
    case 2: return opIntersection(d,result);
    case 3: return opSmoothUnionMat(d,result,smoothness,matA,matB);
    case 4: return opSmoothSubtraction(d,result,smoothness);
    case 5: return opSmoothIntersection(d,result,smoothness);
  }
}

float3 applyRepOp(int opIndex, float3 p, float3 spacing, int3 limit){
  switch(opIndex)
  {
    default:
    case 0: return p;
    case 1: return opLimRepetition(p,spacing,limit);
    case 2: return opRepetition(p,spacing);
  }
}

float3 applyDefOp(int opIndex, float3 p, float4 defP){
  switch(opIndex)
  {
    default:
    case 0: return p;
    case 1: return opElongate(p,defP.xyz);
  }
}

float applyMorphOp(float3 p, float prevPrim, int morphPrim, float morph, float roundness){
  float mD = evalPrimitive(morphPrim, p) - roundness;
  return lerp(prevPrim,mD,morph);
}

float applyTerrainOp(float3 p, float d, int octaves, float4 terrain, float minD){
  float s = terrain.x, n;
  for(int i=0; i<octaves; i++){

    if(i==0){
      n = s*sdBaseOctahedron(p);
      n = smax(n, d - terrain.z * s*2, terrain.w * s);
      d = smin(n, d,               terrain.w/4.0 * s);
    }else{
      n = s*sdBaseSphere(p);
      n = smax(n, d - terrain.z * s, terrain.w * s);
      d = smin(n, d,               terrain.w * s);
    }

    // prepare next octave
    p = mul(float3x3( 0.00, 1.60, 1.20,
            -1.60, 0.72,-0.96,
            -1.20,-0.96, 1.28 ),p);
    s = terrain.y*s;

    if(s<minD) break;
  }
  return d;
}


//---------------------------------------
// Scene evaluation function
//---------------------------------------
float map(float3 point, int level){
  const float iniD = 10000.0f;
  const float nearRange = MAX_BRICK_VALUES[level];
  float result = iniD;

  for(int obIdx = 0; obIdx < pushConst.numObjects; obIdx++) {
    float3 p = float3(point);
    Bbox bbox = aabbs[obIdx];

    // If not inside bbox primitive continue with next 
    if(!nearBbox(p,bbox,nearRange))
      continue;

    SceneObject params = objects[obIdx];

    p = mul(params.tInv, float4(p, 1.0)).xyz;

    p = applyRepOp(params.repOp, p, params.spacing.xyz, params.limit_octaves.xyz);

    p = applyDefOp(params.defOp, p, params.defP);

    p /= params.scale;

    float d = evalPrimitive(params.type, p) - params.roundness;

    d = d>0.0 ? applyTerrainOp(p, d, params.limit_octaves.w, params.terrain, VOXEL_SIZES[0]/10.0): d;

    d = params.morph>0.0 ? applyMorphOp(p,d,params.morphPrim,params.morph,params.roundness) : d;

    d *= params.scale;

    result = evalCombOp(params.combOp, d, result, params.smoothness);
  }

  return result;
}

// Map function for getting the material at point
float map(float3 point, out Material matResult){
  const float iniD = 1e5;
  float result = iniD;

  int firstInsideIdx = -1;
  int secondInsideIdx = -1;

  // Prepass for single primitive near p
  for(int obIdx = 0; obIdx < pushConst.numObjects; obIdx++) {
    Bbox bbox = aabbs[obIdx];
    if(!insideBbox(point,bbox))
      continue;

    if(firstInsideIdx == -1){
      firstInsideIdx = obIdx;
      SceneObject params = objects[obIdx];
      matResult = materials[params.mat];
    }else{
      secondInsideIdx = obIdx;
      break;
    }
  }

  
  if(secondInsideIdx != -1){
    for(int obIdx = firstInsideIdx; obIdx < pushConst.numObjects && result > VOXEL_SIZES[0]/10.0f; obIdx++) {
      float3 p = float3(point);
      Bbox bbox = aabbs[obIdx];

      // If not inside bbox primitive continue with next 
      if(!insideBbox(p,bbox))
        continue;

      SceneObject params = objects[obIdx];

      p = mul(params.tInv, float4(p, 1.0)).xyz;

      p = applyRepOp(params.repOp, p, params.spacing.xyz, params.limit_octaves.xyz);

      p = applyDefOp(params.defOp, p, params.defP);

      p /= params.scale;

      float d = evalPrimitive(params.type, p) - params.roundness;

      d = params.morph>0.0 ? applyMorphOp(p,d,params.morphPrim,params.morph,params.roundness) : d;

      d = d>0.0 ? applyTerrainOp(p, d, params.limit_octaves.w, params.terrain, VOXEL_SIZES[0]/10.0): d;

      d *= params.scale;

      result = evalCombOpMat(params.combOp, d, result, params.smoothness,materials[params.mat],matResult);
    }
  }

  return result;
}