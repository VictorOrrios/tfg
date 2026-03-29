#pragma once
#include "bbox.slang"

//------------------------------------------
// Signed distance functions and operations
//------------------------------------------
// Most of the code is from Iñigo Quilez (https://iquilezles.org/)

//---------------------------------------
// Operations
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
}
float opSmoothIntersection(float a, float b, float k){
  return -opSmoothUnion(-a,-b,k);
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

uint pcgHash(uint x) {
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return x;
}

uint hash(uint3 p) {
    uint h = p.x * 1664525u + p.y * 1013904223u + p.z * 374761393u;
    return pcgHash(h);
}

float hashf(uint3 p) {
    return float(hash(p)) / 4294967295.0;
}

float smin( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return min(a, b) - h*h*0.25/k;
}

float smax( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}

float sph(float3 i,float3 f,float3 c){
    // random radius at grid vertex i+c (please replace this hash by
    // something better if you plan to use this for a real application)
    float3  p = 17.0*fract( (i+c)*0.3183099+float3(0.11,0.17,0.13) );
    float w = fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
    float r = 0.7*w*w;
    // distance to sphere at grid vertex i+c
    return length(f-c) - r; 
}

float sdBase(float3 p){
  int3 i = int3(floor(p));
  float3 f = fract(p);
  // distance to the 8 corners spheres
  return min(min(min(sph(i,f,int3(0,0,0)),
                    sph(i,f,int3(0,0,1))),
                min(sph(i,f,int3(0,1,0)),
                    sph(i,f,int3(0,1,1)))),
            min(min(sph(i,f,int3(1,0,0)),
                    sph(i,f,int3(1,0,1))),
                min(sph(i,f,int3(1,1,0)),
                    sph(i,f,int3(1,1,1)))));
}



float sdFbm(float3 p, float d, TerrainParams tp){
  float s = tp.sIni;
  for(int i=0; i<tp.numOctaves; i++){
    // evaluate new octave
    float n = s*sdBase(p);

    // add
    n = smax(n,d-tp.smaxDecay*s,tp.smaxIncrease*s);
    d = smin(n,d      ,tp.sminIncrease*s);

    // prepare next octave
    p = mul(float3x3( 0.00, 1.60, 1.20,
            -1.60, 0.72,-0.96,
            -1.20,-0.96, 1.28 ),p);
    s = tp.sDecay*s;
  }
  return d;
}

//---------------------------------------
// Int to sdf op/primitive
//---------------------------------------

float evalPrimitive(int primType, float3 p, TerrainParams tp){
  switch(primType)
  {
    default:
    case 0: return sdEmpty(p);
    case 1: return sdBox(p);
    case 2: return sdSphere(p);
    case 3: return sdTorus(p);
    case 4: return sdSnowMan(p);
    case 5: return sdFbm(p,sdPlane(p,float3(0,1,0),1.0),tp);
    //case 5: return sdPlane(p,float3(0,1,0),1.0);
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

float3 applyRepOp(int opIndex, float3 p, float3 spacing, int3 limit){
  switch(opIndex)
  {
    default:
    case 0: return p;
    case 1: return opLimRepetition(p,spacing,limit);
    case 2: return opRepetition(p,spacing);
  }
}

float3 applyDefOp(int opIndex, float3 p, float3 defP){
  switch(opIndex)
  {
    default:
    case 0: return p;
    case 1: return opElongate(p,defP);
  }
}


//---------------------------------------
// Scene evaluation function
//---------------------------------------
float map(float3 point, float nearRange, int numObjects, StructuredBuffer<Bbox> aabbs, StructuredBuffer<SceneObject> objects, TerrainParams tp){
  const float iniD = 10000.0f;
  float result = iniD;

  for(int obIdx = 0; obIdx < numObjects; obIdx++) {
    float3 p = float3(point);
    Bbox bbox = aabbs[obIdx];

    // If not inside bbox primitive continue with next 
    if(!nearBbox(p,bbox,nearRange))
      continue;

    SceneObject params = objects[obIdx];

    p = mul(params.tInv, float4(p, 1.0)).xyz;

    p = applyRepOp(params.repOp, p, params.spacing.xyz, params.limit.xyz);

    p = applyDefOp(params.defOp, p, params.defP.xyz);

    p /= params.scale;

    float d = evalPrimitive(params.type, p, tp) - params.roundness;

    d *= params.scale;

    result = evalCombOp(params.combOp, d, result, params.smoothness);
  }

  return result;
}
