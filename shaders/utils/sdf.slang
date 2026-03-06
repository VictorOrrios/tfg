#pragma once
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