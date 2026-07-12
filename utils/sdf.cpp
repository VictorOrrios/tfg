#include "sdf.hpp"
#include "glm/common.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/matrix.hpp>
#include "../shaders/shaderio.h"

glm::vec3 hash3(glm::vec3 p){
  p = glm::fract(p * 0.1031f);
  p += glm::dot(p,glm::vec3(p.y,p.z,p.x) + 33.33f);
  return glm::fract((glm::vec3(p.x,p.x,p.y) + glm::vec3(p.y,p.z,p.z)) * glm::vec3(p.z,p.y,p.x));
}

//---------------------------------------
// Operations
//---------------------------------------

float opUnion(float a, float b){
  return glm::min(a,b);
}
float opSmoothUnion(float a, float b, float k){
  k *= 4.0;
  float h = glm::max(k-glm::abs(a-b),0.0f);
  return glm::min(a, b) - h*h*0.25/k;
}

float opSubtraction(float a, float b){
  return glm::max(-a,b);
}
float opSmoothSubtraction(float a, float b, float k){
  return -opSmoothUnion(a,-b,k);
}

float opIntersection(float a, float b){
  return glm::max(a,b);
}
float opSmoothIntersection(float a, float b, float k){
  return -opSmoothUnion(-a,-b,k);
}

float opXor(float a, float b){
  return glm::max(glm::min(a,b),-glm::max(a,b));
}

glm::vec3 opRepetition(glm::vec3 p, glm::vec3 spacing) {
  return p - spacing * round(p / glm::max(spacing, 0.00001f));
}
glm::vec3 opLimRepetition(glm::vec3 p, glm::vec3 spacing, glm::vec3 limit) {
  return p - spacing * glm::clamp(round(p / spacing),-limit, limit);
}

glm::vec3 opElongate(glm::vec3 p, glm::vec3 defP) {
  return p - glm::clamp(p, -defP, defP);
}

float smin(float a, float b, float k){
  float h = glm::max(k-glm::abs(a-b),0.0f);
  return glm::min(a, b) - h*h*0.25/k;
}

float smax(float a, float b, float k){
  float h = glm::max(k-glm::abs(a-b),0.0f);
  return glm::max(a, b) + h*h*0.25/k;
}

//---------------------------------------
// 3D SDF Primitives
//---------------------------------------

float sdSphere(glm::vec3 p, float s){
  return length(p) - s;
}
float sdSphere(glm::vec3 p){
  return length(p) - 0.5f;
}

float sdBox(glm::vec3 p, glm::vec3 b) {
  glm::vec3 q = glm::abs(p) - b;
  return length(glm::max(q, glm::vec3(0.0f))) +
         glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}
float sdBox(glm::vec3 p) {
  return sdBox(p,glm::vec3(0.5));
}

// Pre: n must be normalized
float sdPlane(glm::vec3 p, glm::vec3 n, float h ){
  return dot(p,n) + h;
}
float sdPlane(glm::vec3 p){
  return dot(p,glm::vec3(0,1,0));
}

float sdCapsule(glm::vec3 p, glm::vec3 a, glm::vec3 b, float r){
  glm::vec3 pa = p - a, ba = b - a;
  float h = glm::clamp( dot(pa,ba)/dot(ba,ba), 0.0f, 1.0f);
  return length( pa - ba*h ) - r;
}

float sdRoundedCylinder(glm::vec3 p, float ra, float rb, float h){
  glm::vec2 d = glm::vec2( length(glm::vec2(p.x,p.z))-ra+rb, glm::abs(p.y) - h + rb );
  return glm::min(glm::max(d.x,d.y),0.0f) + length(glm::max(d,0.0f)) - rb;
}

float sdTorus(glm::vec3 p, glm::vec2 t){
  glm::vec2 q = glm::vec2(length(glm::vec2(p.x,p.z))-t.x,p.y);
  return length(q)-t.y;
}
float sdTorus(glm::vec3 p){
  return sdTorus(p,glm::vec2(0.275,0.15));
}

float sdOctahedron(glm::vec3 p, float s){
  p = glm::abs(p);
  float m = p.x+p.y+p.z-s;
  glm::vec3 q;
  if( 3.0*p.x < m ) q = p;
  else if( 3.0*p.y < m ) q = glm::vec3(p.y,p.z,p.x);
  else if( 3.0*p.z < m ) q = glm::vec3(p.z,p.x,p.y);
  else return m*0.57735027;
    
  float k = glm::clamp(0.5f*(q.z-q.y+s),0.0f,s); 
  return length(glm::vec3(q.x,q.y-s+k,q.z-k)); 
}

float sdEmpty(glm::vec3 p) { return 1000000.0f; }

// Made by Victor :p
float sdSnowMan(glm::vec3 point){
  const float scale = 0.23f;
  const glm::vec3 pos = glm::vec3(0.0, -0.25, 0.0);
  glm::vec3 p = (point - pos) / scale;
  float r = sdSphere(p,1.0);
  r = opSmoothUnion(r,sdSphere(p-glm::vec3(0,1.5,0),0.6),0.1);
  r = opSmoothUnion(r,sdSphere(p-glm::vec3(0.3,1.6,0.5),0.1),0.01);
  r = opSmoothUnion(r,sdSphere(p-glm::vec3(-0.3,1.6,0.5),0.1),0.01);
  r = opSmoothUnion(r,sdCapsule(p,glm::vec3(0.0),glm::vec3(1.6,0.8,0.0),0.15),0.05);
  r = opSmoothUnion(r,sdCapsule(p,glm::vec3(0.0),glm::vec3(-1.6,0.8,0.0),0.15),0.05);
  r = opSmoothUnion(r,sdCapsule(p,glm::vec3(0.0,1.4,0.0),glm::vec3(0.0,1.3,0.8),0.05),0.01);
  r = opUnion(r,sdRoundedCylinder(p-glm::vec3(0.0,2.1,0.0),0.7,0.05,0.1));
  r = opUnion(r,sdRoundedCylinder(p-glm::vec3(0.0,2.5,0.0),0.4,0.05,0.5));
  return r * scale;
}

float sphSphere(glm::vec3 i,glm::vec3 f,glm::vec3 c){
  glm::vec3 p = 17.0f*glm::fract(hash3(i+c)+glm::vec3(0.11,0.17,0.13));
  float w = glm::fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
  float r = 0.7*w*w;
  return length(f-c) - r; 
}

float sphOctahedron(glm::vec3 i,glm::vec3 f,glm::vec3 c){
  glm::vec3 p = 17.0f*glm::fract(hash3(i+c)+glm::vec3(0.11,0.17,0.13));
  float w = glm::fract( p.x*p.y*p.z*(p.x+p.y+p.z) );
  float r = 0.7*w*w;
  return sdOctahedron(f-c,r);
}

float sdBaseSphere(glm::vec3 p){
  glm::ivec3 i = glm::ivec3(floor(p));
  glm::vec3 f = glm::fract(p);
  return glm::min(glm::min(glm::min(sphSphere(i,f,glm::ivec3(0,0,0)),
                    sphSphere(i,f,glm::ivec3(0,0,1))),
                glm::min(sphSphere(i,f,glm::ivec3(0,1,0)),
                    sphSphere(i,f,glm::ivec3(0,1,1)))),
            glm::min(glm::min(sphSphere(i,f,glm::ivec3(1,0,0)),
                    sphSphere(i,f,glm::ivec3(1,0,1))),
                glm::min(sphSphere(i,f,glm::ivec3(1,1,0)),
                    sphSphere(i,f,glm::ivec3(1,1,1)))));
}

float sdBaseOctahedron(glm::vec3 p){
  glm::ivec3 i = glm::ivec3(floor(p));
  glm::vec3 f = glm::fract(p);
  return glm::min(glm::min(glm::min(sphOctahedron(i,f,glm::ivec3(0,0,0)),
                    sphOctahedron(i,f,glm::ivec3(0,0,1))),
                glm::min(sphOctahedron(i,f,glm::ivec3(0,1,0)),
                    sphOctahedron(i,f,glm::ivec3(0,1,1)))),
            glm::min(glm::min(sphOctahedron(i,f,glm::ivec3(1,0,0)),
                    sphOctahedron(i,f,glm::ivec3(1,0,1))),
                glm::min(sphOctahedron(i,f,glm::ivec3(1,1,0)),
                    sphOctahedron(i,f,glm::ivec3(1,1,1)))));
}


//---------------------------------------
// Int to sdf op/primitive
//---------------------------------------

float evalPrimitive(int primType, glm::vec3 p){
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
    case 2: return opSmoothUnion(d,result,smoothness);
    case 3: return opSmoothSubtraction(d,result,smoothness);
  }
}

glm::vec3 applyRepOp(int opIndex, glm::vec3 p, glm::vec3 spacing, glm::ivec3 limit){
  switch(opIndex)
  {
    default:
    case 0: return p;
    case 1: return opLimRepetition(p,spacing,limit);
    case 2: return opRepetition(p,spacing);
  }
}

glm::vec3 applyDefOp(int opIndex, glm::vec3 p, glm::vec3 defP){
  switch(opIndex)
  {
    default:
    case 0: return p;
    case 1: return opElongate(p,defP);
  }
}

float applyMorphOp(glm::vec3 p, float prevPrim, int morphPrim, float morph, float roundness){
  float mD = evalPrimitive(morphPrim, p) - roundness;
  return glm::mix(prevPrim,mD,morph);
}

float applyTerrainOp(glm::vec3 p, float d, int octaves, glm::vec4 terrain, float minD){
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
    glm::mat3 kernel = glm::mat3( 
        0.00f,-1.60f,-1.20f,
        1.60f,  0.72f, -0.96f,
        1.20f, -0.96f,  1.28f 
    );
    p = kernel*p;
    s = terrain.y*s;

    if(s<minD) break;
  }
  return d;
}
