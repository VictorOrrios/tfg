#include "sdf.hpp"
#include "glm/common.hpp"
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

float opUnion(float a, float b) { return glm::min(a, b); }
float opUnion(float a, float b, float kk) { return opUnion(a,b); }

float opSubtraction(float a, float b) { return glm::max(-a, b); }
float opSubtraction(float a, float b, float kk) { return opSubtraction(a,b); }

float opIntersection(float a, float b) { return glm::max(a, b); }
float opIntersection(float a, float b, float kk) { return opIntersection(a,b); }

float opXor(float a, float b) {
  return glm::max(glm::min(a, b), -glm::max(a, b));
}

float opSmoothUnion(float a, float b, float k) {
  k *= 4.0f;
  float h = glm::max(k - glm::abs(a - b), 0.0f);
  return glm::min(a, b) - h * h * 0.25f / k;
}

float opSmoothSubtraction(float a, float b, float k) {
  return -opSmoothUnion(a, -b, k);
}

float opSmoothIntersection(float a, float b, float k) {
  return -opSmoothUnion(-a, -b, k);
}

glm::vec3 opNone(const glm::vec3 &p, const glm::vec3 &kk1, const glm::vec3 &kk2){
  return p;
}

glm::vec3 opRepetition(const glm::vec3 &p, const glm::vec3 &spacing){
  return p - spacing * glm::round(p / spacing);
}
glm::vec3 opRepetition(const glm::vec3 &p, const glm::vec3 &spacing, const glm::vec3 &kk){
  return opRepetition(p,spacing);
}

glm::vec3 opLimRepetition(const glm::vec3 &p, const glm::vec3 &spacing, const glm::vec3 &limit){
  return p - spacing * glm::clamp(glm::round(p / spacing),
                                         glm::vec3(-limit),
                                         glm::vec3(limit));
}

glm::vec3 opNone(const glm::vec3 &p, const glm::vec3 &kk){
  return p;
}

// Modified to make it anisotropic
glm::vec3 opTwist(const glm::vec3 &p, const glm::vec3 &defP){
  glm::vec3 axis = glm::normalize(defP);
  float k = glm::length(defP);

  float angle = k * glm::dot(p, axis);

  float c = cos(angle);
  float s = sin(angle);

  glm::mat3 R = glm::mat3(
    c + axis.x * axis.x * (1 - c),
    axis.x * axis.y * (1 - c) - axis.z * s,
    axis.x * axis.z * (1 - c) + axis.y * s,

    axis.y * axis.x * (1 - c) + axis.z * s,
    c + axis.y * axis.y * (1 - c),
    axis.y * axis.z * (1 - c) - axis.x * s,

    axis.z * axis.x * (1 - c) - axis.y * s,
    axis.z * axis.y * (1 - c) + axis.x * s,
    c + axis.z * axis.z * (1 - c)
  );

  return R * p;
}

// Modified to make it anisotropic
glm::vec3 opBend(const glm::vec3 &p, const glm::vec3 &defP){
  glm::vec3 axis = glm::normalize(glm::vec3(defP.y,defP.z,defP.x));
  float k = glm::length(defP);

  float angle = k * glm::dot(p, axis);

  float c = cos(angle);
  float s = sin(angle);

  // yzx
  glm::mat3 R = glm::mat3(
    axis.y * axis.x * (1 - c) + axis.z * s,
    c + axis.y * axis.y * (1 - c),
    axis.y * axis.z * (1 - c) - axis.x * s,

    axis.z * axis.x * (1 - c) + axis.y * s,
    axis.z * axis.y * (1 - c) + axis.x * s,
    c + axis.z * axis.z * (1 - c),

    c + axis.x * axis.x * (1 - c),
    axis.x * axis.y * (1 - c) - axis.z * s,
    axis.x * axis.z * (1 - c) + axis.y * s
  );

  glm::vec3 r = R * p;
  return glm::vec3(r.y,r.z,r.x);
}

glm::vec3 opElongate(const glm::vec3 &p, const glm::vec3 &defP){
  return p - glm::clamp(p,-defP, defP); 
}


float sdSphere(const glm::vec3 &p, float s) { return glm::length(p) - s; }
float sdSphere(const glm::vec3 &p) { return glm::length(p) - 0.5f; }

float sdBox(const glm::vec3 &p, const glm::vec3 b) {
  glm::vec3 q = abs(p) - b;
  return length(glm::max(q, glm::vec3(0.0f))) +
         glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

float sdBox(const glm::vec3 &p) {
  glm::vec3 q = abs(p) - glm::vec3(0.5f);
  return length(glm::max(q, glm::vec3(0.0f))) +
         glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

float sdEmpty(const glm::vec3 &p) {
  return 1000000.0f;
}

float sdPlane(const glm::vec3 &p, const glm::vec3 &n, float h) {
  return glm::dot(p, n) + h;
}

float sdCapsule(const glm::vec3 &p, const glm::vec3 &a, const glm::vec3 &b,
                float r) {
  glm::vec3 pa = p - a;
  glm::vec3 ba = b - a;
  float h = glm::clamp(glm::dot(pa, ba) / glm::dot(ba, ba), 0.0f, 1.0f);
  return glm::length(pa - ba * h) - r;
}

float sdRoundedCylinder(const glm::vec3 &p, float ra, float rb, float h) {
  glm::vec2 d(glm::length(glm::vec2(p.x, p.z)) - ra + rb,
              glm::abs(p.y) - h + rb);
  return glm::min(glm::max(d.x, d.y), 0.0f) +
         glm::length(glm::max(d, glm::vec2(0.0f))) - rb;
}

float sdSnowMan(const glm::vec3 &point) {
  const float scale = 0.23f;
  const glm::vec3 pos = glm::vec3(0.0,-0.25,0.0);
  glm::vec3 p = (point-pos) / scale;
  float r = sdSphere(p, 1.0f);
  r = opSmoothUnion(r, sdSphere(p - glm::vec3(0, 1.5f, 0), 0.6f), 0.1f);

  r = opSmoothUnion(r, sdSphere(p - glm::vec3(0.3f, 1.6f, 0.5f), 0.1f), 0.01f);
  r = opSmoothUnion(r, sdSphere(p - glm::vec3(-0.3f, 1.6f, 0.5f), 0.1f), 0.01f);
  r = opSmoothUnion(
      r, sdCapsule(p, glm::vec3(0.0f), glm::vec3(1.6f, 0.8f, 0.0f), 0.15f),
      0.05f);
  r = opSmoothUnion(
      r, sdCapsule(p, glm::vec3(0.0f), glm::vec3(-1.6f, 0.8f, 0.0f), 0.15f),
      0.05f);
  r = opSmoothUnion(r,
                    sdCapsule(p, glm::vec3(0.0f, 1.4f, 0.0f),
                              glm::vec3(0.0f, 1.3f, 0.8f), 0.05f),
                    0.01f);
  r = opUnion(
      r, sdRoundedCylinder(p - glm::vec3(0.0f, 2.1f, 0.0f), 0.7f, 0.05f, 0.1f));
  r = opUnion(
      r, sdRoundedCylinder(p - glm::vec3(0.0f, 2.5f, 0.0f), 0.4f, 0.05f, 0.5f));
  return r * scale;
}