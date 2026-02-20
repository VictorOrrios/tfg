#include "sdf.hpp"
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

float opUnion(float a, float b) { return glm::min(a, b); }

float opSubtraction(float a, float b) { return glm::max(-a, b); }

float opIntersection(float a, float b) { return glm::max(a, b); }

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

float sdSphere(const glm::vec3 &p, float s) { return glm::length(p) - s; }
float sdSphere(const glm::vec3 &p) { return glm::length(p) - 1.0f; }

float sdBox(const glm::vec3 &p, const glm::vec3 b) {
  glm::vec3 q = abs(p) - b;
  return length(glm::max(q, glm::vec3(0.0f))) +
         glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

float sdBox(const glm::vec3 &p) {
  glm::vec3 q = abs(p) - glm::vec3(1.0f);
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
  glm::vec3 p = point / 0.15f;
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
  return r * 0.15f;
}