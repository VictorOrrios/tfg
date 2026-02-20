#pragma once

#include <glm/glm.hpp>

float opUnion(float a, float b);

float opSubtraction(float a, float b);

float opIntersection(float a, float b);

float opXor(float a, float b);

float opSmoothUnion(float a, float b, float k);

float opSmoothSubtraction(float a, float b, float k);

float opSmoothIntersection(float a, float b, float k);

float sdSphere(const glm::vec3 &p, float s);
float sdSphere(const glm::vec3 &p);

float sdBox(const glm::vec3 &p, const glm::vec3 b);
float sdBox(const glm::vec3 &p);

float sdEmpty(const glm::vec3 &p);

float sdPlane(const glm::vec3 &p, const glm::vec3 &n, float h);

float sdCapsule(const glm::vec3 &p, const glm::vec3 &a,
                       const glm::vec3 &b, float r);

float sdRoundedCylinder(const glm::vec3 &p, float ra, float rb,float h);

float sdSnowMan(const glm::vec3 &p);