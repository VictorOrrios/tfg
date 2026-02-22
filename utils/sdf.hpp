#pragma once

#include <glm/glm.hpp>

float opUnion(float a, float b);
float opUnion(float a, float b, float kk);

float opSubtraction(float a, float b);
float opSubtraction(float a, float b, float kk);

float opIntersection(float a, float b);
float opIntersection(float a, float b, float kk);

float opXor(float a, float b);

float opSmoothUnion(float a, float b, float k);

float opSmoothSubtraction(float a, float b, float k);

float opSmoothIntersection(float a, float b, float k);

glm::vec3 opNone(const glm::vec3 &p, const glm::vec3 &kk1, const glm::vec3 &kk2);

glm::vec3 opRepetition(const glm::vec3 &p, const glm::vec3 &spacing);
glm::vec3 opRepetition(const glm::vec3 &p, const glm::vec3 &spacing, const glm::vec3 &kk);

glm::vec3 opLimRepetition(const glm::vec3 &p, const glm::vec3 &spacing, const glm::vec3 &limit);

glm::vec3 opNone(const glm::vec3 &p, const glm::vec3 &kk);

glm::vec3 opTwist(const glm::vec3 &p, const glm::vec3 &defP);

glm::vec3 opBend(const glm::vec3 &p, const glm::vec3 &defP);

glm::vec3 opElongate(const glm::vec3 &p, const glm::vec3 &defP);

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