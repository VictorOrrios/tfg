#pragma once

#include <glm/glm.hpp>

float evalPrimitive(int primType, glm::vec3 p);

float evalCombOp(int opIndex, float d, float result, float smoothness);

glm::vec3 applyRepOp(int opIndex, glm::vec3 p, glm::vec3 spacing, glm::ivec3 limit);

glm::vec3 applyDefOp(int opIndex, glm::vec3 p, glm::vec3 defP);

float applyMorphOp(glm::vec3 p, float prevPrim, int morphPrim, float morph, float roundness);

float applyTerrainOp(glm::vec3 p, float d, int octaves, glm::vec4 terrain, float minD);