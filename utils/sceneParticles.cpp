
#include <omp.h>
#include "nvutils/logger.hpp"
#include "scene.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

static float frac(float x) { return x - floorf(x); }

static glm::vec3 particle_freqs(int i, float speed) {
  return speed * glm::vec3(0.5f + frac(i * 0.6180339887f),
                           0.5f + frac(i * 0.7548776662f),
                           0.5f + frac(i * 0.5698402910f));
}

static glm::vec3 particle_phases(int i) {
  return 2.0f * float(M_PI) *
         glm::vec3(frac(i * 0.3819660113f), frac(i * 0.2451223338f),
                   frac(i * 0.4301597090f));
}

static glm::vec3 particle_axis(int i)
{
    glm::vec3 a(
        frac(i * 0.6180339887f) - 0.5f,
        frac(i * 0.5698402910f) - 0.5f,
        frac(i * 0.2451223338f) - 0.5f);

    float len = glm::length(a);

    if (len < 1e-4f)
        return glm::vec3(0.f, 1.f, 0.f);

    return a / len;
}

glm::quat particleRotation(int index, float time, float speed)
{
    glm::vec3 axis = particle_axis(index);
    glm::vec3 phi = particle_phases(index);

    float angle = 0.7f * speed * time + phi.x;

    return glm::angleAxis(angle, axis);
}

glm::vec3 particlePosition(int index, float time, float speed,
                           const glm::vec3 &domainMin,
                           const glm::vec3 &domainMax) {
  glm::vec3 w = particle_freqs(index, speed);
  glm::vec3 phi = particle_phases(index);

  glm::vec3 u(cosf(w.x * time + phi.x), cosf(w.y * time + phi.y),
              cosf(w.z * time + phi.z));

  glm::vec3 extent = domainMax - domainMin;

  return domainMin + (u * 0.5f + 0.5f) * extent;
}

void Scene::animateParticles(float time){
  const float speed = 1.0;
  const glm::vec3 domainMin = glm::vec3(-1.0);
  const glm::vec3 domainMax = glm::vec3(1.0);
  LOGI("Anim: %f\n",time);  

  #pragma omp parallel for schedule(static)
  for(int i = 0; i<m_root.size(); i++){
    Node& n = m_root[i];
    n.gp.position = particlePosition(i,time,speed,domainMin,domainMax);
    n.gp.rotation = particleRotation(i, time, speed);
    updateNodeData(&n);
  }
}