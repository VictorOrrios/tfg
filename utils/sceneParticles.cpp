
#include "nvutils/logger.hpp"
#include "scene.hpp"
#include "rng.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <numbers>
#include <omp.h>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;

static float frac(float x) { return x - floorf(x); }

static glm::vec3 particle_freqs(int i, float speed) {
  return speed * glm::vec3(0.5f + frac(i * 0.6180339887f),
                           0.5f + frac(i * 0.7548776662f),
                           0.5f + frac(i * 0.5698402910f));
}

static glm::vec3 particle_phases(int i) {
  return 2.0f * std::numbers::pi_v<float> *
         glm::vec3(frac(i * 0.3819660113f), frac(i * 0.2451223338f),
                   frac(i * 0.4301597090f));
}

static glm::vec3 particle_axis(int i) {
  glm::vec3 a(frac(i * 0.6180339887f) - 0.5f, frac(i * 0.5698402910f) - 0.5f,
              frac(i * 0.2451223338f) - 0.5f);

  float len = glm::length(a);

  if (len < 1e-4f)
    return glm::vec3(0.f, 1.f, 0.f);

  return a / len;
}

glm::quat particleRotation(int index, float time, float speed) {
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

void Scene::dynamicTestUpdate(float time) {
  static float timeLast = 0;
  float delta = time - timeLast;
  timeLast = time;
  static float result = 0.0;

  static ofstream output("output.txt");
  
  static int phase = -1;
  static int maxPhase = 4;
  
  static int iniPower = 4;
  static int maxPower = 11;
  static int currPower = maxPower;
  static int maxSamples = 140;
  static int currSamples = maxSamples+1;

  const std::string phases []= {
    "boxes","cylinders","mixed","spheres", "spheres_blobby"
  };  

  // If ended
  if(phase>maxPhase) return;
  if(time<1.0) return;

  if(currSamples >= maxSamples){
    // Record test
    if(phase>=0){
      output << phases[phase]<<" "<<(1<<currPower)<<" "<<(result/maxSamples*1000.0)<<endl;
    }

    // Advance power
    currPower++;
    currSamples = 0;
    result = 0.0;

    // Advance phase
    if(currPower>maxPower){
      phase += 1;
      currPower = iniPower;

      if(phase>maxPhase){
        LOGI("TEST DONE");
        output.close();
      }
    }

    // Update scene
    setSceneObjects(phase,1<<currPower);

  }else{
    // Record sample
    result+=delta;
    currSamples++;
  }

  animateParticles(time+10.0);
}

void Scene::setSceneObjects(int preset, int numOfParticles) {
  LOGI("Testing %d objects\n",numOfParticles);
  shaderio::PrimType type = shaderio::PrimType::Box;
  switch (preset) {
    case 0: type = shaderio::PrimType::Box; break;
    case 1: type = shaderio::PrimType::Cylinder; break;
    case 2: type = shaderio::PrimType::Box; break;
    case 3: case 4: 
      type = shaderio::PrimType::Sphere; break;
  }

  // Delete old
  for(auto& node:m_root)
    m_removeList.push_back(node.gp.bbox);
  m_root.clear();
  m_root.reserve(numOfParticles);
  m_selected = -1;

  // Create new
  for(int i = 0; i<numOfParticles; i++){
    if(preset == 2){
      float r = randomFloat1();
      shaderio::PrimType types[] = {shaderio::PrimType::Box,shaderio::PrimType::Cone, shaderio::PrimType::Cylinder, shaderio::PrimType::Sphere, shaderio::PrimType::Torus};
      type = types[int(r*5.0)];
    }
    Node *particle = createNode(type);
    particle->gp.scale = 0.2;
    particle->gp.position = glm::vec3(0.0);
    particle->gp.rotation = glm::vec3(0.0);
    if(preset != 4){
      particle->sdp.combOp = (int)CombinationOp::Union;
      particle->sdp.smoothness = 0.0;
    }else{
      particle->sdp.combOp = (int)CombinationOp::Union + 2;
      particle->sdp.smoothness = 0.02;
    }
    particle->gp.mat = 0;
    updateNodeData(particle);
    addNode(particle);
  }

  m_selected = -1;
}

void Scene::animateParticles(float time) {
  const float speed = 1.0;
  const glm::vec3 domainMin = glm::vec3(-1.0);
  const glm::vec3 domainMax = glm::vec3(1.0);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < m_root.size(); i++) {
    Node &n = m_root[i];
    n.gp.position = particlePosition(i, time, speed, domainMin, domainMax);
    n.gp.rotation = particleRotation(i, time, speed);
    updateNodeData(&n);
  }
}