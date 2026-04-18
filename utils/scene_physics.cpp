#include "scene.hpp"

void integrate(Scene::Node* n, float dt, glm::vec3 gravity){
  Scene::GeneralParams& gp = n->gp;
  Scene::PhysicsParams& pyp = n->pyp;

  if(pyp.invMass <= 0.0)
    return;

  // Linear motion
  gp.prev_position = gp.position;
  pyp.vel += gravity * dt;
  gp.position += pyp.vel * dt;

  // Angular motion
  gp.prev_rotation = gp.rotation;
  

}

void Scene::updateNodePysicsData(Node *n) {
  GeneralParams& gp = n->gp;
  PhysicsParams& pyp = n->pyp;

  if(pyp.physicsActive){
    // Sphere
    float mass = 4.0 / 3.0 * std::numbers::pi * gp.scale * gp.scale * gp.scale * pyp.density;
    pyp.invMass = 1/mass;
    float I = 2.0 / 5.0 * mass * gp.scale * gp.scale;
    float I_inv = 1.0/I;
    pyp.invInertia = glm::vec3(I_inv,I_inv,I_inv);
  }else{
    gp.prev_position = gp.position;
    gp.prev_rotation = gp.rotation;
  }
}

/*
XPBD

simulate(∆𝑡):
  ∆𝑡𝑠 ← ∆𝑡/𝑛
  for 𝑛 substeps
    for all particles 𝑖
      𝐯𝑖 ← 𝐯𝑖 + ∆𝑡𝑠𝐠
      𝐩𝑖 ← 𝐱𝑖
      𝐱𝑖 ← 𝐱𝑖 + ∆𝑡𝑠𝐯𝑖
    for all constraints 𝐶
      solve(𝐶, ∆𝑡𝑠)
    for all particles 𝑖
      𝐯𝑖 ← (𝐱𝑖 − 𝐩𝑖)/∆𝑡𝑠

solve(𝐶, ∆𝑡):
  for all particles 𝑖 of 𝐶
    compute ∆𝐱𝑖
    𝐱𝑖 ← 𝐱𝑖 + ∆𝐱𝑖
*/
void Scene::simulate(float dt){


  const glm::vec3 velocity(0.1,0.0,0.0);
  glm::vec3 dx = velocity*dt;
  m_root[1].gp.position += dx;
  updateNodeData(&m_root[1]);
}