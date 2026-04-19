#include "glm/geometric.hpp"
#include "glm/matrix.hpp"
#include "scene.hpp"

void integrate(Scene::Node* n, float dt, glm::vec3 gravity){
  Scene::GeneralParams& gp = n->gp;
  Scene::PhysicsParams& pyp = n->pyp;

  if(!pyp.physicsActive)
    return;

  // Linear motion
  pyp.prev_position = gp.position;
  pyp.vel += gravity * dt;
  gp.position += pyp.vel * dt;

  // Angular motion
  pyp.prev_rotation = gp.rotation;
  glm::quat d_rot = glm::quat(
    pyp.omega.x,
    pyp.omega.y,
    pyp.omega.z,
    0
  );
  d_rot *= gp.rotation;
  gp.rotation += 0.5f * dt * d_rot;
  gp.rotation = glm::normalize(gp.rotation);
  pyp.inv_rotation = glm::inverse(gp.rotation);
}

void updateVelocities(Scene::Node* n, float dt){
  Scene::GeneralParams& gp = n->gp;
  Scene::PhysicsParams& pyp = n->pyp;

  if(!pyp.physicsActive)
    return;

  // Linear motion
  pyp.vel = (gp.position - pyp.prev_position)/dt;

  // Angular motion
  glm::quat pre_rot_inv = glm::inverse(pyp.prev_rotation);
  glm::quat d_rot = gp.rotation * pyp.prev_rotation;
  pyp.omega = glm::vec3(
    d_rot.x * 2.0 / dt,
    d_rot.y * 2.0 / dt,
    d_rot.z * 2.0 / dt
  );
  if(d_rot.w < 0.0)
    pyp.omega *= -1;
  
  // Apply dampening to velocity here
}

float getInverseMass(Scene::Node* n, glm::vec3 normal, glm::vec3 pos){
  Scene::GeneralParams& gp = n->gp;
  Scene::PhysicsParams& pyp = n->pyp;

  if(!pyp.physicsActive)
    return 0.0;

  glm::vec3 rn = pos - gp.position;
  rn = glm::cross(rn, normal);
  rn = pyp.inv_rotation * rn;

  float w = pyp.inv_mass +
    rn.x * rn.x * pyp.inv_inertia.x + 
    rn.y * rn.y * pyp.inv_inertia.y + 
    rn.z * rn.z * pyp.inv_inertia.z;

  return w;  
}

void applyCorrection(Scene::Node* n, glm::vec3 corr, glm::vec3 pos){
  Scene::GeneralParams& gp = n->gp;
  Scene::PhysicsParams& pyp = n->pyp;

  if(!pyp.physicsActive)
    return;

  // Linear
  gp.position += corr * pyp.inv_mass;

  // Angualar
  glm::vec3 dOmega = pos - gp.position;
  dOmega = glm::cross(dOmega, corr);
  dOmega = pyp.inv_rotation * dOmega;
  dOmega *= pyp.inv_inertia;
  dOmega = gp.rotation * dOmega;

  glm::quat d_rot = glm::quat(
    dOmega.x,
    dOmega.y,
    dOmega.z,
    0.0
  );

  d_rot *= gp.rotation;
  gp.rotation += 0.5f * d_rot;
  gp.rotation = glm::normalize(gp.rotation);
  pyp.inv_rotation = glm::inverse(gp.rotation);
}

float applyCorrection(Scene::Node* n, Scene::Node* other_n, float dt, float compliance, glm::vec3 corr, glm::vec3 pos, glm::vec3 other_pos){
  Scene::GeneralParams& gp = n->gp;
  Scene::PhysicsParams& pyp = n->pyp;
  Scene::GeneralParams& gpO = other_n->gp;
  Scene::PhysicsParams& pypO = other_n->pyp;

  float C = glm::length(corr);

  if(C == 0.0)
    return 0.0;

  float dt_m2 = 1.0f / (dt*dt);
  glm::vec3 normal = glm::normalize(corr);

  float w = getInverseMass(n, normal, pos) + getInverseMass(other_n, normal, other_pos);

  if(w == 0.0)
    return 0;

  // XPBD
  float alpha = compliance * dt_m2;
  float lambda = -C / (w+alpha);
  normal *= -lambda;

  applyCorrection(n,normal,pos);
  applyCorrection(other_n,-normal,other_pos);
  
  return lambda * dt_m2;
}

void solveDistanceConstrain(Scene::Node *na, Scene::Node *nb, float distance, float compliance, float dt){
  Scene::GeneralParams& gpA = na->gp;
  Scene::PhysicsParams& pypA = na->pyp;
  Scene::GeneralParams& gpB = nb->gp;
  Scene::PhysicsParams& pypB = nb->pyp;
  
  glm::vec3 corr = gpB.position - gpA.position;
  float curr_dist = glm::length(corr);
  corr = glm::normalize(corr);
  float force = applyCorrection(na,nb,dt,compliance,corr,gpA.position,gpB.position);

}

void Scene::updateNodePysicsData(Node *n) {
  GeneralParams& gp = n->gp;
  PhysicsParams& pyp = n->pyp;

  if(pyp.physicsActive){
    // Sphere
    float mass = 4.0 / 3.0 * std::numbers::pi * gp.scale * gp.scale * gp.scale * pyp.density;
    pyp.inv_mass = 1/mass;
    float I = 2.0 / 5.0 * mass * gp.scale * gp.scale;
    float I_inv = 1.0/I;
    pyp.inv_inertia = glm::vec3(I_inv,I_inv,I_inv);
    pyp.inv_rotation = glm::inverse(gp.rotation);
  }else{
    pyp.vel = glm::vec3(0.0);
    pyp.prev_position = gp.position;
    pyp.prev_rotation = gp.rotation;
    pyp.inv_mass = 0.0;
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
  float dts = dt/SIM_NUM_SUBSTEPS;

  for(int sub_step = 0; sub_step < SIM_NUM_SUBSTEPS; sub_step++){
    for(int i = 0; i<m_root.size(); i++)
      integrate(&m_root[i], dt, m_gravity);

    for(int i = 0; i<m_root.size(); i++)
      updateVelocities(&m_root[i], dt);

  }

  for(int i = 0; i<m_root.size(); i++){
    Node& n = m_root[i];
    if(n.pyp.physicsActive)
      updateNodeData(&n);
  }
}