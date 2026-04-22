#include "glm/geometric.hpp"
#include "scene.hpp"
#include "nvutils/logger.hpp"
#include <numbers>

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
    0,
    pyp.omega.x,
    pyp.omega.y,
    pyp.omega.z
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
  glm::quat d_rot = gp.rotation * pre_rot_inv;
  pyp.omega = glm::vec3(
    d_rot.x * 2.0 / dt,
    d_rot.y * 2.0 / dt,
    d_rot.z * 2.0 / dt
  );
  if(d_rot.w < 0.0)
    pyp.omega *= -1;
/* 
  LOGI("v %f,%f,%f omega %f,%f,%f d_rot %f,%f,%f,%f rot %f,%f,%f,%f pre_rot_inv %f,%f,%f,%f prev_rotation %f,%f,%f,%f\n",
    pyp.vel.x,pyp.vel.y,pyp.vel.z,
    pyp.omega.x,pyp.omega.y,pyp.omega.z,
    d_rot.x,d_rot.y,d_rot.z,d_rot.w,
    gp.rotation.x,gp.rotation.y,gp.rotation.z,gp.rotation.w,
    pre_rot_inv.x,pre_rot_inv.y,pre_rot_inv.z,pre_rot_inv.w,
    pyp.prev_rotation.x,pyp.prev_rotation.y,pyp.prev_rotation.z,pyp.prev_rotation.w
  );
   */
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
/* 
  LOGI("rn %f,%f,%f inv_mass %f\n",
    rn.x,rn.y,rn.z,
    pyp.inv_mass
  );
 */
  return w;  
}

void applyCorrection(Scene::Node* n, glm::vec3 corr, glm::vec3 point){
  Scene::GeneralParams& gp = n->gp;
  Scene::PhysicsParams& pyp = n->pyp;

  if(!pyp.physicsActive)
    return;

  // Linear
  gp.position += corr * pyp.inv_mass;

  // Angular
  glm::vec3 dOmega = point - gp.position;
  dOmega = glm::cross(dOmega, corr);
  dOmega = pyp.inv_rotation * dOmega;
  dOmega *= pyp.inv_inertia;
  dOmega = gp.rotation * dOmega;

  glm::quat d_rot = glm::quat(
    0.0,
    dOmega.x,
    dOmega.y,
    dOmega.z  
  );

  d_rot *= gp.rotation;
  gp.rotation += 0.5f * d_rot;
  gp.rotation = glm::normalize(gp.rotation);
  pyp.inv_rotation = glm::inverse(gp.rotation);
}

float applyCorrection(Scene::Node* na, Scene::Node* nb, float dt, float compliance, glm::vec3 corr, glm::vec3 pointA, glm::vec3 pointB){
  Scene::GeneralParams& gpa = na->gp;
  Scene::PhysicsParams& pypa = na->pyp;

  float C = glm::length(corr);

  if(C == 0.0)
    return 0.0;

  float dt_m2 = 1.0f / (dt*dt);
  glm::vec3 normal = glm::normalize(corr);

  float w = getInverseMass(na, normal, pointA);
  if(nb != nullptr)
    w += getInverseMass(nb, normal, pointB);
  //LOGI("w %f\n",w);

  if(w == 0.0)
    return 0;

  // XPBD
  float alpha = compliance * dt_m2;
  float lambda = -C / (w+alpha);
  normal *= -lambda;

  applyCorrection(na,normal,pointA);
  if(nb != nullptr){
    normal *= -1;
    applyCorrection(nb,normal,pointB);
  }

  
  return lambda * dt_m2;
}

glm::vec3 local2World(Scene::Node *n, glm::vec3 p){
  return (n->gp.rotation * p) + n->gp.position;
}

glm::vec3 world2Local(Scene::Node *n, glm::vec3 p){
  return n->pyp.inv_rotation * (p - n->gp.position);
}

void solveDistanceConstrain(Scene::Node *n, glm::vec3 attach_point ,glm::vec3 fixed_pos, float distance, float compliance, bool push, bool pull, float dt){
  Scene::GeneralParams& gp = n->gp;
  Scene::PhysicsParams& pyp = n->pyp;

  glm::vec3 attach_point_world = local2World(n,attach_point);
  
  glm::vec3 corr = fixed_pos - attach_point_world;
  float curr_dist = glm::length(corr);

  if((!push && curr_dist <= distance) ||
     (!pull && curr_dist >= distance))
    return;

  corr = glm::normalize(corr);
  corr *= curr_dist - distance;
  float force = applyCorrection(
    n,nullptr,
    dt,compliance,corr,
    attach_point_world,fixed_pos);
  /* 
  LOGI("curr_dist %f force %f corr %f,%f,%f new pos %f,%f,%f\n",
    curr_dist,force,
    corr.x,corr.y,corr.z,
    gpA.position.x,gpA.position.y,gpA.position.z
  );
 */
}

void Scene::solveCollisionConstraint(int nodeIdx, float compliance, float dt){
  Node& n = m_root[nodeIdx];
  Scene::GeneralParams& gp = n.gp;
  Scene::PhysicsParams& pyp = n.pyp;
  
  float sdValue = map(gp.position, nodeIdx);
  float radius = gp.scale/2.0;
  float C = radius - sdValue;
  
  if(C <= 0)
    return;

  glm::vec3 normal = evalNormal(gp.position,nodeIdx); 
  glm::vec3 corr = normal*C;

  float force = applyCorrection(
    &n,nullptr,
    dt,compliance,corr,
    gp.position,glm::vec3(0.0));
  
}

void Scene::updateNodePysicsData(Node *n) {
  GeneralParams& gp = n->gp;
  PhysicsParams& pyp = n->pyp;

  if(pyp.physicsActive){
    // Sphere
    float mass = 4.0 / 3.0 * std::numbers::pi * gp.scale * gp.scale * gp.scale * pyp.density;
    pyp.inv_mass = 1.0f/mass;
    float I = 2.0 / 5.0 * mass * gp.scale * gp.scale;
    float I_inv = 1.0/I;
    pyp.inv_inertia = glm::vec3(I_inv,I_inv,I_inv);
    pyp.inv_rotation = glm::inverse(gp.rotation);
  }else{
    pyp.vel = glm::vec3(0.0);
    pyp.omega = glm::vec3(0.0);
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
void Scene::simulate(float dts){
  if(dts <= 0.0) 
    return;

  int lastIdx = m_root.size()-1;

  for(int sub_step = 0; sub_step < SIM_NUM_SUBSTEPS; sub_step++){
    if(m_root[lastIdx].pyp.physicsActive){
      //LOGI("====================\n");
      for(int i = 0; i<m_root.size(); i++)
        integrate(&m_root[i], dts, m_gravity);

      //solveDistanceConstrain(&m_root[lastIdx],glm::vec3(0,0.35,0),glm::vec3(0,1,0),1.0,0.001,false,true,dts);

      solveCollisionConstraint(lastIdx, 0.0, dts);

      for(int i = 0; i<m_root.size(); i++)
        updateVelocities(&m_root[i], dts);
    }
  }

  for(int i = 0; i<m_root.size(); i++){
    Node& n = m_root[i];
    if(n.pyp.physicsActive)
      updateNodeData(&n);
  }
}

float Scene::sphereTrace(glm::vec3 orig, glm::vec3 dir, int objIdxExcluded){
  const int MAX_ITERATIONS = shaderio::NUM_VOXELS_PER_AXIS * int(CLIPMAP_LEVELS*0.5);
  const float MIN_DIST = 0.0001;
  const float maxDepth = 100.0;
  
  float depth = 0.0;

  for(int i = 0; i < MAX_ITERATIONS; i++){
    glm::vec3 p = orig + dir * depth;
    float dist = map(p,objIdxExcluded);
    
    if(glm::abs(dist) < MIN_DIST){
      return depth;  // Hit
    }
    
    if(depth >= maxDepth) return -1.0;  // No hit

    depth += dist;
  }
  
  return -1.0;
}

void Scene::centerCamAction(glm::vec3 pos, glm::vec3 dir){
  int lastIdx = m_root.size()-1;
  float depth = sphereTrace(pos, dir, lastIdx);
  glm::vec3 p = pos + dir*depth;
  glm::vec3 normal = evalNormal(p,lastIdx);
  if(depth >= 0){
    m_root[lastIdx].gp.position = p;
  }else{
    m_root[lastIdx].gp.position = glm::vec3(0,-10,0);
  }
  updateNodeData(&m_root[lastIdx]);
}
