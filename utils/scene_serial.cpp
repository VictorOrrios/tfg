#include "nvutils/bounding_box.hpp"
#include "scene.hpp"
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>

//------------------------------
// Math types
//------------------------------
namespace glm {

template <class Archive> void serialize(Archive &ar, glm::vec3 &v) {
  ar(v.x, v.y, v.z);
}

template <class Archive> void serialize(Archive &ar, glm::vec4 &v) {
  ar(v.x, v.y, v.z, v.w);
}

template <class Archive> void serialize(Archive &ar, glm::quat &q) {
  ar(q.x, q.y, q.z, q.w);
}

template <class Archive> void serialize(Archive &ar, glm::mat4 &m) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      ar(m[i][j]);
}

template <class Archive> void serialize(Archive &ar, glm::i32vec3 &v) {
  ar(v.x, v.y, v.z);
}

} // namespace glm

//------------------------------
// NVUtils bbox
//------------------------------
namespace nvutils {

template <class Archive> void save(Archive &ar, const Bbox &b) {
  ar(b.min(), b.max());
}

template <class Archive> void load(Archive &ar, Bbox &b) {
  glm::vec3 mn, mx;
  ar(mn, mx);
  b = Bbox(mn, mx);
}

} // namespace nvutils

//------------------------------
// Enums
//------------------------------
template <class Archive> void serialize(Archive &ar, Scene::CombinationOp &e) {
  ar((int &)e);
}

template <class Archive> void serialize(Archive &ar, Scene::RepetitionOp &e) {
  ar((int &)e);
}

template <class Archive> void serialize(Archive &ar, Scene::DeformationOp &e) {
  ar((int &)e);
}

template <class Archive> void serialize(Archive &ar, Scene::UserAction &e) {
  ar((int &)e);
}

//------------------------------
// Params
//------------------------------
template <class Archive> void serialize(Archive &ar, Scene::GeneralParams &p) {
  ar(p.type, p.mat, p.position, p.rotation, p.tInv, p.scale, p.bbox,
     p.prevBbox);
}

template <class Archive> void serialize(Archive &ar, Scene::GuizmoParams &g) {
  ar(g.guizmoOp, g.guizmoMode, g.matrix);
}

template <class Archive> void serialize(Archive &ar, Scene::PhysicsParams &p) {
  ar(p.physicsActive, p.density, p.inv_mass, p.prev_position, p.inv_rotation,
     p.prev_rotation, p.vel, p.omega, p.inv_inertia, p.pos_diff, p.pos_delta,
     p.omega_delta);
}

template <class Archive> void serialize(Archive &ar, Scene::SDFParams &s) {
  ar(s.roundness, s.combOp, s.smoothness, s.repOp, s.spacing, s.limit, s.defOp,
     s.defP, s.octaves, s.terrain, s.morphPrim, s.morph);
}

//------------------------------
// Node
//------------------------------
template <class Archive> void serialize(Archive &ar, Scene::Node &n) {
  ar(n.id, n.needsRefresh, n.needsRemoval, n.gp, n.sdp, n.pyp, n.gzp);
}

//------------------------------
// Material
//------------------------------
template <class Archive> void serialize(Archive &ar, Scene::Material &m) {
  ar(m.id, m.name, m.albedo, m.shininess, m.roughness, m.metalness, m.type);
}


//------------------------------
// Save & Load
//------------------------------
bool Scene::saveToFile(const std::string &path) {
  std::ofstream file(path);
  if (!file.is_open())
    return false;

  cereal::JSONOutputArchive ar(file);

  ar(m_root);
  ar(m_mat);

  return true;
}

bool Scene::loadFromFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    return false;

  for(auto n:m_root){
    m_removeList.push_back(n.gp.bbox);
  }

  cereal::JSONInputArchive ar(file);

  ar(m_root);
  ar(m_mat);

  uint max_id = 0;
  for(auto& n:m_root){
    markRefresh(&n);
    max_id = glm::max(max_id,n.id);
  }
  m_nextID = max_id + 1;

  m_needsRefresh = true;
  m_selected = -1;
  m_ignoreNextDynamicUpdate = true;

  return true;
}