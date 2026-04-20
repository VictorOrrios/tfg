#pragma once

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_int3_sized.hpp>
#include <glm/gtx/quaternion.hpp>
#include "nvutils/bounding_box.hpp"
#include <imgui.h>
#include <string>
#include <vector>
#include "../shaders/shaderio.h"
#include <nvvk/profiler_vk.hpp>
#include "ImGuizmo.h"

class Scene {
public:
  enum class NodeType { Empty, Box, Sphere, Torus, Snowman, Plane };
  enum class CombinationOp { Union, Substraction, Intersection };
  enum class RepetitionOp { NoneOP, LimRepetition, IlimRepetition };
  enum class DeformationOp { NoneOP, Elongate };

  struct GeneralParams{
    NodeType type;
    int mat;
    glm::vec3 position;
    glm::quat rotation;
    glm::mat4 tInv;
    float scale;
    nvutils::Bbox bbox;
    nvutils::Bbox prevBbox;
  };

  struct GuizmoParams{
    ImGuizmo::OPERATION guizmoOp;
    ImGuizmo::MODE guizmoMode;
    glm::mat4 matrix;
  };

  struct PhysicsParams {
    bool physicsActive;
    float density;
    float inv_mass;
    glm::vec3 prev_position;
    glm::quat inv_rotation;
    glm::quat prev_rotation;
    glm::vec3 vel;
    glm::vec3 omega;
    glm::vec3 inv_inertia;
  };

  struct SDFParams {
    float roundness;
    int combOp;
    float smoothness;
    int repOp;
    glm::vec3 spacing;
    glm::i32vec3 limit;
    int defOp;
    glm::vec3 defP;
    int octaves;
    glm::vec4 terrain;
    int morphPrim;
    float morph;
  };

  struct Material {
    uint32_t id;
    std::string name;
    glm::vec3 albedo;
    float shininess;
    float roughness;
    float metalness;
  };

  struct Node {
    uint32_t id;
    bool needsRefresh;
    GeneralParams gp;
    SDFParams     sdp;
    PhysicsParams pyp;
    GuizmoParams  gzp;
  };

  Scene();

  void draw();
  void drawGuizmo(ImVec2 viewportPos, ImVec2 viewportSize, glm::mat4 cameraView, glm::mat4 cameraProjection);

  void simulate(float dt);
  void centerCamAction(glm::vec3 pos, glm::vec3 dir);

  std::vector<float> generateDenseGrid();
  std::vector<nvutils::Bbox> getAllBboxes();
  std::vector<shaderio::SceneObject> getObjects();
  std::vector<shaderio::Material> getMaterials();
  std::vector<shaderio::BuildJob> getBuildJobs(glm::ivec3 currCamId0, glm::ivec3 prevCamId0);
  std::vector<shaderio::BuildJob> getDenseBuildJobs(glm::ivec3 currCamId0, glm::ivec3 prevCamId0);

  bool m_needsRefresh = true;
  bool m_usingGuizmo = false;

private:
  std::string nodeTypeToString(NodeType type);
  std::string getLabel(Node *n);
  std::string getLabel(Material mat);
  uint32_t getNextId();

  void drawPrimitives();
  void drawButtonGroup();
  void drawNodeParams();

  void drawMaterials();
  void drawMaterialParams();

  void deleteSelected();
  void addNode(NodeType t);
  void addNode(Node*);
  Node* createNode(NodeType t);

  Material createMaterial();
  int addMaterial(Material mat);

  void solveCollisionConstraint(int nodeId, float compliance, float dt);
  float sphereTrace(glm::vec3 orig, glm::vec3 dir, int objIdxExcluded = -1);

  void updateNodeData(Node *n);
  void updateNodePysicsData(Node *n);
  void markRefresh(Node* n);
  void generateMatrix(Node *n);
  void generateBBox(Node *n);
  float map(glm::vec3 p);
  float mapExclude(glm::vec3 p, int objIdxExcluded);

  std::vector<shaderio::BuildJob> createBaseBuildJobs(nvutils::Bbox aabb, glm::ivec3 camId0);
  std::vector<shaderio::BuildJob> createCamBuildJobs(glm::ivec3 currCamId0, glm::ivec3 prevCamId0);
  std::vector<shaderio::BuildJob> splitBuildJob(shaderio::BuildJob);

  std::vector<Node> m_root;
  std::vector<Material> m_mat;
  std::vector<nvutils::Bbox> m_removeList;
  int m_selected = -1;
  int m_selectedMat = -1;
  uint32_t m_nextID = 1;

  glm::vec3 m_gravity = glm::vec3(0.0f,-9.8f,0.0f);
};