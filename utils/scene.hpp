#pragma once

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_int3_sized.hpp"
#include "nvutils/bounding_box.hpp"
#include <imgui.h>
#include <string>
#include <vector>
#include "../shaders/shaderio.h"

class Scene {
public:
  enum class NodeType { Empty, Box, Sphere, Torus, Snowman };
  enum class CombinationOp { Union, Substraction, Intersection };
  enum class RepetitionOp { NoneOP, LimRepetition, IlimRepetition };
  enum class DeformationOp { NoneOP, Elongate };

  struct NodeParams {
    NodeType type;
    glm::vec3 position;
    glm::vec3 rotation;
    glm::mat4 tInv;
    float scale;
    float roundness;
    int combOp;
    float smoothness;
    int repOp ;
    glm::vec3 spacing;
    glm::i32vec3 limit;
    int defOp ;
    glm::vec3 defP;
  };

  struct Node {
    uint32_t id;
    NodeParams p;
    nvutils::Bbox bbox;
  };

  Scene();

  void draw();

  std::vector<float> generateDenseGrid();
  std::vector<nvutils::Bbox> getBboxes();
  std::vector<shaderio::SceneObject> getObjects();
  std::vector<shaderio::BuildJob> getBuildJobs(std::vector<nvutils::Bbox> aabbs);

  bool m_needsRefresh = true;

private:
  std::string nodeTypeToString(NodeType type);
  std::string getLabel(Node *n);
  uint32_t getNextId();

  void drawPrimitives();
  void drawButtonGroup();

  void deleteSelected();
  void addNode(NodeType t);
  void addNode(Node*);
  Node* createNode(NodeType t);

  void updateNodeData(Node *n);
  void generateMatrix(Node *n);
  void generateBBox(Node *n);
  float map(glm::vec3 p);

  std::vector<shaderio::BuildJob> splitBuildJob(shaderio::BuildJob);

  std::vector<Node> m_root;
  int m_selected = -1;
  uint32_t m_nextID = 1;
};