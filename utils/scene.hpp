#pragma once

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_int3_sized.hpp"
#include "../shaders/shaderio.h"
#include "nvutils/bounding_box.hpp"
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

class Scene {
public:
  enum class NodeType { Empty, Box, Sphere, Snowman };
  enum class CombinationOp { Union, Substraction, Intersection };
  enum class RepetitionOp { NoneOP, LimRepetition, IlimRepetition };
  enum class DeformationOp { NoneOP, Twist, Bend, Elongate };

  struct NodeParams {
    glm::vec3 position;
    glm::vec3 rotation;
    glm::mat4 tInv;
    float scale;
    float roundness;
    int combOPUI;
    int combOP;
    float smoothness;
    int repOp ;
    glm::vec3 spacing;
    glm::i32vec3 limit;
    int defOp ;
    glm::vec3 defP;
  };

  struct Node {
    uint32_t id;
    std::vector<std::unique_ptr<Node>> children;
    NodeType type;
    NodeParams p;
    nvutils::Bbox bbox;
  };

  struct FlatNode {
    Scene::NodeParams p;
    int type;
    uint32_t firstChild;
    uint32_t childCount;
  };

  Scene();

  void draw();

  std::vector<float> generateDenseGrid(int num_voxels_per_axis);
  std::vector<shaderio::SceneObject> getObjects();

  bool m_needsRefresh = false;

private:
  std::string nodeTypeToString(NodeType type);
  std::string getLabel(Node *n);
  uint32_t getNextId();

  void drawNode(Node *node);
  void drawButtonGroup();

  void deleteSelected();
  bool deleteNodeRecursive(Node *parent, Node *target);

  Node* addChild(NodeType t);

  void updateNodeData(Node *n);
  void generateMatrix(Node *n);
  void generateBBox(Node *n);
  float map(glm::vec3 p, std::vector<FlatNode> flat);
  std::vector<Scene::FlatNode> flattenNode(Node *root);

  std::unique_ptr<Node> m_root;
  Node *m_selected = nullptr;
  uint32_t m_nextID = 1;
};