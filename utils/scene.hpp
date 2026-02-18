#pragma once

#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_int3_sized.hpp"
#include <imgui.h>
#include <memory>
#include <string>
#include <vector>

class Scene {
public:
  enum class NodeType { Box, Sphere, Snowman, _COUNT };
  enum class SceneOperation { Union, Substraction, Intersection, _COUNT };
  enum class AxisOperation { NoneOP, Symmetry, Repetition, _COUNT };

  struct OpPramas {
    float smoothness;
    bool symX;
    bool symY;
    bool symZ;
    glm::vec3 symOffset;
    glm::vec3 spacing;
    glm::i32vec3 limit;
  };

  struct Node {
    uint32_t id;
    std::vector<std::unique_ptr<Node>> children;
    std::string label;
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::vec3 albedo;
    bool physicsActive;
    int sop;
    int aop;
    OpPramas opP;
  };

  Scene();

  void draw();

private:
  std::string nodeTypeToString(NodeType type);
  uint32_t getNextId();

  void drawNode(Node *node);
  void drawButtonGroup();

  void deleteSelected();
  bool deleteNodeRecursive(Node *parent, Node *target);

  std::unique_ptr<Node> addChild(NodeType t);

  std::unique_ptr<Node> m_root;
  Node *m_selected = nullptr;
  uint32_t m_nextID = 1;
};