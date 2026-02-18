#include "scene.hpp"
#include "glm/common.hpp"
#include "glm/ext/vector_float3.hpp"
#include "imgui.h"
#include "sdf.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/matrix.hpp>
#include <omp.h>
#include <stack>
#include <string>
#include <vector>

//------------------
// Helper functions
//------------------

static std::string NodeTypeNames[] = {
    "Box",
    "Sphere",
    "Snowman",
};

std::string Scene::nodeTypeToString(NodeType type) {
  return NodeTypeNames[(int)type];
}

std::string Scene::getLabel(Node *n) {
  return nodeTypeToString(n->type) + "##" + std::to_string(n->id);
}

static constexpr const char *SceneOperationNames[] = {
    "Union",
    "Substraction",
    "Intersection",
};

static constexpr const char *AxisOperationNames[] = {"None", "Symmetry",
                                                     "Repetition"};

uint32_t Scene::getNextId() { return m_nextID++; }

//------------------
// Draw functions
//------------------

void Scene::draw() {
  ImGui::Begin("Scene");

  drawButtonGroup();

  drawNode(m_root.get());

  if (m_selected != nullptr && m_selected != m_root.get()) {
    ImGui::Begin("Object");

    bool dirty = false;

    dirty |= ImGui::InputFloat3("Position", &m_selected->p.position.x);
    dirty |= ImGui::InputFloat3("Rotation", &m_selected->p.rotation.x);
    dirty |= ImGui::InputFloat("Scale", &m_selected->p.scale);
    ImGui::ColorEdit3("Albedo", &m_selected->p.albedo.x);
    ImGui::Checkbox("Physics active", &m_selected->p.physicsActive);

    dirty |=
        ImGui::Combo("Scene operation", &m_selected->p.sop, SceneOperationNames,
                     IM_ARRAYSIZE(SceneOperationNames));
    dirty |= ImGui::SliderFloat("Smoothness", &m_selected->p.smoothness, 0.0f,
                                0.25f);
    dirty |= ImGui::Combo("Axis operation", &m_selected->p.aop,
                          AxisOperationNames, IM_ARRAYSIZE(AxisOperationNames));

    AxisOperation selAOp = (AxisOperation)m_selected->p.aop;
    if (selAOp == AxisOperation::Symmetry) {
      dirty |= ImGui::Checkbox("X", &m_selected->p.symX);
      ImGui::SameLine();
      dirty |= ImGui::Checkbox("Y", &m_selected->p.symY);
      ImGui::SameLine();
      dirty |= ImGui::Checkbox("Z", &m_selected->p.symZ);
      ImGui::InputFloat3("Offset", &m_selected->p.symOffset.x);
    } else if (selAOp == AxisOperation::Repetition) {
      dirty |= ImGui::InputFloat3("Spacing", &m_selected->p.spacing.x);
      dirty |=
          ImGui::DragInt3("Limit", &m_selected->p.limit.x, 0.1f, 0, INT_MAX);
    }

    if (dirty) {
      generateMatrix(m_selected);
      m_needsRefresh = true;
    }

    ImGui::End();
  }
  ImGui::End();
}

void Scene::drawButtonGroup() {
  if (ImGui::Button("Add"))
    ImGui::OpenPopup("AddNodePopup");

  ImGui::SameLine();

  if (ImGui::Button("Delete")) {
    deleteSelected();
  }

  if (ImGui::BeginPopup("AddNodePopup")) {
    for (int i = 0; i < NodeTypeNames->length(); ++i) {
      if (ImGui::MenuItem(nodeTypeToString((NodeType)i).c_str()))
        addChild((NodeType)i);
    }

    ImGui::EndPopup();
  }
}

void Scene::drawNode(Scene::Node *node) {
  ImGuiTreeNodeFlags flags =
      ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;

  bool isSelected = node == m_selected;

  if (node->children.size() <= 0)
    flags |= ImGuiTreeNodeFlags_Leaf;

  if (isSelected)
    flags |= ImGuiTreeNodeFlags_Selected;

  std::string label = node == m_root.get() ? "Scene" : getLabel(node).c_str();
  bool open = ImGui::TreeNodeEx(label.c_str(), flags);

  if (ImGui::IsItemClicked())
    m_selected = node;

  if (open) {
    for (auto &&child : node->children)
      drawNode(child.get());

    ImGui::TreePop();
  }
}

//------------------
// Tree functions
//------------------

void Scene::deleteSelected() {
  // Cant delete root node
  if (m_selected == m_root.get())
    return;

  deleteNodeRecursive(m_root.get(), m_selected);

  m_selected = nullptr;
  m_needsRefresh = true;
}

bool Scene::deleteNodeRecursive(Node *parent, Node *target) {
  for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
    if (it->get() == target) {
      parent->children.erase(it);
      return true;
    }

    if (deleteNodeRecursive(it->get(), target))
      return true;
  }
  return false;
}

std::unique_ptr<Scene::Node> Scene::addChild(NodeType t) {
  if (m_selected == nullptr)
    m_selected = m_root.get();

  auto node = std::make_unique<Node>();

  node->id = getNextId();
  node->type = t;
  node->p.position = glm::vec3(0.0);
  node->p.rotation = glm::vec3(0.0);
  node->p.tInv = glm::mat4(1.0f);
  node->p.scale = m_selected->p.scale;
  node->p.albedo = m_selected->p.albedo;
  node->p.physicsActive = false;

  m_selected->children.push_back(std::move(node));
  m_selected = m_selected->children[m_selected->children.size() - 1].get();

  m_needsRefresh = true;
  return node;
}

//------------------
// Scene generation
//------------------
void Scene::generateMatrix(Node *n) {

  glm::mat4 transform4x4 = glm::mat4(1.0f);

  transform4x4 = glm::translate(transform4x4, n->p.position);
  
  if (n->p.rotation != glm::vec3(0.0)) {
    transform4x4 =
        glm::rotate(transform4x4, n->p.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    transform4x4 =
        glm::rotate(transform4x4, n->p.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    transform4x4 =
        glm::rotate(transform4x4, n->p.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
  }

  n->p.tInv = glm::inverse(transform4x4);
}

int Scene::flattenNode(Node *n, std::vector<FlatNode> &out) {
  int myIndex = out.size();
  out.push_back({});

  FlatNode &fn = out.back();
  fn.p = n->p;
  fn.type = (uint16_t)n->type;
  fn.firstChild = 0;
  fn.childCount = n->children.size();

  if (!n->children.empty()) {
    fn.firstChild = out.size();

    for (auto &c : n->children) {
      flattenNode(c.get(), out);
    }
  }

  return myIndex;
}

float Scene::map(glm::vec3 point, std::vector<FlatNode> flat) {
  // TODO: This needs to have better visibility
  const float iniD = 1000000.0f;
  constexpr int MAX_STACK = 12;

  struct StackNode {
    uint32_t idx;
    glm::vec3 point;
    int nextChild;
    float parent_value;
    float current_value;
  };

  Node *root = m_root.get();

  StackNode stack[MAX_STACK];
  int sp = 0;

  using sdfFunc = float (*)(const glm::vec3 &);
  static sdfFunc sdfTable[3] = {sdSphere, sdBox, sdSnowMan};

  stack[sp++] = {0, point, 1, iniD, iniD};
  stack[sp++] = {flat[0].firstChild, point, 0, iniD, 0};

  float popValue = 0.0, notUnion, isIntersection;

  while (sp > 0) {
    StackNode &sn = stack[sp - 1];
    glm::vec3 &p = sn.point;
    float &d = sn.current_value;
    FlatNode &fn = flat[sn.idx];
    NodeParams &params = fn.p;

    // If it's first time seeing this node calculate the primitive sdf and point
    if (sn.nextChild == 0) {

      if (params.aop == (int)AxisOperation::Symmetry) {
        if (params.symX)
          p.x = glm::abs(p.x);
        if (params.symY)
          p.y = glm::abs(p.y);
        if (params.symZ)
          p.z = glm::abs(p.z);
      } else if (params.aop == (int)AxisOperation::Repetition) {
        p -= params.spacing * glm::clamp(glm::round(p / params.spacing),
                                         glm::vec3(-params.limit),
                                         glm::vec3(params.limit));
      }

      p = glm::vec3(params.tInv * glm::vec4(p, 1.0f));
      p /= params.scale;

      d = sdfTable[fn.type](p);

      d *= params.scale;
    } else {
      // If not then use the popValue the child has calculated
      d = popValue;
    }

    if (sn.nextChild >= fn.childCount) {
      // No more children to process => Apply the node scene operation
      switch ((SceneOperation)params.sop) {
      case SceneOperation::Union:
        if (params.smoothness > 0.0f) {
          popValue = opSmoothUnion(sn.current_value, sn.parent_value,
                                   params.smoothness);
        } else {
          popValue = opUnion(sn.current_value, sn.parent_value);
        }
        break;
      case SceneOperation::Substraction:
        if (params.smoothness > 0.0f) {
          popValue = opSmoothSubtraction(sn.current_value, sn.parent_value,
                                         params.smoothness);
        } else {
          popValue = opSubtraction(sn.current_value, sn.parent_value);
        }
        break;
      case SceneOperation::Intersection:
        if (params.smoothness > 0.0f) {
          popValue = opSmoothIntersection(sn.current_value, sn.parent_value,
                                          params.smoothness);
        } else {
          popValue = opIntersection(sn.current_value, sn.parent_value);
        }
        break;
      }

      sp--;
    } else {
      // There is still children to process => Push next child to stack
      stack[sp++] = {fn.firstChild + sn.nextChild, p, 0, d, 0};
      // Update the next child to visit when revisiting this node
      sn.nextChild++;
    }
  }

  return popValue;
}

std::vector<float> Scene::generateDenseGrid(int num_voxels_per_axis) {
  Node *root = m_root.get();
  glm::vec3 center = glm::vec3(0.5, 0.5, 0.5);
  int total_voxels =
      num_voxels_per_axis * num_voxels_per_axis * num_voxels_per_axis;
  std::vector<float> data(total_voxels);

  int axis_size = num_voxels_per_axis;
  int axis_size_sq = axis_size * axis_size;

  std::vector<FlatNode> flat;
  flattenNode(root, flat);

  float emptyRoot = root->children.size() <= 0;

  if (emptyRoot) {
    for (int i = 0; i < total_voxels; i++) {
      data[i] = 10000.f;
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int z = 0; z < axis_size; z++) {
      for (int y = 0; y < axis_size; y++) {
        for (int x = 0; x < axis_size; x++) {
          glm::vec3 point =
              (glm::vec3(x + 0.5f, y + 0.5f, z + 0.5f) / float(axis_size) -
               center);

          float d = map(point, flat);

          data[z * axis_size_sq + y * axis_size + x] = d;
        }
      }
    }
  }

  return data;
}

//------------------
// Constructor
//------------------

Scene::Scene() {
  // Initialize root node (scene node)
  m_root = std::make_unique<Node>();
  m_root->id = getNextId();
  m_root->p.position = glm::vec3(0, 0, 0);
  m_root->p.rotation = glm::vec3(0, 0, 0);
  m_root->p.scale = 1.0;
  m_root->p.albedo = glm::vec3(1, 1, 1);
  m_root->p.physicsActive = false;
  generateMatrix(m_root.get());

  // Make the selected node the scene node by default
  m_selected = m_root.get();

  // Create the scene
  addChild(NodeType::Snowman);
}