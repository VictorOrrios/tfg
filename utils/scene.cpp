#include "scene.hpp"
#include "imgui.h"
#include <string>
#include <unordered_map>

//------------------
// Helper functions
//------------------

std::string Scene::nodeTypeToString(NodeType type) {
  static std::unordered_map<NodeType, std::string> type_label = {
      {NodeType::Box, "Box"},
      {NodeType::Sphere, "Sphere"},
      {NodeType::Snowman, "Snowman"},
  };

  return type_label[type];
}

static constexpr const char* SceneOperationNames[] = {
    "Union",
    "Substraction",
    "Intersection",
};

static constexpr const char* AxisOperationNames[] = {
    "None",
    "Symmetry",
    "Repetition"
};

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

    ImGui::InputFloat3("Position", &m_selected->position.x);
    ImGui::InputFloat3("Rotation", &m_selected->rotation.x);
    ImGui::InputFloat3("Scale", &m_selected->scale.x);
    ImGui::ColorEdit3("Albedo", &m_selected->albedo.x);
    ImGui::Checkbox("Physics active", &m_selected->physicsActive);
    
    ImGui::Combo("Scene operation", &m_selected->sop, SceneOperationNames, IM_ARRAYSIZE(SceneOperationNames));
    
    ImGui::SliderFloat("Smoothness", &m_selected->opP.smoothness, 0.0f, 10.0f);

    ImGui::Combo("Axis operation", &m_selected->aop, AxisOperationNames, IM_ARRAYSIZE(AxisOperationNames));

    AxisOperation selAOp = (AxisOperation)m_selected->aop;
    if (selAOp == AxisOperation::Symmetry) {
      ImGui::Checkbox("X", &m_selected->opP.symX);
      ImGui::SameLine();
      ImGui::Checkbox("Y", &m_selected->opP.symY);
      ImGui::SameLine();
      ImGui::Checkbox("Z", &m_selected->opP.symZ);
      ImGui::InputFloat3("Offset", &m_selected->opP.symOffset.x);
    }else if (selAOp == AxisOperation::Repetition) {
      ImGui::InputFloat3("Spacing", &m_selected->opP.spacing.x);
      ImGui::DragInt3("Limit", &m_selected->opP.limit.x, 0.1f, 0, INT_MAX);
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
    for (int i = 0; i < (int)NodeType::_COUNT; ++i) {
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

  bool open = ImGui::TreeNodeEx(node->label.c_str(), flags);

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
  auto node = std::make_unique<Node>();

  node->id = getNextId();
  node->label = nodeTypeToString((NodeType)t) + "##" + std::to_string(node->id);
  node->position = m_selected->position;
  node->rotation = m_selected->rotation;
  node->scale = m_selected->scale;
  node->albedo = m_selected->albedo;
  node->physicsActive = false;

  m_selected->children.push_back(std::move(node));

  return node;
}

//------------------
// Constructor
//------------------

Scene::Scene() {
  // Initialize root node (scene node)
  m_root = std::make_unique<Node>();
  m_root->id = getNextId();
  m_root->label = "Scene##" + std::to_string(m_root->id);
  m_root->position = glm::vec3(0, 0, 0);
  m_root->rotation = glm::vec3(0, 0, 0);
  m_root->scale = glm::vec3(1, 1, 1);
  m_root->albedo = glm::vec3(1, 1, 1);
  m_root->physicsActive = false;

  // Make the selected node the scene node by default
  m_selected = m_root.get();

  // Create the scene
  addChild(NodeType::Snowman);
}