// TODO: Clean imports
#include "scene.hpp"
#include "glm/common.hpp"
#include "nvutils/bounding_box.hpp"
#include "sdf.hpp"
#include <glm/ext/vector_float3.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/matrix.hpp>
#include <imgui.h>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>

//------------------
// Defintions
//------------------

static std::string NodeTypeNames[] = {
    "Empty", "Box", "Sphere", "Torus", "Snowman",
};
using sdf3DPrimitiveF = float (*)(const glm::vec3 &);
static sdf3DPrimitiveF primFTable[5] = {sdEmpty, sdBox, sdSphere, sdTorus,
                                        sdSnowMan};

static constexpr const char *CombinationOpNames[] = {
    "Union",
    "Substraction",
    "Intersection",
};
using combinationOpF = float (*)(float, float, float);
static combinationOpF combFTable[6] = {
    opUnion,       opSubtraction,       opIntersection,
    opSmoothUnion, opSmoothSubtraction, opSmoothIntersection};

static constexpr const char *RepetitionOpnames[] = {
    "None", "Limited repetition", "Unlimited repetition"};
using repetitionOpF = glm::vec3 (*)(const glm::vec3 &, const glm::vec3 &,
                                    const glm::vec3 &);
static repetitionOpF repFTable[3] = {opNone, opLimRepetition, opRepetition};

static constexpr const char *DeformationOpNames[] = {
    "None",
    "Elongate",
};
using deformationOpF = glm::vec3 (*)(const glm::vec3 &, const glm::vec3 &);
static deformationOpF defFTable[4] = {opNone, opElongate};

//------------------
// Helper functions
//------------------

std::string Scene::nodeTypeToString(NodeType type) {
  return NodeTypeNames[(int)type];
}

std::string Scene::getLabel(Node *n) {
  return nodeTypeToString(n->type) + "##" + std::to_string(n->id);
}

uint32_t Scene::getNextId() { return m_nextID++; }

//------------------
// Draw functions
//------------------

void Scene::draw() {
  ImGui::Begin("Scene");

  drawButtonGroup();

  drawPrimitives();

  if (m_selected != -1) {
    ImGui::Begin("Object");

    Node &selectedNode = m_root[m_selected];

    const std::string id = "##" + std::to_string(selectedNode.id);
    bool dirty = false;
    int combOpUI = selectedNode.p.combOp >= 3 ? selectedNode.p.combOp - 3
                                              : selectedNode.p.combOp;

    dirty |= ImGui::InputFloat3(("Position" + id).c_str(),
                                &selectedNode.p.position.x);
    dirty |= ImGui::InputFloat3(("Rotation" + id).c_str(),
                                &selectedNode.p.rotation.x);
    ImGui::Separator();
    dirty |= ImGui::InputFloat(("Scale" + id).c_str(), &selectedNode.p.scale);

    dirty |= ImGui::SliderFloat(("Roundness" + id).c_str(),
                                &selectedNode.p.roundness, 0.0f,
                                selectedNode.p.scale * 0.25);
    ImGui::Separator();

    dirty |= ImGui::Combo(("Combination operation" + id).c_str(), &combOpUI,
                          CombinationOpNames, IM_ARRAYSIZE(CombinationOpNames));
    dirty |= ImGui::SliderFloat(("Smoothness" + id).c_str(),
                                &selectedNode.p.smoothness, 0.0f,
                                selectedNode.p.scale * 0.2);
    ImGui::Separator();

    dirty |= ImGui::Combo(("Deformation operation" + id).c_str(),
                          &selectedNode.p.defOp, DeformationOpNames,
                          IM_ARRAYSIZE(DeformationOpNames));
    if (selectedNode.p.defOp == (int)DeformationOp::Elongate) {
      dirty |= ImGui::InputFloat3(("Deformation" + id).c_str(),
                                  &selectedNode.p.defP.x);
    }

    ImGui::Separator();

    dirty |= ImGui::Combo(("Repetition operation" + id).c_str(),
                          &selectedNode.p.repOp, RepetitionOpnames,
                          IM_ARRAYSIZE(RepetitionOpnames));

    if ((RepetitionOp)selectedNode.p.repOp != RepetitionOp::NoneOP) {
      dirty |= ImGui::InputFloat3(("Spacing" + id).c_str(),
                                  &selectedNode.p.spacing.x);
      if ((RepetitionOp)selectedNode.p.repOp == RepetitionOp::LimRepetition)
        dirty |= ImGui::DragInt3(("Limit" + id).c_str(),
                                 &selectedNode.p.limit.x, 0.1f, 0, INT_MAX);
    }

    if (dirty) {
      // If smoothness != 0 then apply the smooth combination operations (3,4,5)
      // if not use the faster version (0,1,2)
      if (selectedNode.p.smoothness > 0.0f) {
        selectedNode.p.combOp = combOpUI + 3;
      } else {
        selectedNode.p.combOp = combOpUI;
      }
      // Update the transformation matrix and bounding box
      updateNodeData(&selectedNode);
      // Flag to render engine that scene needs grid regeneration
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
        addNode((NodeType)i);
    }

    ImGui::EndPopup();
  }
}

void Scene::drawPrimitives() {
  ImGuiSelectableFlags selectableFlags = 0;

  int idx = 0;
  for (auto &node : m_root) {
    bool isSelected = idx == m_selected;
    std::string label = getLabel(&node).c_str();

    // Draw primitive
    if (ImGui::Selectable(label.c_str(), isSelected, selectableFlags)) {
      m_selected = idx;
    }

    // Drag source
    if (ImGui::BeginDragDropSource()) {
      ImGui::SetDragDropPayload("DND_SCENE_NODE", &idx, sizeof(int));
      ImGui::Text("%s", label.c_str());
      ImGui::EndDragDropSource();
    }

    if (ImGui::BeginDragDropTarget()) {
      if (const ImGuiPayload *payload =
              ImGui::AcceptDragDropPayload("DND_SCENE_NODE")) {
        int sourceIdx = *(const int *)payload->Data;

        if (sourceIdx != idx) {
          auto movedItem = std::move(m_root[sourceIdx]);
          m_root.erase(m_root.begin() + sourceIdx);
          m_root.insert(m_root.begin() + idx, std::move(movedItem));

          m_selected = idx;

          m_needsRefresh = true;
        }
      }
      ImGui::EndDragDropTarget();
    }

    idx++;
  }
}

//------------------
// Tree functions
//------------------

void Scene::deleteSelected() {
  // Cant delete empty
  if (m_selected == -1)
    return;

  m_root.erase(m_root.begin() + m_selected);
  m_selected = -1;
  m_needsRefresh = true;
}

Scene::Node *Scene::createNode(NodeType t) {
  Node *node = new Node({
      .id = getNextId(),
      .type = t,
  });

  if (m_selected != -1) {
    Node selectedNode = m_root[m_selected];
    node->p.position = selectedNode.p.position;
    node->p.rotation = selectedNode.p.rotation;
    node->p.scale = selectedNode.p.scale;
  }

  updateNodeData(node);

  return node;
}

void Scene::addNode(NodeType t) { addNode(createNode(t)); }

void Scene::addNode(Node *node) {
  int insertIdx = m_selected == -1 ? 0 : m_selected + 1;

  m_root.insert(m_root.begin() + insertIdx, *node);
  m_needsRefresh = true;
  m_selected = insertIdx;
}

//------------------
// Scene generation
//------------------
void Scene::updateNodeData(Node *n) {
  generateMatrix(n);
  generateBBox(n);
}

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

// TODO: Make children expand parent bbox and make children use the parents
// p.tInv
void Scene::generateBBox(Node *n) {
  float spacing = 0.1; // Safety margin
  spacing += n->p.smoothness * 5;
  spacing += n->p.roundness;

  glm::vec3 min(-0.5 - spacing);
  glm::vec3 max(0.5 + spacing);

  min *= n->p.scale;
  max *= n->p.scale;

  if (n->p.defOp == (int)DeformationOp::Elongate) {
    min -= n->p.defP;
    max += n->p.defP;
  }

  glm::vec3 repOffset = glm::vec3(0.0);
  if (n->p.repOp == (int)RepetitionOp::LimRepetition) {
    repOffset = n->p.spacing * glm::vec3(n->p.limit);
  } else if (n->p.repOp == (int)RepetitionOp::IlimRepetition) {
    repOffset =
        glm::step(0.0001f, n->p.spacing) * std::numeric_limits<float>::max();
  }
  min -= repOffset;
  max += repOffset;

  nvutils::Bbox bboxt(min, max);
  bboxt = bboxt.transform(glm::inverse(n->p.tInv));

  min = glm::max(bboxt.min(), -0.5f);
  max = glm::min(bboxt.max(), 0.5f);

  n->bbox = nvutils::Bbox(min, max);
}

std::vector<nvutils::Bbox> Scene::getBboxes() {
  std::vector<nvutils::Bbox> out;

  for (auto &node : m_root) {
    out.push_back(node.bbox);
  }

  return out;
}

float Scene::map(glm::vec3 point) {
  const float iniD = 1000000.0f;

  float result = iniD;

  for (auto &node : m_root) {
    NodeParams &params = node.p;
    glm::vec3 p = point;
    float d;

    p = glm::vec3(params.tInv * glm::vec4(p, 1.0f));

    p = repFTable[params.repOp](p, params.spacing, params.limit);

    p = defFTable[params.defOp](p, params.defP);

    d = primFTable[int(node.type)](p / params.scale) - params.roundness;

    d *= params.scale;

    result = combFTable[params.combOp](d, result, params.smoothness);
  }

  return result;
}

std::vector<float> Scene::generateDenseGrid(int num_voxels2) {
  glm::vec3 center = glm::vec3(0.5, 0.5, 0.5);
  const int num_voxels_per_axis = num_voxels2 + 1;
  int total_voxels =
      num_voxels_per_axis * num_voxels_per_axis * num_voxels_per_axis;
  std::vector<float> data(total_voxels);

  int axis_size = num_voxels_per_axis;
  int axis_size_sq = axis_size * axis_size;
  float voxel_size = 1 / float(axis_size);
  float max_d = glm::sqrt(3.0 * 2.5 * 2.5 * voxel_size * voxel_size);

  // If empty scene
  if (m_root.size() <= 0) {
    for (int i = 0; i < total_voxels; i++) {
      data[i] = 10000.f;
    }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_voxels; i++) {
      int z = i / axis_size_sq;
      int y = (i % axis_size_sq) / axis_size;
      int x = i % axis_size;
      glm::vec3 point =
          (glm::vec3(x + 0.5f, y + 0.5f, z + 0.5f) * voxel_size - center);

      float d = glm::min(map(point), max_d);

      data[z * axis_size_sq + y * axis_size + x] = d;
    }
  }

  return data;
}

//------------------
// Constructor
//------------------

Scene::Scene() {
  // Create the scene
  Node *snowMan = createNode(NodeType::Snowman);
  snowMan->p.scale = 0.8;
  snowMan->p.position.z = 0.1;
  updateNodeData(snowMan);
  addNode(snowMan);

  Node *box = createNode(NodeType::Box);
  box->p.scale = 0.2;
  box->p.position = glm::vec3(-0.2, -0.15, 0.35);
  box->p.rotation = glm::vec3(0.2, 0.4, 0.4);
  box->p.combOp = (int)CombinationOp::Union + 3;
  box->p.smoothness = 0.02;
  updateNodeData(box);
  addNode(box);

  Node *sphere = createNode(NodeType::Sphere);
  sphere->p.scale = 0.2;
  sphere->p.position = glm::vec3(0.1, 0.3, 0.1);
  sphere->p.rotation = glm::vec3(0);
  sphere->p.combOp = (int)CombinationOp::Substraction;
  updateNodeData(sphere);
  addNode(sphere);

  Node *sphereGrid = createNode(NodeType::Sphere);
  sphereGrid->p.position = glm::vec3(0, 0, -0.4);
  sphereGrid->p.rotation = glm::vec3(0);
  sphereGrid->p.scale = 0.1;
  sphereGrid->p.repOp = (int)RepetitionOp::IlimRepetition;
  sphereGrid->p.spacing.x = 0.15;
  sphereGrid->p.spacing.y = 0.15;
  updateNodeData(sphereGrid);
  addNode(sphereGrid);

  Node *torus = createNode(NodeType::Torus);
  torus->p.scale = 0.2;
  torus->p.position = glm::vec3(0.35, 0.1, -0.2);
  torus->p.rotation = glm::vec3(0.75, 0, 0);
  updateNodeData(torus);
  addNode(torus);
}