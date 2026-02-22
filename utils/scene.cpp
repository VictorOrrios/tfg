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
    "Empty",
    "Box",
    "Sphere",
    "Snowman",
};
using sdf3DPrimitiveF = float (*)(const glm::vec3 &);
static sdf3DPrimitiveF primFTable[4] = {sdEmpty, sdBox, sdSphere, sdSnowMan};

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
    "Twist",
    "Bend",
    "Elongate",
};
using deformationOpF = glm::vec3 (*)(const glm::vec3 &, const glm::vec3 &);
static deformationOpF defFTable[4] = {opNone, opTwist, opBend, opElongate};

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

  drawNode(m_root.get());

  if (m_selected != nullptr && m_selected != m_root.get()) {
    ImGui::Begin("Object");

    const std::string id = "##" + std::to_string(m_selected->id);
    bool dirty = false;

    dirty |= ImGui::InputFloat3(("Position" + id).c_str(),
                                &m_selected->p.position.x);
    dirty |= ImGui::InputFloat3(("Rotation" + id).c_str(),
                                &m_selected->p.rotation.x);
    ImGui::Separator();
    dirty |= ImGui::InputFloat(("Scale" + id).c_str(), &m_selected->p.scale);

    dirty |=
        ImGui::SliderFloat(("Roundness" + id).c_str(), &m_selected->p.roundness,
                           0.0f, m_selected->p.scale * 0.25);
    ImGui::Separator();

    dirty |= ImGui::Combo(("Combination operation" + id).c_str(),
                          &m_selected->p.combOPUI, CombinationOpNames,
                          IM_ARRAYSIZE(CombinationOpNames));
    dirty |= ImGui::SliderFloat(("Smoothness" + id).c_str(),
                                &m_selected->p.smoothness, 0.0f,
                                m_selected->p.scale * 0.1);
    ImGui::Separator();

    dirty |= ImGui::Combo(("Deformation operation" + id).c_str(),
                          &m_selected->p.defOp, DeformationOpNames,
                          IM_ARRAYSIZE(DeformationOpNames));
    if (m_selected->p.defOp != 0)
      dirty |= ImGui::InputFloat3(("Deformation" + id).c_str(),
                                  &m_selected->p.defP.x);

    ImGui::Separator();

    dirty |= ImGui::Combo(("Repetition operation" + id).c_str(),
                          &m_selected->p.repOp, RepetitionOpnames,
                          IM_ARRAYSIZE(RepetitionOpnames));

    if ((RepetitionOp)m_selected->p.repOp != RepetitionOp::NoneOP) {
      dirty |= ImGui::InputFloat3(("Spacing" + id).c_str(),
                                  &m_selected->p.spacing.x);
      if ((RepetitionOp)m_selected->p.repOp == RepetitionOp::LimRepetition)
        dirty |= ImGui::DragInt3(("Limit" + id).c_str(), &m_selected->p.limit.x,
                                 0.1f, 0, INT_MAX);
    }

    if (dirty) {
      // If smoothness != 0 then apply the smooth combination operations (3,4,5) if not use the faster version (0,1,2)
      if(m_selected->p.smoothness > 0.0f){
        m_selected->p.combOP = m_selected->p.combOPUI + 3;
      }else{
        m_selected->p.combOP = m_selected->p.combOPUI;
      }
      // Update the transformation matrix and bounding box
      updateNodeData(m_selected);
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
    for (int i = 0; i < NodeTypeNames->length() - 1; ++i) {
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

Scene::Node *Scene::addChild(NodeType t) {
  if (m_selected == nullptr)
    m_selected = m_root.get();

  auto node = std::make_unique<Node>();
  Node *nodePtr = node.get();

  node->id = getNextId();
  node->type = t;
  node->p.position = glm::vec3(0.0);
  node->p.rotation = glm::vec3(0.0);
  node->p.tInv = glm::mat4(1.0f);
  node->p.scale = m_selected->p.scale;

  m_selected->children.push_back(std::move(node));
  m_selected = m_selected->children[m_selected->children.size() - 1].get();

  m_needsRefresh = true;
  return nodePtr;
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

void Scene::generateBBox(Node *n) {
  glm::mat4 t = glm::inverse(n->p.tInv);

  const float spacing = 0.1;
  glm::vec3 min(-0.5 - spacing);
  glm::vec3 max(0.5 + spacing);
  min *= n->p.scale;
  max *= n->p.scale;

  nvutils::Bbox bbox(min, max);
  bbox = bbox.transform(t);

  n->bbox = bbox;
}

// Post: Flatten node using breadth first
std::vector<Scene::FlatNode> Scene::flattenNode(Node *root) {
  std::vector<Scene::FlatNode> out;
  if (!root)
    return out;

  struct Item {
    Node *node;
    int fIdx;
  };

  std::queue<Item> q;
  out.push_back({});
  q.push({root, 0});

  while (!q.empty()) {
    Item it = q.front();
    q.pop();

    Node *n = it.node;
    FlatNode &fn = out[it.fIdx];
    fn.p = n->p;
    fn.type = (int)n->type;
    fn.childCount = n->children.size();
    fn.firstChild = out.size();

    for (auto &c : n->children) {
      q.push({c.get(), (int)out.size()});
      out.push_back({});
    }
  }

  return out;
}

std::vector<shaderio::SceneObject> Scene::getObjects() {
  std::vector<shaderio::SceneObject> out;

  std::queue<Node *> q;
  q.push(m_root.get());

  while (!q.empty()) {
    Node *n = q.front();
    q.pop();

    nvutils::Bbox &nvbbox = n->bbox;
    shaderio::Bbox shbbox({nvbbox.min(), nvbbox.max()});
    out.push_back({shbbox});

    for (auto &c : n->children) {
      q.push(c.get());
    }
  }

  // out[1].bbox.bMin = glm::vec3(-0.5);
  // out[1].bbox.bMax = glm::vec3(0.5);

  return out;
}

float Scene::map(glm::vec3 point, std::vector<FlatNode> flat) {
  // TODO: This needs to have better visibility
  const float iniD = 1000000.0f;
  constexpr int MAX_STACK = 32;

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
      p = defFTable[params.defOp](p,params.defP);
      
      p = repFTable[params.repOp](p,params.spacing,params.limit);

      p = glm::vec3(params.tInv * glm::vec4(p, 1.0f));

      d = primFTable[fn.type](p / params.scale);

      d *= params.scale;
    } else {
      // If not then use the popValue the child has calculated
      d = popValue;
    }

    if (sn.nextChild >= fn.childCount) {
      // No more children to process => Apply the node scene operation
      popValue = combFTable[params.combOP](sn.current_value, sn.parent_value,params.smoothness);

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

  std::vector<FlatNode> flat = flattenNode(root);

  float emptyRoot = root->children.size() <= 0;

  if (emptyRoot) {
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
          (glm::vec3(x + 0.5f, y + 0.5f, z + 0.5f) / float(axis_size) - center);

      float d = map(point, flat);

      data[z * axis_size_sq + y * axis_size + x] = d;
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
  updateNodeData(m_root.get());

  // Make the selected node the scene node by default
  m_selected = m_root.get();

  // Create the scene
  Node *snowMan = addChild(NodeType::Snowman);
  snowMan->p.scale = 0.8;
  updateNodeData(snowMan);

  Node *box = addChild(NodeType::Box);
  box->p.scale = 0.2;
  box->p.position.x = -0.2;
  box->p.position.y = -0.15;
  box->p.position.z = 0.25;
  box->p.rotation.x = 0.2;
  box->p.rotation.y = 0.4;
  box->p.rotation.z = 0.4;
  box->p.smoothness = 0.02;
  updateNodeData(box);

  m_selected = snowMan;
  Node *sphere = addChild(NodeType::Sphere);
  sphere->p.scale = 0.2;
  sphere->p.position.x = 0.1;
  sphere->p.position.y = 0.3;
  sphere->p.combOP = (int)CombinationOp::Substraction;
  updateNodeData(sphere);

  m_selected = m_root.get();
  Node *sphereGrid = addChild(NodeType::Sphere);
  sphereGrid->p.scale = 0.1;
  sphereGrid->p.position.z = -0.4;
  sphereGrid->p.repOp = (int)RepetitionOp::IlimRepetition;
  sphereGrid->p.spacing.x = 0.15;
  sphereGrid->p.spacing.y = 0.15;
  updateNodeData(sphereGrid);
}