// TODO: Clean imports
#include "scene.hpp"
#include "glm/common.hpp"
#include "glm/exponential.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_int3.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/trigonometric.hpp"
#include "nvutils/bounding_box.hpp"
#include "nvutils/logger.hpp"
#include "sdf.hpp"
#include <cmath>
#include <glm/ext/vector_float3.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/matrix.hpp>
#include <imgui.h>
#include <omp.h>
#include <string>
#include <vector>

//------------------
// Defintions
//------------------
// TODO: CPU scene map not ~fully~ supported

static constexpr const char * NodeTypeNames[] = {
    "Empty", "Box", "Sphere", "Torus", "Snowman", "Plane"
};
using sdf3DPrimitiveF = float (*)(const glm::vec3 &);
static sdf3DPrimitiveF primFTable[6] = {sdEmpty, sdBox, sdSphere, sdTorus,
                                        sdSnowMan, sdPlane};

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
static deformationOpF defFTable[2] = {opNone, opElongate};

static constexpr const char *MorphPrimNames[] = {
  "Box",
  "Sphere",
  "Torus",
  "Snowman",
};

//------------------
// Helper functions
//------------------

std::string Scene::nodeTypeToString(NodeType type) {
  return NodeTypeNames[(int)type];
}

std::string Scene::getLabel(Node *n) {
  return nodeTypeToString(n->p.type) + "##" + std::to_string(n->id);
}

std::string Scene::getLabel(Material  mat) {
  return mat.name + "##" + std::to_string(mat.id);
}

uint32_t Scene::getNextId() { return m_nextID++; }


//------------------
// Draw functions
//------------------

void Scene::draw() {
  ImGui::Begin("Scene");

  if (ImGui::BeginTabBar("Scene")){
    if (ImGui::BeginTabItem("Objects")){
      drawButtonGroup();
      ImGui::Separator();
      drawPrimitives();
      drawNodeParams();
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Materials")){
      drawMaterials();
      drawMaterialParams();
      ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
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
    for (int i = 0; i < IM_ARRAYSIZE(NodeTypeNames); ++i) {
      if (ImGui::MenuItem(nodeTypeToString((NodeType)i).c_str()))
        addNode((NodeType)i);
    }

    ImGui::EndPopup();
  }
}

void Scene::drawPrimitives() {
  ImGuiSelectableFlags selectableFlags = 0;

  bool clickedOnItem = false;

  for (int idx = 0; idx < m_root.size(); idx++) {
    auto& node = m_root[idx];
    bool isSelected = idx == m_selected;
    std::string label = getLabel(&node).c_str();

    // Draw primitive
    if (ImGui::Selectable(label.c_str(), isSelected, selectableFlags)) {
      m_selected = idx;
      clickedOnItem = true;
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
          updateNodeData(&m_root[m_selected]);
        }
      }
      ImGui::EndDragDropTarget();
    }
  }

  if (!clickedOnItem && ImGui::IsMouseClicked(0) && ImGui::IsWindowHovered()) {
    m_selected = -1;
  }
}

template<typename T>
bool ComboVector(const char* label, int* current_item, std::vector<T>& vec){
  auto getter = [](void* data, int idx, const char** out_text) -> bool
  {
    auto& v = *static_cast<std::vector<T>*>(data);
    if (idx < 0 || idx >= (int)v.size()) return false;
    *out_text = v[idx].name.c_str();
    return true;
  };

  return ImGui::Combo(label, current_item, getter, &vec, (int)vec.size());
}


void Scene::drawNodeParams(){
  

  if (m_selected != -1) {
    ImGui::Begin("Object");

    Node &selectedNode = m_root[m_selected];

    const std::string id = "##" + std::to_string(selectedNode.id);
    bool dirty = false;
    int combOpUI = selectedNode.p.combOp >= 3 ? selectedNode.p.combOp - 3
                                              : selectedNode.p.combOp;

    dirty |= ComboVector("Material", &selectedNode.p.mat, m_mat);

    if (ImGui::IsKeyPressed(ImGuiKey_T))
        selectedNode.p.gzParam.guizmoOp = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        selectedNode.p.gzParam.guizmoOp = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_E))
        selectedNode.p.gzParam.guizmoOp = ImGuizmo::SCALE;
    if (ImGui::RadioButton("Translate", selectedNode.p.gzParam.guizmoOp == ImGuizmo::TRANSLATE))
        selectedNode.p.gzParam.guizmoOp = ImGuizmo::TRANSLATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Rotate", selectedNode.p.gzParam.guizmoOp == ImGuizmo::ROTATE))
        selectedNode.p.gzParam.guizmoOp = ImGuizmo::ROTATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Scale", selectedNode.p.gzParam.guizmoOp == ImGuizmo::SCALE))
        selectedNode.p.gzParam.guizmoOp = ImGuizmo::SCALE;
 

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
                                0.04);
    ImGui::Separator();

    dirty |= ImGui::Combo(("Morphing primitive" + id).c_str(),
                          &selectedNode.p.morphPrim, MorphPrimNames,
                          IM_ARRAYSIZE(MorphPrimNames));

    dirty |= ImGui::SliderFloat(("Morphing" + id).c_str(),
                                &selectedNode.p.morph, 0.0f, 1.0f);

    ImGui::Separator();


    dirty |= ImGui::Combo(("Deformation operation" + id).c_str(),
                          &selectedNode.p.defOp, DeformationOpNames,
                          IM_ARRAYSIZE(DeformationOpNames));
    if (selectedNode.p.defOp == (int)DeformationOp::Elongate) {
      dirty |= ImGui::InputFloat3(("Elongation" + id).c_str(),
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

    ImGui::Separator();

    dirty |= ImGui::SliderInt(("Terrain octaves" + id).c_str(),
                                    &selectedNode.p.octaves,0,20);
    if(selectedNode.p.octaves > 0){
      dirty |= ImGui::SliderFloat(("Initial size" + id).c_str(),
                                    &selectedNode.p.terrain.x,0.001,2.0);
      dirty |= ImGui::SliderFloat(("Size increase" + id).c_str(),
                                    &selectedNode.p.terrain.y,0.001,0.75);
      dirty |= ImGui::SliderFloat(("Inflation" + id).c_str(),
                                    &selectedNode.p.terrain.z,0.001,1);
      dirty |= ImGui::SliderFloat(("Erosion" + id).c_str(),
                                    &selectedNode.p.terrain.w,0.001,1);
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
}

void Scene::drawMaterials(){
  if(ImGui::Button("Add material"))
    addMaterial(createMaterial());
  
  ImGuiSelectableFlags selectableFlags = 0;

  bool clickedOnItem = false;

  for (int idx = 0; idx < m_mat.size(); idx++) {
    auto& mat = m_mat[idx];
    bool isSelected = idx == m_selectedMat;
    std::string label = getLabel(mat).c_str();

    // Draw material
    if (ImGui::Selectable(label.c_str(), isSelected, selectableFlags)) {
      m_selectedMat = idx;
      clickedOnItem = true;
    }
  }

  if (!clickedOnItem && ImGui::IsMouseClicked(0) && ImGui::IsWindowHovered()) {
    m_selectedMat = -1;
  }

}

void Scene::drawMaterialParams(){
  if (m_selectedMat != -1) {
    ImGui::Begin("Material");

    Material& mat = m_mat[m_selectedMat];

    const std::string id = "##" + std::to_string(mat.id);
    bool dirty = false;

    char buffer[256];
    strncpy(buffer, mat.name.c_str(), sizeof(buffer));
    buffer[sizeof(buffer) - 1] = '\0';
    if (ImGui::InputText(("Name" + id).c_str(), buffer, sizeof(buffer)))
      mat.name = std::string(buffer);

    dirty |= ImGui::ColorEdit3(("Albedo" + id).c_str(),
                                &mat.albedo.x);

    dirty |= ImGui::SliderFloat(("Roughness" + id).c_str(),
                                &mat.roughness, 0.0f, 1.0f);

    dirty |= ImGui::SliderFloat(("Metalness" + id).c_str(),
                                &mat.metalness, 0.0f, 1.0f);

    if (dirty) {
      m_needsRefresh = true;
    }

    ImGui::End();
  }
}

void Scene::drawGuizmo(ImVec2 viewportPos, ImVec2 viewportSize, glm::mat4 cameraView, glm::mat4 cameraProjection){
  if(m_selected != -1){ 
    ImGuizmo::BeginFrame();

    ImGuizmo::SetDrawlist();

    ImGuizmo::SetRect(
      viewportPos.x,
      viewportPos.y,
      viewportSize.x,
      viewportSize.y
    );

    Node &selectedNode = m_root[m_selected];
    GuizmoParams& gzP = selectedNode.p.gzParam;

    cameraProjection[1][1] *= -1.0f;
    ImGuizmo::Manipulate(
      glm::value_ptr(cameraView), 
      glm::value_ptr(cameraProjection), 
      gzP.guizmoOp, 
      gzP.guizmoMode, 
      glm::value_ptr(gzP.matrix),
      NULL, NULL);


    m_usingGuizmo = ImGuizmo::IsUsing();

    if(m_usingGuizmo){
      float& scale = selectedNode.p.scale;
      glm::vec3 scale_vec, rot_deg;
      ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(gzP.matrix), &selectedNode.p.position.x, &rot_deg.x, &scale_vec.x);
      selectedNode.p.rotation = glm::radians(rot_deg);
      selectedNode.p.scale = scale_vec[0];
      updateNodeData(&selectedNode);
      m_needsRefresh = true;
    }
  }
}


//------------------
// Tree functions
//------------------

void Scene::deleteSelected() {
  // Cant delete empty
  if (m_selected == -1)
    return;

  m_removeList.push_back(m_root[m_selected].bbox);
  m_root.erase(m_root.begin() + m_selected);
  m_selected = -1;
  m_needsRefresh = true;
}

Scene::Node *Scene::createNode(NodeType t) {
  Node *node = new Node({
      .id = getNextId(),
      .p={
        .type = t,
        .gzParam = {
          ImGuizmo::TRANSLATE,
          ImGuizmo::WORLD,
          glm::mat4(1.0)
        },
        .terrain = glm::vec4(1.0,0.5,0.1,0.3),
      },
      .bbox=nvutils::Bbox(glm::vec3(0.0),glm::vec3(0.0)),
      .needsRefresh=false,
  });

  if (m_selected != -1) {
    Node selectedNode = m_root[m_selected];
    node->p.position = selectedNode.p.position;
    node->p.rotation = selectedNode.p.rotation;
    node->p.scale = selectedNode.p.scale;
  }else{
    node->p.scale = 1.0;
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
// Material functions
//------------------
Scene::Material Scene::createMaterial(){
  return {
    .id = getNextId(),
    .name = "Material "+std::to_string(m_mat.size()),
    .albedo = glm::vec3(1.0),
    .roughness = 0.5f,
    .metalness = 0.1f,
  };
}

int Scene::addMaterial(Material mat){
  if(m_mat.size() >= MAX_MATERIALS){
    LOGW("Scene material vector full, skipping material\n");
  }else{
    m_mat.push_back(mat);
  }
  return m_mat.size()-1;
}

//------------------
// Scene generation
//------------------
void Scene::updateNodeData(Node *n) {
  markRefresh(n);
  generateMatrix(n);
  generateBBox(n);
}

void Scene::markRefresh(Node* n){
  if(!n->needsRefresh){
    n->prevBbox = nvutils::Bbox(n->bbox);
    n->needsRefresh = true;
    m_needsRefresh = true;
  }
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

  glm::vec3 scale_vec(n->p.scale), rot_deg(glm::degrees(n->p.rotation));
  ImGuizmo::RecomposeMatrixFromComponents(
    glm::value_ptr(n->p.position), 
    glm::value_ptr(rot_deg),
    glm::value_ptr(scale_vec),
    glm::value_ptr(n->p.gzParam.matrix));

  n->p.tInv = glm::inverse(transform4x4);
}

// TODO: Make bboxes with intersection op include all bbox above it in the scene
void Scene::generateBBox(Node *n) {
  const glm::vec3 worldMin(-1000.0);
  const glm::vec3 worldMax(1000.0);

  glm::vec3 min, max;
  if(n->p.type == NodeType::Plane){
    min = worldMin;
    max = worldMax;
    max.y = 0.1;
  }else{
    min = glm::vec3(-0.5);
    max = glm::vec3(0.5);
  }

  float spacing = 0.15; // Safety margin
  spacing += n->p.smoothness * 5;
  spacing += n->p.roundness;

  if(n->p.octaves > 0){
    spacing += n->p.terrain.x * n->p.terrain.z
    * (1.0f - glm::pow(n->p.terrain.y,n->p.octaves-1))/(1.0f - n->p.terrain.y);
    spacing += n->p.terrain.w/8.0;
  }

  min -= spacing;
  max += spacing;

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

  min = glm::max(bboxt.min(), worldMin);
  max = glm::min(bboxt.max(), worldMax);

  n->bbox = nvutils::Bbox(min, max);
}

std::vector<nvutils::Bbox> Scene::getAllBboxes() {
  std::vector<nvutils::Bbox> out;

  for (auto &node : m_root) {
    out.push_back(node.bbox);
  }

  return out;
}

glm::mat4 columnMajorToRowMajor(const glm::mat4& mat) {
  glm::mat4 result;
  
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      result[row][col] = mat[col][row];
    }
  }
  
  return result;
}


std::vector<shaderio::SceneObject> Scene::getObjects(){
  std::vector<shaderio::SceneObject> out;

  for (auto &node : m_root) {
    NodeParams& p = node.p; 

    int morphPrim = p.morphPrim+1;
    
    out.push_back({
      .tInv=columnMajorToRowMajor(p.tInv),
      .spacing=glm::vec4(p.spacing,0),
      .defP=glm::vec4(p.defP,0),
      .terrain=glm::vec4(p.terrain),
      .limit_octaves=glm::ivec4(p.limit,p.octaves),
      .type=int(p.type),
      .combOp=p.combOp,
      .repOp=p.repOp,
      .defOp=p.defOp,
      .morphPrim=morphPrim,
      .scale=p.scale,
      .roundness=p.roundness,
      .smoothness=p.smoothness,
      .morph=p.morph,
      .mat=uint(p.mat),
    });
  }

  return out;
}

std::vector<shaderio::Material> Scene::getMaterials(){
  std::vector<shaderio::Material> out;

  for (auto &mat : m_mat) {
    out.push_back({
      .albedo = glm::vec4(mat.albedo,0.0),
      .roughness = mat.roughness,
      .metalness = mat.metalness
    });
  }

  return out;
}


bool pointInBBox(const glm::vec3& p, const nvutils::Bbox& bbox) {
  glm::vec3 min = bbox.min();
  glm::vec3 max = bbox.max();

  return glm::all(glm::greaterThanEqual(p, min)) &&
           glm::all(glm::lessThanEqual(p, max));
}

float Scene::map(glm::vec3 point) {
  const float iniD = 1000000.0f;

  float result = iniD;

  for (auto &node : m_root) {
    glm::vec3 p = point;
    float d;

    // If not inside bbox primitive continue with next 
    if(!pointInBBox(p,node.bbox))
      continue;

    NodeParams &params = node.p;

    p = glm::vec3(params.tInv * glm::vec4(p, 1.0f));

    p = repFTable[params.repOp](p, params.spacing, params.limit);

    p = defFTable[params.defOp](p, params.defP);

    d = primFTable[int(node.p.type)](p / params.scale) - params.roundness;

    d *= params.scale;

    result = combFTable[params.combOp](d, result, params.smoothness);
  }

  return result;
}

std::vector<float> Scene::generateDenseGrid() {
  glm::vec3 center = glm::vec3(0.5, 0.5, 0.5);
  const int num_values_per_axis = shaderio::NUM_VALUES_PER_AXIS;
  int total_voxels =
      num_values_per_axis * num_values_per_axis * num_values_per_axis;
  std::vector<float> data(total_voxels);

  int axis_size = num_values_per_axis;
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
      glm::vec3 point = (glm::vec3(x,y,z) * voxel_size - center);

      float d = glm::min(map(point), max_d);

      data[z * axis_size_sq + y * axis_size + x] = d;
    }
  }

  return data;
}

inline glm::ivec3 id0LevelTransform(glm::ivec3 id0, int level){
  return glm::floor(glm::vec3(id0)/float(1<<level));
}

std::vector<shaderio::BuildJob> Scene::createBaseBuildJobs(nvutils::Bbox bbox, glm::ivec3 camId0){
  const glm::ivec3 zeros(0);
  const glm::ivec3 max_index(NUM_BRICKS_PER_AXIS-1);
  const glm::ivec3 hole_min(NUM_BRICKS_PER_AXIS/4);
  const glm::ivec3 hole_max(NUM_BRICKS_PER_AXIS*3/4-1);
  
  std::vector<shaderio::BuildJob> jobs;
  
  //for(int level=CLIPMAP_LEVELS-1 ; level>=0; level--){
  for(int level=0 ; level<CLIPMAP_LEVELS; level++){
    glm::ivec3 camId = id0LevelTransform(camId0, level);

    glm::ivec3 min_id = glm::floor(bbox.min()/shaderio::BRICK_SIZES[level]);
    glm::ivec3 max_id = glm::floor(bbox.max()/shaderio::BRICK_SIZES[level]);

    glm::ivec3 min_rel_id = min_id - camId + (NUM_BRICKS_PER_AXIS/2);
    glm::ivec3 max_rel_id = max_id - camId + (NUM_BRICKS_PER_AXIS/2);

    // Completly out of range check
    if(glm::any(glm::lessThan(max_rel_id,zeros)) || glm::any(glm::greaterThan(min_rel_id,max_index)))
      continue;

    // Completly inside the hole in levels > 0
    if(
      level > 0 &&
      glm::all(glm::greaterThanEqual(min_rel_id,hole_min)) &&
      glm::all(glm::lessThanEqual(max_rel_id,hole_max))
    )
      continue;

    // Clamp min and max to relative ids bounds
    min_rel_id = glm::max(min_rel_id,zeros);
    max_rel_id = glm::min(max_rel_id,max_index);

    // Calculate number of bricks
    glm::ivec3 num_b = glm::abs(min_rel_id - max_rel_id) + glm::ivec3(1);

    // Convert back to global id
    min_id = min_rel_id + camId - (NUM_BRICKS_PER_AXIS/2);

    jobs.push_back({
      .min_id_level=glm::ivec4(min_id,level),
      .num_b=glm::ivec4(num_b,0)
    });
  }

  return jobs;
}

std::vector<shaderio::BuildJob> Scene::createCamBuildJobs(glm::ivec3 currCamId0, glm::ivec3 prevCamId0){
  std::vector<shaderio::BuildJob> out;

  for(int level=0; level<CLIPMAP_LEVELS; level++){
    glm::ivec3 currCamId = id0LevelTransform(currCamId0,level);
    glm::ivec3 currMinId = currCamId - NUM_BRICKS_PER_AXIS/2;
    glm::ivec3 currMaxId = currCamId + NUM_BRICKS_PER_AXIS/2 - 1;
    glm::ivec3 currHMinId = currCamId - NUM_BRICKS_PER_AXIS/4;
    glm::ivec3 currHMaxId = currCamId + NUM_BRICKS_PER_AXIS/4 - 1;

    glm::ivec3 prevCamId = id0LevelTransform(prevCamId0,level);
    glm::ivec3 prevMinId = prevCamId - NUM_BRICKS_PER_AXIS/2;
    glm::ivec3 prevMaxId = prevCamId + NUM_BRICKS_PER_AXIS/2 - 1;
    glm::ivec3 prevHMinId = prevCamId - NUM_BRICKS_PER_AXIS/4;
    glm::ivec3 prevHMaxId = prevCamId + NUM_BRICKS_PER_AXIS/4 - 1;

    for(int axis = 0; axis < 3; ++axis){
      if(prevCamId[axis] == currCamId[axis])
        continue;

      // New area -> Grid outer reach
      glm::ivec3 minId = currMinId;
      glm::ivec3 maxId = currMaxId;

      if(prevCamId[axis] <= currCamId[axis]){
        minId[axis] = prevMaxId[axis];
      }else{
        maxId[axis] = prevMinId[axis];
      }

      glm::ivec3 num_b = glm::abs(minId - maxId) + glm::ivec3(1);

      out.push_back({
        .min_id_level=glm::ivec4(minId,level),
        .num_b=glm::ivec4(num_b,0)
      });

      // Hole area -> Grid inner reach
      if(level > 0){
        minId = prevHMinId;
        maxId = prevHMaxId;

        if(prevCamId[axis] <= currCamId[axis]){
          maxId[axis] = currHMinId[axis];
        }else{
          minId[axis] = currHMaxId[axis];
        }

        num_b = glm::abs(minId - maxId) + glm::ivec3(1);

        out.push_back({
          .min_id_level=glm::ivec4(minId,level),
          .num_b=glm::ivec4(num_b,0)
        });
      }

      // Grid inner reach -> Hole area
      if(level > 0){
        minId = currHMinId;
        maxId = currHMaxId;

        if(prevCamId[axis] <= currCamId[axis]){
          minId[axis] = prevHMaxId[axis];
        }else{
          maxId[axis] = prevHMinId[axis];
        }

        num_b = glm::abs(minId - maxId) + glm::ivec3(1);

        out.push_back({
          .min_id_level=glm::ivec4(minId,level),
          .num_b=glm::ivec4(num_b,0)
        });
      }
    }
  }

  return out;
}


// Splits BuildJobs into chunks that have a max size of MAX_BUILD_JOB_SIZE³
std::vector<shaderio::BuildJob> Scene::splitBuildJob(shaderio::BuildJob buildJ){
  const glm::ivec3 base_min_id = glm::ivec3(buildJ.min_id_level.x,buildJ.min_id_level.y,buildJ.min_id_level.z);
  const glm::ivec3 base_num_b = glm::ivec3(buildJ.num_b.x,buildJ.num_b.y,buildJ.num_b.z);
  const glm::ivec3 max_chunk = glm::ivec3(MAX_BUILD_JOB_SIZE);

  std::vector<shaderio::BuildJob> out;

  for(int z = 0; z<buildJ.num_b.z; z+= MAX_BUILD_JOB_SIZE)
  for(int y = 0; y<buildJ.num_b.y; y+= MAX_BUILD_JOB_SIZE)
  for(int x = 0; x<buildJ.num_b.x; x+= MAX_BUILD_JOB_SIZE)
  {
    glm::ivec3 offset = glm::ivec3(x,y,z);
    glm::ivec3 num_b = glm::min(base_num_b-offset,max_chunk);
    glm::ivec3 min_id = base_min_id+offset;
    out.push_back({
      .min_id_level = glm::ivec4(min_id,buildJ.min_id_level.w),
      .num_b = glm::ivec4(num_b,0)
    });
  };

  return out;
}

std::vector<shaderio::BuildJob> Scene::getBuildJobs(glm::ivec3 currCamId0, glm::ivec3 prevCamId0){
  std::vector<nvutils::Bbox> aabbs;
  std::vector<shaderio::BuildJob> out, baseJobs, levelSplitted;

  for (auto &node : m_root) {
    if(node.needsRefresh){
      aabbs.push_back(node.bbox);
      aabbs.push_back(node.prevBbox);
      node.prevBbox = nvutils::Bbox(node.bbox);
      node.needsRefresh = false;
    }
  }

  out.reserve(aabbs.size()*4+3);
  baseJobs = createCamBuildJobs(currCamId0,prevCamId0);
  
  for(auto& bbox: aabbs){
    // Negative volume build job check
    if(glm::any(glm::lessThan(bbox.max(),bbox.min())))
      continue;

    levelSplitted = createBaseBuildJobs(bbox, currCamId0);
    baseJobs.insert(baseJobs.end(),levelSplitted.begin(),levelSplitted.end());
  }

  for(auto& bbox: m_removeList){
    // Negative volume build job check
    if(glm::any(glm::lessThan(bbox.max(),bbox.min())))
      continue;

    levelSplitted = createBaseBuildJobs(bbox, currCamId0);
    baseJobs.insert(baseJobs.end(),levelSplitted.begin(),levelSplitted.end());
  }
  m_removeList.clear();

  for(auto& buildJob: baseJobs){
    auto splited = splitBuildJob(buildJob);
    out.insert(out.end(),splited.begin(),splited.end());
  }

  return out;
}

std::vector<shaderio::BuildJob> Scene::getDenseBuildJobs(glm::ivec3 currCamId0, glm::ivec3 prevCamId0){
  std::vector<shaderio::BuildJob> out, baseJobs;

  nvutils::Bbox bbox(glm::vec3(-100000.0),glm::vec3(100000.0));
  baseJobs = createBaseBuildJobs(bbox,currCamId0);

  for(auto& buildJob: baseJobs){
    auto splited = splitBuildJob(buildJob);
    out.insert(out.end(),splited.begin(),splited.end());
  }

  return out;
}


//------------------
// Constructor
//------------------
Scene::Scene() {
  Material mat = createMaterial();
  mat.name = "Default";
  int matIdx = addMaterial(mat);

  mat = createMaterial();
  mat.name = "New material";
  mat.albedo = glm::vec3(1,0,0);
  int newMat = addMaterial(mat);

  // Create the scene
  Node *terrain = createNode(NodeType::Plane);
  terrain->p.position = glm::vec3(0.0,-2.0,0.0);
  terrain->p.scale = 10.0;
  terrain->p.octaves = 8;
  terrain->p.terrain = glm::vec4(1.5,0.35,0.08,0.28);
  updateNodeData(terrain);
  addNode(terrain);

  Node *snowMan = createNode(NodeType::Snowman);
  snowMan->p.scale = 0.8;
  snowMan->p.position = glm::vec3(0.0,0.0,-1.0);
  updateNodeData(snowMan);
  addNode(snowMan);

  Node *box = createNode(NodeType::Box);
  box->p.scale = 0.2;
  box->p.position = glm::vec3(-0.2, -0.15, -0.75);
  box->p.rotation = glm::vec3(0.2, 0.4, 0.4);
  box->p.combOp = (int)CombinationOp::Union + 3;
  box->p.smoothness = 0.02;
  box->p.mat = newMat;
  updateNodeData(box);
  addNode(box);

  Node *sphere = createNode(NodeType::Sphere);
  sphere->p.scale = 0.2;
  sphere->p.position = glm::vec3(0.1, 0.3, -0.9);
  sphere->p.rotation = glm::vec3(0);
  sphere->p.combOp = (int)CombinationOp::Substraction;
  sphere->p.mat = newMat;
  updateNodeData(sphere);
  addNode(sphere);

  Node *sphereGrid = createNode(NodeType::Sphere);
  sphereGrid->p.position = glm::vec3(0, 0, -1.4);
  sphereGrid->p.rotation = glm::vec3(0);
  sphereGrid->p.scale = 0.1;
  sphereGrid->p.repOp = (int)RepetitionOp::LimRepetition;
  sphereGrid->p.spacing = glm::vec3(0.14,0.14,0);
  sphereGrid->p.limit = glm::ivec3(13,13,1);
  sphereGrid->p.mat = newMat;
  updateNodeData(sphereGrid);
  addNode(sphereGrid);

  Node *torus = createNode(NodeType::Torus);
  torus->p.scale = 0.2;
  torus->p.position = glm::vec3(0.35, 0.1, -1.2);
  torus->p.rotation = glm::vec3(0.75, 0, 0);
  torus->p.mat = newMat;
  updateNodeData(torus);
  addNode(torus);

  Node *test = createNode(NodeType::Snowman);
  test->p.scale = 0.4;
  test->p.position = glm::vec3(1.5,1.5,1.5);
  test->p.rotation = glm::vec3(0.0);
  updateNodeData(test);
  addNode(test);

  for(int level=1; level<3; level++){
    Node *snowManL = createNode(NodeType::Snowman);
    snowManL->p.scale = 0.5*(1<<level);
    snowManL->p.position = glm::vec3(-((L0_AXIS_WORLD_SIZE*0.5)*(1<<level))*3/4, 0.0, 0.0);
    snowManL->p.rotation = glm::vec3(0.0,0.4*level,0.0);
    updateNodeData(snowManL);
    addNode(snowManL);
  }

  m_selected = 1;
}