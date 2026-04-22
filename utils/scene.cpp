// TODO: Clean imports
#include "scene.hpp"
#include "glm/common.hpp"
#include "glm/exponential.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_int3.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/trigonometric.hpp"
#include <glm/gtx/rotate_vector.hpp>
#include "nvutils/bounding_box.hpp"
#include "nvutils/logger.hpp"
#include "sdf.hpp"
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

static constexpr const char *CombinationOpNames[] = {
  "Union",
  "Substraction",
  "Intersection",
};

static constexpr const char *RepetitionOpnames[] = {
    "None", "Limited repetition", "Unlimited repetition"};

static constexpr const char *DeformationOpNames[] = {
  "None",
  "Elongate",
};

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
  return nodeTypeToString(n->gp.type) + "##" + std::to_string(n->id);
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
    int combOpUI = selectedNode.sdp.combOp >= 3 ? selectedNode.sdp.combOp - 3
                                              : selectedNode.sdp.combOp;

    dirty |= ComboVector("Material", &selectedNode.gp.mat, m_mat);

    if (ImGui::IsKeyPressed(ImGuiKey_T))
        selectedNode.gzp.guizmoOp = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        selectedNode.gzp.guizmoOp = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_E))
        selectedNode.gzp.guizmoOp = ImGuizmo::SCALE;
    if (ImGui::RadioButton("Translate", selectedNode.gzp.guizmoOp == ImGuizmo::TRANSLATE))
        selectedNode.gzp.guizmoOp = ImGuizmo::TRANSLATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Rotate", selectedNode.gzp.guizmoOp == ImGuizmo::ROTATE))
        selectedNode.gzp.guizmoOp = ImGuizmo::ROTATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Scale", selectedNode.gzp.guizmoOp == ImGuizmo::SCALE))
        selectedNode.gzp.guizmoOp = ImGuizmo::SCALE;
 

    dirty |= ImGui::InputFloat3(("Position" + id).c_str(),
                                &selectedNode.gp.position.x);
    dirty |= ImGui::InputFloat3(("Rotation" + id).c_str(),
                                &selectedNode.gp.rotation.x);
    dirty |= ImGui::InputFloat(("Scale" + id).c_str(), &selectedNode.gp.scale);

    ImGui::Separator();
    dirty |= ImGui::Checkbox("Physics active", &selectedNode.pyp.physicsActive);
    dirty |= ImGui::SliderFloat("Density", &selectedNode.pyp.density, 0.01f, 10.0f);

    ImGui::Separator();
    dirty |= ImGui::SliderFloat(("Roundness" + id).c_str(),
                                &selectedNode.sdp.roundness, 0.0f,
                                selectedNode.gp.scale * 0.25);
    ImGui::Separator();

    dirty |= ImGui::Combo(("Combination operation" + id).c_str(), &combOpUI,
                          CombinationOpNames, IM_ARRAYSIZE(CombinationOpNames));
    dirty |= ImGui::SliderFloat(("Smoothness" + id).c_str(),
                                &selectedNode.sdp.smoothness, 0.0f,
                                0.04);
    ImGui::Separator();

    dirty |= ImGui::Combo(("Morphing primitive" + id).c_str(),
                          &selectedNode.sdp.morphPrim, MorphPrimNames,
                          IM_ARRAYSIZE(MorphPrimNames));

    dirty |= ImGui::SliderFloat(("Morphing" + id).c_str(),
                                &selectedNode.sdp.morph, 0.0f, 1.0f);

    ImGui::Separator();


    dirty |= ImGui::Combo(("Deformation operation" + id).c_str(),
                          &selectedNode.sdp.defOp, DeformationOpNames,
                          IM_ARRAYSIZE(DeformationOpNames));
    if (selectedNode.sdp.defOp == (int)DeformationOp::Elongate) {
      dirty |= ImGui::InputFloat3(("Elongation" + id).c_str(),
                                  &selectedNode.sdp.defP.x);
    }

    ImGui::Separator();

    dirty |= ImGui::Combo(("Repetition operation" + id).c_str(),
                          &selectedNode.sdp.repOp, RepetitionOpnames,
                          IM_ARRAYSIZE(RepetitionOpnames));

    if ((RepetitionOp)selectedNode.sdp.repOp != RepetitionOp::NoneOP) {
      dirty |= ImGui::InputFloat3(("Spacing" + id).c_str(),
                                  &selectedNode.sdp.spacing.x);
      if ((RepetitionOp)selectedNode.sdp.repOp == RepetitionOp::LimRepetition)
        dirty |= ImGui::DragInt3(("Limit" + id).c_str(),
                                 &selectedNode.sdp.limit.x, 0.1f, 0, INT_MAX);
    }

    ImGui::Separator();

    dirty |= ImGui::SliderInt(("Terrain octaves" + id).c_str(),
                                    &selectedNode.sdp.octaves,0,20);
    if(selectedNode.sdp.octaves > 0){
      dirty |= ImGui::SliderFloat(("Initial size" + id).c_str(),
                                    &selectedNode.sdp.terrain.x,0.001,2.0);
      dirty |= ImGui::SliderFloat(("Size increase" + id).c_str(),
                                    &selectedNode.sdp.terrain.y,0.001,0.75);
      dirty |= ImGui::SliderFloat(("Inflation" + id).c_str(),
                                    &selectedNode.sdp.terrain.z,0.001,1);
      dirty |= ImGui::SliderFloat(("Erosion" + id).c_str(),
                                    &selectedNode.sdp.terrain.w,0.001,1);
    }

    if(dirty) {
      // If smoothness != 0 then apply the smooth combination operations (3,4,5)
      // if not use the faster version (0,1,2)
      if (selectedNode.sdp.smoothness > 0.0f) {
        selectedNode.sdp.combOp = combOpUI + 3;
      } else {
        selectedNode.sdp.combOp = combOpUI;
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

    dirty |= ImGui::SliderFloat(("Shininess" + id).c_str(),
                                &mat.shininess, 0.0f, MAX_SHININESS);

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
    GuizmoParams& gzP = selectedNode.gzp;

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
      float& scale = selectedNode.gp.scale;
      glm::vec3 scale_vec, rot_deg;
      ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(gzP.matrix), &selectedNode.gp.position.x, &rot_deg.x, &scale_vec.x);
      selectedNode.gp.rotation = glm::radians(rot_deg);
      selectedNode.gp.scale = scale_vec[0];
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

  m_root[m_selected].needsRemoval = true;
  m_needsRefresh = true;
  m_removeList.push_back(m_root[m_selected].gp.bbox);
  m_selected = -1;
}

void Scene::flushDeletedNodes(){
  m_root.erase(
    std::remove_if(m_root.begin(), m_root.end(),
      [](const Node& n) {
        return n.needsRemoval;
      }),
    m_root.end()
  );
}

Scene::Node *Scene::createNode(NodeType t) {
  Node *node = new Node({
      .id = getNextId(),
      .needsRefresh=false,
      .needsRemoval=false,
      .gp={
        .type = t,
        .rotation=glm::quat(glm::vec3(0.0)),
        .bbox=nvutils::Bbox(glm::vec3(0.0),glm::vec3(0.0)),
      },
      .sdp={
        .terrain = glm::vec4(1.0,0.5,0.1,0.3),
      },
      .pyp={
        .density = 1.0
      },
      .gzp = {
        ImGuizmo::TRANSLATE,
        ImGuizmo::WORLD,
        glm::mat4(1.0)
      },
  });

  if (m_selected != -1) {
    Node selectedNode = m_root[m_selected];
    node->gp.position = selectedNode.gp.position;
    node->gp.rotation = selectedNode.gp.rotation;
    node->gp.scale = selectedNode.gp.scale;
  }else{
    node->gp.scale = 1.0;
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
    .shininess = 64.0f,
    .roughness = 0.8,
    .metalness = 0.0,
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
  updateNodePysicsData(n);
}

void Scene::markRefresh(Node* n){
  if(!n->needsRefresh){
    n->gp.prevBbox = nvutils::Bbox(n->gp.bbox);
    n->needsRefresh = true;
    m_needsRefresh = true;
  }
}

void Scene::generateMatrix(Node *n) {

  glm::mat4 transform4x4 = glm::mat4(1.0f);

  transform4x4 = glm::translate(transform4x4, n->gp.position);

  if (n->gp.rotation != glm::quat(1.0, 0.0, 0.0, 0.0)) {
    glm::mat4 rotMatrix = glm::toMat4(n->gp.rotation);
    transform4x4 *= rotMatrix;
  }

  glm::vec3 scale_vec(n->gp.scale);
  glm::vec3 rot_deg(glm::degrees(glm::eulerAngles(n->gp.rotation)));
  ImGuizmo::RecomposeMatrixFromComponents(
    glm::value_ptr(n->gp.position), 
    glm::value_ptr(rot_deg),
    glm::value_ptr(scale_vec),
    glm::value_ptr(n->gzp.matrix));

  n->gp.tInv = glm::inverse(transform4x4);
}

// TODO: Make bboxes with intersection op include all bbox above it in the scene
void Scene::generateBBox(Node *n) {
  const glm::vec3 worldMin(-1000.0);
  const glm::vec3 worldMax(1000.0);

  glm::vec3 min, max;
  if(n->gp.type == NodeType::Plane){
    min = worldMin;
    max = worldMax;
    max.y = 0.1;
  }else{
    min = glm::vec3(-0.5);
    max = glm::vec3(0.5);
  }

  float spacing = 0.15; // Safety margin
  spacing += n->sdp.smoothness * 5;
  spacing += n->sdp.roundness;

  if(n->sdp.octaves > 0){
    spacing += n->sdp.terrain.x * n->sdp.terrain.z
    * (1.0f - glm::pow(n->sdp.terrain.y,n->sdp.octaves-1))/(1.0f - n->sdp.terrain.y);
    spacing += n->sdp.terrain.w/8.0;
  }

  min -= spacing;
  max += spacing;

  min *= n->gp.scale;
  max *= n->gp.scale;

  if (n->sdp.defOp == (int)DeformationOp::Elongate) {
    min -= n->sdp.defP;
    max += n->sdp.defP;
  }

  glm::vec3 repOffset = glm::vec3(0.0);
  if (n->sdp.repOp == (int)RepetitionOp::LimRepetition) {
    repOffset = n->sdp.spacing * glm::vec3(n->sdp.limit);
  } else if (n->sdp.repOp == (int)RepetitionOp::IlimRepetition) {
    repOffset =
        glm::step(0.0001f, n->sdp.spacing) * std::numeric_limits<float>::max();
  }
  min -= repOffset;
  max += repOffset;

  nvutils::Bbox bboxt(min, max);
  bboxt = bboxt.transform(glm::inverse(n->gp.tInv));

  min = glm::max(bboxt.min(), worldMin);
  max = glm::min(bboxt.max(), worldMax);

  n->gp.bbox = nvutils::Bbox(min, max);
}

std::vector<nvutils::Bbox> Scene::getAllBboxes() {
  std::vector<nvutils::Bbox> out;

  for (auto &node : m_root) {
    out.push_back(node.gp.bbox);
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
  
  return glm::transpose(mat);
}


std::vector<shaderio::SceneObject> Scene::getObjects(){
  std::vector<shaderio::SceneObject> out;

  for (auto &node : m_root) {
    GeneralParams& p = node.gp; 
    SDFParams& sdp = node.sdp; 

    int morphPrim = sdp.morphPrim+1;
    
    out.push_back({
      .tInv=columnMajorToRowMajor(p.tInv),
      .spacing=glm::vec4(sdp.spacing,0),
      .defP=glm::vec4(sdp.defP,0),
      .terrain=glm::vec4(sdp.terrain),
      .limit_octaves=glm::ivec4(sdp.limit,sdp.octaves),
      .type=int(p.type),
      .combOp=sdp.combOp,
      .repOp=sdp.repOp,
      .defOp=sdp.defOp,
      .morphPrim=morphPrim,
      .scale=p.scale,
      .roundness=sdp.roundness,
      .smoothness=sdp.smoothness,
      .morph=sdp.morph,
      .mat=uint(p.mat),
    });
  }

  return out;
}

std::vector<shaderio::Material> Scene::getMaterials(){
  std::vector<shaderio::Material> out;

  for (auto &mat : m_mat) {
    out.push_back({
      .albedo_shininess = glm::vec4(mat.albedo, mat.shininess),
      .alpha_metalness = glm::vec2(mat.roughness*mat.roughness, mat.metalness),
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


float Scene::map(glm::vec3 point, int objIdxExcluded) {
  const float iniD = 10000.0f;
  float result = iniD;

  for(int obIdx = 0; obIdx < m_root.size(); obIdx++) {
    if(obIdx == objIdxExcluded)
      continue;
    
    glm::vec3 p = glm::vec3(point);
    Node& n = m_root[obIdx];
    nvutils::Bbox& bbox = n.gp.bbox;
    Scene::GeneralParams& gp = n.gp;
    Scene::SDFParams& sdp = n.sdp;

    p = gp.tInv * glm::vec4(p, 1.0);

    p = applyRepOp(sdp.repOp, p, sdp.spacing, sdp.limit);

    p = applyDefOp(sdp.defOp, p, sdp.defP);

    p /= gp.scale;

    float d = evalPrimitive(int(gp.type), p) - sdp.roundness;

    d = d>0.0 ? applyTerrainOp(p, d, sdp.octaves, sdp.terrain, shaderio::VOXEL_SIZES[0]/10.0): d;

    d = sdp.morph>0.0 ? applyMorphOp(p,d,sdp.morphPrim,sdp.morph,sdp.roundness) : d;

    d *= gp.scale;

    result = evalCombOp(sdp.combOp, d, result, sdp.smoothness);
  }

  return result;
}

glm::vec3 Scene::evalNormal(glm::vec3 p, int objIdxExcluded) {
  const float h = 0.0001f;
  const glm::vec2 k = glm::vec2(1.0f, -1.0f);
  
  return glm::normalize(
      glm::vec3(k.x, k.y, k.y) * map(p + glm::vec3(k.x, k.y, k.y) * h, objIdxExcluded) +
      glm::vec3(k.y, k.y, k.x) * map(p + glm::vec3(k.y, k.y, k.x) * h, objIdxExcluded) +
      glm::vec3(k.y, k.x, k.y) * map(p + glm::vec3(k.y, k.x, k.y) * h, objIdxExcluded) +
      glm::vec3(k.x, k.x, k.x) * map(p + glm::vec3(k.x, k.x, k.x) * h, objIdxExcluded)
  );
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

std::vector<shaderio::BuildJob> Scene::createBaseBuildJobs(nvutils::Bbox bbox, glm::ivec3 camId0){
  const glm::ivec3 zeros(0);
  const glm::ivec3 max_index(NUM_BRICKS_PER_AXIS-1);
  const glm::ivec3 hole_min(NUM_BRICKS_PER_AXIS/4+1);
  const glm::ivec3 hole_max(NUM_BRICKS_PER_AXIS*3/4);
  
  std::vector<shaderio::BuildJob> jobs;
  
  //for(int level=CLIPMAP_LEVELS-1 ; level>=0; level--){
  for(int level=0 ; level<CLIPMAP_LEVELS; level++){
    glm::ivec3 camId = camId0>>level;

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
      glm::all(glm::lessThan(max_rel_id,hole_max))
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
    glm::ivec3 currCamId = currCamId0>>level;
    glm::ivec3 currMinId = currCamId - NUM_BRICKS_PER_AXIS/2;
    glm::ivec3 currMaxId = currCamId + NUM_BRICKS_PER_AXIS/2;
    glm::ivec3 currHMinId = currCamId - NUM_BRICKS_PER_AXIS/4 + 1;
    glm::ivec3 currHMaxId = currCamId + NUM_BRICKS_PER_AXIS/4;

    glm::ivec3 prevCamId = prevCamId0>>level;
    glm::ivec3 prevMinId = prevCamId - NUM_BRICKS_PER_AXIS/2;
    glm::ivec3 prevMaxId = prevCamId + NUM_BRICKS_PER_AXIS/2;
    glm::ivec3 prevHMinId = prevCamId - NUM_BRICKS_PER_AXIS/4 + 1;
    glm::ivec3 prevHMaxId = prevCamId + NUM_BRICKS_PER_AXIS/4;

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

      glm::ivec3 num_b = glm::abs(minId - maxId);

      out.push_back({
        .min_id_level=glm::ivec4(minId,level),
        .num_b=glm::ivec4(num_b,0)
      });

      if(level > 0){
        // Hole area -> Grid inner reach
        minId = prevHMinId;
        maxId = prevHMaxId;

        if(prevCamId[axis] <= currCamId[axis]){
          maxId[axis] = currHMinId[axis];
        }else{
          minId[axis] = currHMaxId[axis];
        }

        num_b = glm::abs(minId - maxId);

        out.push_back({
          .min_id_level=glm::ivec4(minId,level),
          .num_b=glm::ivec4(num_b,0)
        });

        // Grid inner reach -> Hole area
        minId = currHMinId;
        maxId = currHMaxId;

        if(prevCamId[axis] <= currCamId[axis]){
          minId[axis] = prevHMaxId[axis];
        }else{
          maxId[axis] = prevHMinId[axis];
        }

        num_b = glm::abs(minId - maxId);

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
  const glm::ivec3 hole_min(NUM_BRICKS_PER_AXIS/4+1);
  const glm::ivec3 hole_max(NUM_BRICKS_PER_AXIS*3/4);

  const glm::ivec3 base_min_id(buildJ.min_id_level.x,buildJ.min_id_level.y,buildJ.min_id_level.z);
  const glm::ivec3 base_num_b(buildJ.num_b.x,buildJ.num_b.y,buildJ.num_b.z);
  const glm::ivec3 max_chunk(MAX_BUILD_JOB_SIZE);

  int level = buildJ.min_id_level.w;
  std::vector<shaderio::BuildJob> out;

  for(int z = 0; z<buildJ.num_b.z; z+= MAX_BUILD_JOB_SIZE)
  for(int y = 0; y<buildJ.num_b.y; y+= MAX_BUILD_JOB_SIZE)
  for(int x = 0; x<buildJ.num_b.x; x+= MAX_BUILD_JOB_SIZE)
  {
    glm::ivec3 offset = glm::ivec3(x,y,z);
    glm::ivec3 num_b = glm::min(base_num_b-offset,max_chunk);
    glm::ivec3 min_id = base_min_id+offset;

    out.push_back({
        .min_id_level = glm::ivec4(min_id,level),
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
      aabbs.push_back(node.gp.bbox);
      aabbs.push_back(node.gp.prevBbox);
      node.gp.prevBbox = nvutils::Bbox(node.gp.bbox);
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
  mat.shininess = 1.0;
  int matIdx = addMaterial(mat);

  mat = createMaterial();
  mat.name = "Red";
  mat.albedo = glm::vec3(1,0,0);
  int red = addMaterial(mat);

  mat = createMaterial();
  mat.name = "Blue";
  mat.albedo = glm::vec3(0,0,1);
  int blue = addMaterial(mat);

  // Create the scene
  Node *terrain = createNode(NodeType::Plane);
  terrain->gp.position = glm::vec3(0.0,-1.8,0.0);
  terrain->gp.scale = 10.0;
  terrain->sdp.octaves = 8;
  terrain->sdp.terrain = glm::vec4(1.5,0.35,0.08,0.28);
  updateNodeData(terrain);
  addNode(terrain);

  Node *snowMan = createNode(NodeType::Snowman);
  snowMan->gp.scale = 0.8;
  snowMan->gp.position = glm::vec3(0.0,0.0,-1.0);
  updateNodeData(snowMan);
  addNode(snowMan);

  Node *box = createNode(NodeType::Box);
  box->gp.scale = 0.2;
  box->gp.position = glm::vec3(-0.2, -0.15, -0.75);
  box->gp.rotation = glm::vec3(0.2, 0.4, 0.4);
  box->sdp.combOp = (int)CombinationOp::Union + 3;
  box->sdp.smoothness = 0.02;
  box->gp.mat = red;
  updateNodeData(box);
  addNode(box);

  Node *sphere = createNode(NodeType::Sphere);
  sphere->gp.scale = 0.2;
  sphere->gp.position = glm::vec3(0.1, 0.3, -0.9);
  sphere->gp.rotation = glm::vec3(0);
  sphere->sdp.combOp = (int)CombinationOp::Substraction;
  sphere->gp.mat = red;
  updateNodeData(sphere);
  addNode(sphere);

  Node *sphereGrid = createNode(NodeType::Sphere);
  sphereGrid->gp.position = glm::vec3(0, 0, -1.4);
  sphereGrid->gp.rotation = glm::vec3(0);
  sphereGrid->gp.scale = 0.1;
  sphereGrid->sdp.repOp = (int)RepetitionOp::LimRepetition;
  sphereGrid->sdp.spacing = glm::vec3(0.14,0.14,0);
  sphereGrid->sdp.limit = glm::ivec3(13,13,1);
  sphereGrid->gp.mat = red;
  updateNodeData(sphereGrid);
  addNode(sphereGrid);

  Node *torus = createNode(NodeType::Torus);
  torus->gp.scale = 0.2;
  torus->gp.position = glm::vec3(0.35, 0.1, -1.2);
  torus->gp.rotation = glm::vec3(0.75, 0, 0);
  torus->gp.mat = red;
  updateNodeData(torus);
  addNode(torus);

  Node *test = createNode(NodeType::Snowman);
  test->gp.scale = 0.4;
  test->gp.position = glm::vec3(1.5,1.5,1.5);
  test->gp.rotation = glm::vec3(0.0);
  updateNodeData(test);
  addNode(test);

  for(int level=1; level<3; level++){
    Node *snowManL = createNode(NodeType::Snowman);
    snowManL->gp.scale = 0.5*(1<<level);
    snowManL->gp.position = glm::vec3(-((L0_AXIS_WORLD_SIZE*0.5)*(1<<level))*3/4, 0.0, 0.0);
    snowManL->gp.rotation = glm::vec3(0.0,0.4*level,0.0);
    updateNodeData(snowManL);
    addNode(snowManL);
  }

  Node *sphere_main = createNode(NodeType::Sphere);
  sphere_main->gp.scale = 0.2;
  sphere_main->gp.position = glm::vec3(0.0,0.0,-0.5);
  sphere_main->gp.rotation = glm::vec3(0);
  sphere_main->gp.mat = blue;
  updateNodeData(sphere_main);
  addNode(sphere_main);

} 