// TODO: Clean imports
#include "scene.hpp"
#include "glm/common.hpp"
#include "glm/exponential.hpp"
#include "glm/ext/vector_int3.hpp"
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

static std::string NodeTypeNames[] = {
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
  "Terrain",
};
using deformationOpF = glm::vec3 (*)(const glm::vec3 &, const glm::vec3 &);
static deformationOpF defFTable[2] = {opNone, opElongate};

//------------------
// Helper functions
//------------------

std::string Scene::nodeTypeToString(NodeType type) {
  return NodeTypeNames[(int)type];
}

std::string Scene::getLabel(Node *n) {
  return nodeTypeToString(n->p.type) + "##" + std::to_string(n->id);
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
                                0.04);
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
        .terrain = glm::vec4(1.0,0.5,0.1,0.3)
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
    out.push_back({
      .tInv=columnMajorToRowMajor(p.tInv),
      .position=glm::vec4(p.position,0),
      .rotation=glm::vec4(p.rotation,0),
      .spacing=glm::vec4(p.spacing,0),
      .defP=glm::vec4(p.defP,0),
      .terrain=glm::vec4(p.terrain),
      .limit=glm::ivec4(p.limit,0),
      .type=int(p.type),
      .combOp=p.combOp,
      .repOp=p.repOp,
      .defOp=p.defOp,
      .scale=p.scale,
      .roundness=p.roundness,
      .smoothness=p.smoothness,
      .octaves=p.octaves,
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
  updateNodeData(box);
  addNode(box);

  Node *sphere = createNode(NodeType::Sphere);
  sphere->p.scale = 0.2;
  sphere->p.position = glm::vec3(0.1, 0.3, -0.9);
  sphere->p.rotation = glm::vec3(0);
  sphere->p.combOp = (int)CombinationOp::Substraction;
  updateNodeData(sphere);
  addNode(sphere);

  Node *sphereGrid = createNode(NodeType::Sphere);
  sphereGrid->p.position = glm::vec3(0, 0, -1.4);
  sphereGrid->p.rotation = glm::vec3(0);
  sphereGrid->p.scale = 0.1;
  sphereGrid->p.repOp = (int)RepetitionOp::LimRepetition;
  sphereGrid->p.spacing = glm::vec3(0.14,0.14,0);
  sphereGrid->p.limit = glm::ivec3(13,13,1);
  updateNodeData(sphereGrid);
  addNode(sphereGrid);

  Node *torus = createNode(NodeType::Torus);
  torus->p.scale = 0.2;
  torus->p.position = glm::vec3(0.35, 0.1, -1.2);
  torus->p.rotation = glm::vec3(0.75, 0, 0);
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