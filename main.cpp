/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// This example demonstrates a minimal Vulkan application using the NVIDIA
// Vulkan utility libraries. It creates a window displaying a single colored
// pixel that animates through the HSV color space.

// TODO: Change the comment paragraph



#include "glm/common.hpp"
#include "glm/ext/scalar_constants.hpp"
#include "nvvk/barriers.hpp"
#include <cstring>
#include <string>
#define VMA_IMPLEMENTATION
// TODO: Organize and label imports

#include <vulkan/vulkan_core.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_vulkan.h>

#include "shaders/shaderio.h"           // Shared between host and device
#include "utils/path_utils.hpp"
#include "utils/utils.hpp"
#include "utils/scene.hpp"
#include "utils/rng.hpp"

#include "_autogen/compute_tracing.slang.h"
#include "_autogen/lighting.slang.h"
#include "_autogen/raytracing.slang.h"
#include "_autogen/brick.slang.h"
#include "_autogen/build.slang.h"
#include "_autogen/ao.slang.h"
#include "_autogen/bilateral_h.slang.h"
#include "_autogen/bilateral_v.slang.h"
#include "_autogen/simulation.slang.h"
#include "_autogen/sky_simple.slang.h"
#include "_autogen/tonemapper.slang.h"

#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/bounding_box.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>              // Timers for profiling
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/gbuffers.hpp>                // GBuffer management
#include <nvslang/slang.hpp>              // Slang compiler
#include "nvvk/descriptors.hpp"           // Descriptor set management
#include <nvapp/elem_camera.hpp>           // Camera manipulator
#include <nvutils/camera_manipulator.hpp>
#include <nvgui/tonemapper.hpp>            // Tonemapper widget
#include <nvshaders_host/tonemapper.hpp>   // Tonemapper shader
#include <nvgui/camera.hpp>                // Camera widget
#include <nvvk/formats.hpp>
#include <nvvk/shaders.hpp>
#include <nvvk/pipeline.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>
#include <cstdint>
#include <vector>
#include <glm/vector_relational.hpp>
#include <nvvk/resources.hpp>
#include <numeric>
#include <nvvk/validation_settings.hpp>  

const char* DebugModes[] = {
    "Debug color",
    "Albedo",
    "Normal",
    "Depth",
    "Shadow",
    "Position",
    "AO",
    "Bounding boxes",
};

const char* DebugPalettes[] = {
    "Magma",
    "Warm ice",
    "Viridis",
    "Plasma",
    "Turbo",
    "Inferno",
};

const char* TracingModes[] = {
    "Compute",
    "RTX",
    "Map",
};

const char* NormalModes[] = {
    "Analytic",
    "Tethrahedron",
};

class AppElement : public nvapp::IAppElement
{
  enum
  {
    eImgNormal,
    eImgAlbedo,
    eImgRendered,
    eImgTonemapped,
    eImgShadow,
    eImgShadowScratch,
    eImgPosition,
    eImgAO,
    eImgAOScratch,
  };

public:
  struct Info{
    nvutils::ProfilerManager*   profilerManager{};
    nvutils::ParameterRegistry* parameterRegistry{};
  };

  struct Pipeline{
    VkPipeline        pipeline{};
    VkPipelineLayout  layout{};
    VkShaderModule    shader{};
  };

  struct RWBuffer{
    nvvk::Buffer  nvbuffer{};
    void*         mappedData = nullptr;
    uint          count = 0;
  };

  AppElement(const Info& info)
      : m_info(info)
  {
    // Add run parameter example
    //m_info.parameterRegistry->add({"animate"}, &m_animate);
  }

  ~AppElement() override = default;

  void onAttach(nvapp::Application* app) override{
    SCOPED_TIMER(__FUNCTION__);

    // Save the application pointer
    m_app = app;

    // Get ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    m_rtProperties.pNext = &m_asProperties;
    prop2.pNext          = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);


    // Initialize allocator
    VmaAllocatorCreateInfo allocatorInfo = {
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4, 
    };
    NVVK_CHECK(m_alloc.init(allocatorInfo));

    // Initialize core components
    m_samplerPool.init(app->getDevice());
    m_stagingUploader.init(&m_alloc, true);
    m_sbtGen.init(app->getDevice(),m_rtProperties);



    // TODO set back to on when proper lighting solution is made
    // Set tonemapping off by default
    m_tonemapperData.isActive = 0;

    setupSlangCompiler();           // Setup slang compiler with correct build config flags
    createSimResources();            // Create the command pool and fence used for simulation
    createScene();                  // Create the scene and fill it up with sdfs
    setupGBuffers();                // Set up the GBuffers to render to
    createRNGTextures();            // Creates the different rng and noise textures used in the shaders
    create3DTextures();             // Creates the different 3d textures used to store voxel grid data
    createAccelerationStructures(); // Creates the bLas and tLas needed for the rt pipeline 
    createDescriptorSetLayout();    // Create the descriptor set layout for the pipelines
    compileShaders();               // Creates and compiles the shaders modules
    createPipelines();              // Create the pipelines

    // Initialize the tonemapper with proe-compiled shader
    m_tonemapper.init(&m_alloc, std::span<const uint32_t>(tonemapper_slang));

    // Init profiler with a single queue
    m_graphicsTimeline = m_info.profilerManager->createTimeline({"Graphics"});
    m_profilerGpuTimer.init(m_graphicsTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
    }

  void destroyPipeline(Pipeline* p){
    VkDevice device = m_app->getDevice();
    vkDestroyPipeline(device,p->pipeline,nullptr);
    vkDestroyPipelineLayout(device,p->layout,nullptr);
  }

  //-------------------------------------------------------------------------------
  // Destroy all elements that were created
  // - Called when the application is shutting down
  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

    VkDevice device = m_app->getDevice();

    m_descPack.deinit();

    destroyPipeline(&m_tracingPipeline);
    destroyPipeline(&m_lightingPipeline);
    destroyPipeline(&m_rtPipeline);
    destroyPipeline(&m_brickJobPipeline);
    destroyPipeline(&m_buildJobPipeline);
    destroyPipeline(&m_aoPipeline);
    destroyPipeline(&m_bilateralHPipeline);
    destroyPipeline(&m_bilateralVPipeline);
    destroyPipeline(&m_simulationPipeline);

    vkDestroyShaderModule(device,m_tracingPipeline.shader,nullptr);
    vkDestroyShaderModule(device,m_lightingPipeline.shader,nullptr);
    vkDestroyShaderModule(device,m_rtPipeline.shader,nullptr);
    vkDestroyShaderModule(device,m_brickJobPipeline.shader,nullptr);
    vkDestroyShaderModule(device,m_buildJobPipeline.shader,nullptr);
    vkDestroyShaderModule(device,m_aoPipeline.shader,nullptr);
    vkDestroyShaderModule(device,m_bilateralHPipeline.shader,nullptr);
    vkDestroyShaderModule(device,m_bilateralVPipeline.shader,nullptr);
    vkDestroyShaderModule(device,m_simulationPipeline.shader,nullptr);

    m_alloc.destroyAcceleration(m_bLas);
    m_alloc.destroyAcceleration(m_tLas);

    m_alloc.destroyBuffer(m_sceneInfoB);
    m_alloc.destroyBuffer(m_sceneAabbB);
    m_alloc.destroyBuffer(m_sceneObjectsB);
    m_alloc.destroyBuffer(m_sceneMaterialsB);
    m_alloc.destroyBuffer(m_sceneDynamicObjects.nvbuffer);
    
    m_alloc.destroyBuffer(m_buildJobQueue);
    m_alloc.destroyBuffer(m_brickJobQueue);
   
    m_alloc.destroyBuffer(m_countersB);
    m_alloc.destroyBuffer(m_indirectB);
    m_alloc.destroyBuffer(m_freeListB);

    m_alloc.destroyImage(m_noiseTex);
    m_alloc.destroyBuffer(m_aoKernelsB);
    m_alloc.destroyBuffer(m_shadowKernelsB);

    m_alloc.destroyImage(m_clipMap);
    m_alloc.destroyImage(m_brickAtlas);
    m_alloc.destroyImage(m_matAtlas);
    
    m_alloc.destroyBuffer(m_tLasB);
    m_alloc.destroyBuffer(m_bLasB);
    m_alloc.destroyBuffer(m_instancesB);
    m_alloc.destroyBuffer(m_sbtB);

    m_gBuffers.deinit();
    m_sbtGen.deinit();
    m_stagingUploader.deinit();
    m_tonemapper.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
    m_profilerGpuTimer.deinit();
    m_info.profilerManager->destroyTimeline(m_graphicsTimeline);
  }

  //---------------------------------------------------------------------------------------------------------------
  // Rendering all UI elements, this includes the image of the GBuffer, the camera controls, and the sky parameters.
  // - Called every frame
  void onUIRender() override
  { 
    ImGui::Begin("Settings");
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);

    // Add window information
    const VkExtent2D& appViewportSize = m_app->getViewportSize();
    ImGui::Text("Viewport Size: %d x %d", appViewportSize.width, appViewportSize.height);

    if(ImGui::CollapsingHeader("Camera"))
        nvgui::CameraWidget(m_cameraManip);

    if(ImGui::CollapsingHeader("Tonemapper"))
        nvgui::tonemapperWidget(m_tonemapperData);

    if(!ImGui::CollapsingHeader("Tracing")){
      ImGui::Checkbox("RTX", &m_rtxON);
    }

    if(ImGui::CollapsingHeader("Lighting data")){
      bool dirtyLight = false;
      ImGui::Text("Directional Light");

      dirtyLight |= ImGui::SliderFloat3("Direction", &m_pushConst.lp.lightDir.x, -1.0f, 1.0f);
      dirtyLight |= ImGui::ColorEdit3("Zenith Color", &m_zenithColor.x);
      dirtyLight |= ImGui::ColorEdit3("Horizon Color", &m_horizonColor.x);
      ImGui::SliderFloat("Power", &m_pushConst.lp.lightPower,0.0,10.0f);

      if(dirtyLight){
        m_pushConst.lp.lightColor = glm::mix(
          m_horizonColor,m_zenithColor,
          glm::max(0.0f,glm::dot(m_pushConst.lp.lightDir,glm::vec3(0,1,0))));
        m_pushConst.lp.lightDir = glm::normalize(m_pushConst.lp.lightDir);
      }

      ImGui::Separator();
      ImGui::Text("Ambient Hemispheric");
      ImGui::ColorEdit3("Ambient Top", &m_pushConst.lp.ambientTop.x);
      ImGui::ColorEdit3("Ambient Bottom", &m_pushConst.lp.ambientBottom.x);

      ImGui::Separator();
      ImGui::Text("Fog");
      ImGui::SliderFloat("Fog Density", &m_pushConst.lp.fogDensity, 0.0f, 0.2f);
      ImGui::ColorEdit3("Fog Color", &m_pushConst.lp.fogColor.x);
     
      ImGui::Separator();
      ImGui::Text("Ambient occlussion");
      m_refreshAOkernels |= ImGui::SliderFloat("Radius", &m_pushConst.lp.aoRadius, 0.0f, 5.0f);
      ImGui::SliderFloat("Bias", &m_pushConst.lp.aoBias, 0.0f, 0.01f);
      ImGui::SliderInt("Samples##AO", &m_pushConst.lp.aoSamples, 1, MAX_NUM_AO_KERNELS);
      ImGui::SliderInt("Texel size##AO", &m_pushConst.lp.aoTexelSize, 1, 20);

      ImGui::Separator();
      ImGui::Text("Shadows");
      m_refreshShadowKernels |= ImGui::SliderFloat("Sharpness", &m_pushConst.lp.shadowSharpness, 1.0f, 500.0f);
      ImGui::SliderInt("Samples##Shadow", &m_pushConst.lp.shadowSamples, 1, MAX_NUM_SHADOW_KERNELS);
      ImGui::SliderInt("Texel size##Shadow", &m_pushConst.lp.shadowTexelSize, 1, 20);
    }
    
    if(!ImGui::CollapsingHeader("Debug colors")){
      ImGui::Checkbox("Atlas%", &m_pushConst.debug.brickPercent);
      ImGui::Checkbox("Active", &m_debugActive);
      ImGui::Combo("Mode", &m_debugMode, DebugModes, IM_ARRAYSIZE(DebugModes));
      ImGui::Combo("Palette", &m_pushConst.debug.palette, DebugPalettes, IM_ARRAYSIZE(DebugPalettes));
      if(m_debugActive){
        m_pushConst.debug.mode = m_debugMode + 1;
      }else{
        m_pushConst.debug.mode = 0;
      }

      ImGui::Text("Camera id0: %i,%i,%i",m_sceneInfo.cameraId0.x,m_sceneInfo.cameraId0.y,m_sceneInfo.cameraId0.z);
      if(ImGui::Button("Reset TLas")){
        m_rebuildTlas = true;
      }
    }

    ImGui::End();

    // Draw scene tree and object tab
    m_scene.draw();

    ImGui::Begin("Viewport");

      ImVec2 viewportPos = ImGui::GetCursorScreenPos();
      ImVec2 viewportSize = ImGui::GetContentRegionAvail();

      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(eImgTonemapped),viewportSize);

      const glm::mat4& viewMatrix = m_cameraManip->getViewMatrix();
      const glm::mat4& projMatrix = m_cameraManip->getPerspectiveMatrix();
      static glm::vec3 prevCamCenter = m_cameraManip->getCenter();
      static glm::vec3 prevCamEye = m_cameraManip->getEye();
      
      if(m_scene.m_usingGuizmo){
        m_cameraManip->setCenter(prevCamCenter);
        m_cameraManip->setEye(prevCamEye);
      }else{
        prevCamCenter = m_cameraManip->getCenter();
        prevCamEye = m_cameraManip->getEye();
      }
      
      m_scene.drawGuizmo(viewportPos, viewportSize, viewMatrix, projMatrix);

      
    ImGui::End();
  }

  //---------------------------------------------------------------------------------------------------------------
  // This renders the toolbar of the window
  // - Called when the ImGui menu is rendered
  void onUIMenu() override
  {
    bool vsync = m_app->isVsync();
    bool reload = false;

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
        m_app->close();
      ImGui::EndMenu();
    }
    if(ImGui::BeginMenu("View"))
    {
      ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &vsync);
      ImGui::EndMenu();
    }
    if(ImGui::BeginMenu("Tools"))
    {
      reload |= ImGui::MenuItem("Reload Shaders", "F5");
      ImGui::EndMenu();
    }
    reload |= ImGui::IsKeyPressed(ImGuiKey_F5);
    if(reload){
      vkQueueWaitIdle(m_app->getQueue(0).queue);
      reloadShaders();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      m_app->close();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_V) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl) && ImGui::IsKeyDown(ImGuiKey_LeftShift))
    {
      vsync = !vsync;
    }

    if(vsync != m_app->isVsync())
    {
      m_app->setVsync(vsync);
    }


  }

  void onPreRender() override { m_graphicsTimeline->frameAdvance(); }

  //---------------------------------------------------------------------------------------------------------------
  // When the viewport is resized, the GBuffer must be resized
  // - Called when the Window "viewport is resized
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { 
    NVVK_CHECK(m_gBuffers.update(cmd, size)); 
    const auto profiledSection = m_profilerGpuTimer.cmdAsyncSection(cmd, "Viewport resize");

    // CRITICAL: Needs to update the descriptor set if it resizes the gbuffers
    nvvk::WriteSetContainer writeContainer;

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::renderTarget), 
      m_gBuffers.getDescriptorImageInfo(eImgRendered));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::normalBuffer), 
      m_gBuffers.getDescriptorImageInfo(eImgNormal));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::albedoBuffer), 
      m_gBuffers.getDescriptorImageInfo(eImgAlbedo));
    
    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::shadowBuffer), 
      m_gBuffers.getDescriptorImageInfo(eImgShadow));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::shadowSampler), 
      m_gBuffers.getDescriptorImageInfo(eImgShadow));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::shadowScratchBuffer), 
      m_gBuffers.getDescriptorImageInfo(eImgShadowScratch));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::shadowScratchSampler), 
      m_gBuffers.getDescriptorImageInfo(eImgShadowScratch));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::positionBuffer), 
      m_gBuffers.getDescriptorImageInfo(eImgPosition));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::aoBuffer), 
      m_gBuffers.getDescriptorImageInfo(eImgAO));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::aoScratchBuffer), 
      m_gBuffers.getDescriptorImageInfo(eImgAOScratch));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::aoSample), 
      m_gBuffers.getDescriptorImageInfo(eImgAO));

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::aoScratchSample), 
      m_gBuffers.getDescriptorImageInfo(eImgAOScratch));

    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = m_gBuffersSampler;
    samplerInfo.imageView = VK_NULL_HANDLE;
    samplerInfo.imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::gSampler), 
      samplerInfo);

    // Needs to create a descriptor image info because the GBuffer object doesn't expose a function
    VkDescriptorImageInfo depthImageInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = m_gBuffers.getDepthImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    };
    writeContainer.append(
      m_descPack.makeWrite(shaderio::BindingPoints::depthBuffer), 
      depthImageInfo);
    
    
    vkUpdateDescriptorSets(m_app->getDevice(),  
                        static_cast<uint32_t>(writeContainer.size()),  
                        writeContainer.data(), 0, nullptr);
  }

  //---------------------------------------------------------------------------------------------------------------
  // Rendering the scene
  // The scene is rendered to a GBuffer and the GBuffer is displayed in the ImGui window.
  // Only the ImGui is rendered to the swapchain image.
  // - Called every frame
  void onRender(VkCommandBuffer cmd) override{
    NVVK_DBG_SCOPE(cmd);

    

    {
      // User espcial action
      //glm::vec3 eye = m_cameraManip->getEye();
      //glm::vec3 center = m_cameraManip->getCenter();
      //m_scene.simulate(deltaT);
      //m_scene.centerCamAction(eye, glm::normalize(center-eye));
    }

    simulationPass();

    {
      const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Generation");
      const bool sceneRefresh = m_scene.m_needsRefresh || m_currCamId0 != m_prevCamId0 || m_firstFrame;
      
      
      if(sceneRefresh){
        generationPass(cmd);
        m_scene.m_needsRefresh = false;
      }else{
        // Empty timers so it doesn't break the profiler config
        { const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Build jobs"); }
        { const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Brick jobs"); }
      }
      
      if(m_rtxON && (m_updateTlas || sceneRefresh)){
        updateTopLevelAS(cmd,m_rebuildTlas);
        m_rebuildTlas = false;
      }else{
        // Empty timer so it doesn't break the profiler config
        const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Accel struct update");
      }

      // Post generation submit updates
      m_prevCamId0 = m_currCamId0;
      m_updateTlas = !m_rtxON;
    }

    if(m_rtxON){
      raytracingPass(cmd);
    }else{
      tracingPass(cmd);
    }

    lightingPass(cmd);
    
    postProcess(cmd);

    m_firstFrame = false;
    m_pushConst.frameCount++;
  }

  void bindComputePipeline(VkCommandBuffer cmd, Pipeline* pl){
    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl->pipeline);  
    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl->layout,
                            0, 1, m_descPack.getSetPtr(), 0, nullptr);  
    // Push constants
    vkCmdPushConstants(cmd, pl->layout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
  }

  void tracingPass(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);
    const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Tracing");

    // Bind pipeline
    bindComputePipeline(cmd,&m_tracingPipeline);
    // Dispatch
    VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE_2D);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
    // Wait for tracing to be done
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgAlbedo), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_GENERAL});
  }

  void lightingPass(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);
    const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Lighting");

    {
      const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "AO");

      // Bind pipeline
      bindComputePipeline(cmd,&m_aoPipeline);
      // Dispatch
      VkExtent2D viewportSize = m_gBuffers.getSize();
      viewportSize.width = viewportSize.width/m_pushConst.lp.aoTexelSize;
      viewportSize.height = viewportSize.height/m_pushConst.lp.aoTexelSize;
      VkExtent2D group_counts = nvvk::getGroupCounts(viewportSize, WORKGROUP_SIZE_2D);
      vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
      // Wait for AO to be done
      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgAO), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_GENERAL});
    }

    {
      const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Bilateral blur");
      // Bind pipeline
      bindComputePipeline(cmd,&m_bilateralHPipeline);
      // Dispatch
      VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE_2D);
      vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
      // Wait for target to be done
      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgAOScratch), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_GENERAL});
      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgShadowScratch), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_GENERAL});

      // Bind pipeline
      bindComputePipeline(cmd,&m_bilateralVPipeline);
      // Dispatch
      vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
      // Wait for target to be done
      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgAO), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_GENERAL});
      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgShadow), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_GENERAL});
    }

    {
      const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Shading");
      // Bind pipeline
      bindComputePipeline(cmd,&m_lightingPipeline);
      // Dispatch
      VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE_2D);
      vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
      // Wait for render target to be done
      nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgRendered), VK_IMAGE_LAYOUT_GENERAL,
                                        VK_IMAGE_LAYOUT_GENERAL});
    }
  }

  void raytracingPass(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);
    const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Tracing");

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline.pipeline);  
    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline.layout,
                            0, 1, m_descPack.getSetPtr(), 0, nullptr);  
    // Push constants
    vkCmdPushConstants(cmd, m_rtPipeline.layout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
    // Dispatch
    const VkExtent2D& group_counts = m_app->getViewportSize();
    auto& sbt = m_sbtGen.getSBTRegions();
    vkCmdTraceRaysKHR(cmd,
      &sbt.raygen,
      &sbt.miss,
      &sbt.hit,
      &sbt.callable,
      group_counts.width, 
      group_counts.height,
      1
    );
    // Wait for tracing to be done
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgAlbedo), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_GENERAL});
  }

  void executeBrickJobs(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);
    const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Brick jobs");

    // Bind pipeline
    bindComputePipeline(cmd,&m_brickJobPipeline);
    // Dispatch using buffer
    vkCmdDispatchIndirect(
      cmd,
      m_indirectB.buffer,
      0
    );
  
    nvvk::cmdImageMemoryBarrier(cmd, {m_brickAtlas.image, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_GENERAL});
  }

  void executeBuildJobs(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);
    const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Build jobs");

    // Update the next brick job index
    std::vector<uint32_t> counters(1,0);
    unsigned long size = counters.size() * sizeof(uint32_t);
    vkCmdUpdateBuffer(cmd, m_countersB.buffer, 0, size, counters.data());
    nvvk::cmdBufferMemoryBarrier(cmd, {m_countersB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});

    // Update the indirect dispatch group count buffer
    std::vector<shaderio::DispatchIndirectCommand> indirectV(1,{
      .x=BRICK_JOB_GROUP_X_DISPATCH_SIZE,
      .y=0,.z=1,._pad=0
    });
    size = indirectV.size() * sizeof(shaderio::DispatchIndirectCommand);
    vkCmdUpdateBuffer(cmd, m_indirectB.buffer, 0, size, indirectV.data());
    nvvk::cmdBufferMemoryBarrier(cmd, {m_indirectB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});

    int num_bricks;
    std::vector<shaderio::BuildJob> buildJobs;
    buildJobs = m_scene.getBuildJobs(m_currCamId0,m_prevCamId0);
    //buildJobs = m_scene.getDenseBuildJobs(m_currCamId0,m_prevCamId0);

    if(buildJobs.size() > shaderio::MAX_NUM_BUILD_JOBS)
      LOGE("Not enough space in build job queue to allocale %zu jobs\n",buildJobs.size());
    
    if(buildJobs.size() <= 0){
      //LOGW("Build job queue update size is 0, skipping generation pass\n");
      return;
    }

    size = buildJobs.size() * sizeof(shaderio::BuildJob);
    m_stagingUploader.appendBuffer(m_buildJobQueue,0,size,buildJobs.data());
    m_stagingUploader.cmdUploadAppended(cmd);

    nvvk::cmdBufferMemoryBarrier(cmd, {m_buildJobQueue.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});

    // Bind pipeline
    bindComputePipeline(cmd,&m_buildJobPipeline);
    // Dispatch
    vkCmdDispatch(cmd, 1, 1, buildJobs.size());
  
    nvvk::cmdBufferMemoryBarrier(cmd, {m_brickJobQueue.buffer, 
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});
  }

  void generationPass(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    executeBuildJobs(cmd);
    executeBrickJobs(cmd);
  }

  // Apply post-processing
  void postProcess(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);
    const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Post process");

    // Wait for render target to be done
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgAlbedo), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_GENERAL});

    // No img layout transition needed
    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(eImgRendered),
                            m_gBuffers.getDescriptorImageInfo(eImgTonemapped));

    // Wait for render target to be done
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_GENERAL});
  }

  void simulationPass(){
    static VkCommandBuffer simCmd;

    if(m_firstFrame){
      VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_simCmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
      };

      vkAllocateCommandBuffers(m_app->getDevice(), &allocInfo, &simCmd);
    }

    VkCommandBufferBeginInfo beginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
    };
    //vkResetCommandBuffer(simCmd, 0);
    NVVK_CHECK(vkResetCommandPool(m_app->getDevice(), m_simCmdPool, 0));

    vkBeginCommandBuffer(simCmd, &beginInfo);

    NVVK_DBG_SCOPE(simCmd);

    {
      const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(simCmd, "Buffer updates");

      // Time variable updates
      m_pushConst.time = static_cast<float>(ImGui::GetTime());
      if(m_prevTime < 0){
        m_prevTime = m_pushConst.time;
      }else{
        m_pushConst.dts = (m_pushConst.time - m_prevTime)/SIM_NUM_SUBSTEPS;
        m_prevTime = m_pushConst.time;
      }

      // Dynamic objects processing
      readAndUpdateDynamicObjects(simCmd);

      // Cam and scene info update
      updateSceneBuffer(simCmd);
      updateSceneObjects(simCmd);

      // Kernels update
      if(m_refreshAOkernels || m_firstFrame){
        updateAOkernels(simCmd);
        m_refreshAOkernels = false;
      }
      if(m_refreshShadowKernels || m_firstFrame){
        updateShadowKernels(simCmd);
        m_refreshShadowKernels = false;
      }

      m_test++;
      vkCmdUpdateBuffer(simCmd, m_sceneDynamicObjects.nvbuffer.buffer, 0, sizeof(uint), &m_test);

    }


    {
      const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(simCmd, "Simulation");
      // Bind pipeline
      bindComputePipeline(simCmd,&m_simulationPipeline);
      // Dispatch
      VkExtent2D group_counts(1,1);
      vkCmdDispatch(simCmd, group_counts.width, group_counts.height, 1);

      nvvk::cmdBufferMemoryBarrier(simCmd, {m_sceneDynamicObjects.nvbuffer.buffer, 
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});
    }


    NVVK_CHECK(vkEndCommandBuffer(simCmd));

    VkSubmitInfo submitInfo = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &simCmd
    };

    NVVK_CHECK(vkQueueSubmit(m_app->getQueue(0).queue, 1, &submitInfo, m_simFence));
  }

  void setupSlangCompiler(){
#ifdef NDEBUG
    SCOPED_TIMER(std::string(__FUNCTION__)+": RELEASE configuration");
#else
    SCOPED_TIMER(std::string(__FUNCTION__)+": DEBUG configuration");
#endif

    m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_slangCompiler.defaultTarget();
    m_slangCompiler.defaultOptions();
#ifdef NDEBUG
    m_slangCompiler.addOption({slang::CompilerOptionName::Optimization,
        { slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_MAXIMAL }
    });
    m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation,
        { slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_NONE }
    });
    m_slangCompiler.addOption({slang::CompilerOptionName::WarningsAsErrors,
        { slang::CompilerOptionValueKind::Int, 1 }
    });
    m_slangCompiler.addMacro({"NDEBUGSHADER","1"});
#else
    m_slangCompiler.addOption({slang::CompilerOptionName::Optimization,
        { slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_DEFAULT }
    });
    m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation,
        { slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_STANDARD }
    });
    m_slangCompiler.addOption({slang::CompilerOptionName::WarningsAsErrors,
        { slang::CompilerOptionValueKind::Int, 0 }
    });
#endif
  }

  void setupGBuffers(){
    SCOPED_TIMER(__FUNCTION__);

    // Acquiring the texture sampler which will be used for displaying the GBuffer
    NVVK_CHECK(m_samplerPool.acquireSampler(m_gBuffersSampler));
    NVVK_DBG_NAME(m_gBuffersSampler);

    // Create the G-Buffers
    nvvk::GBufferInitInfo gBufferInit{
        .allocator      = &m_alloc,
        .colorFormats   = {
          VK_FORMAT_R8G8B8A8_SNORM,           // Normal buffer
          VK_FORMAT_R8G8B8A8_UNORM,           // Albedo buffer
          VK_FORMAT_R32G32B32A32_SFLOAT,      // Render target
          VK_FORMAT_R8G8B8A8_UNORM,           // Tonemapped
          VK_FORMAT_R8_UNORM,                 // Shadow buffer
          VK_FORMAT_R8_UNORM,                 // Shadow Scratch buffer
          VK_FORMAT_R32G32B32A32_SFLOAT,      // Position buffer
          VK_FORMAT_R8_UNORM,                 // AO buffer
          VK_FORMAT_R8_UNORM,                 // AO Scratch buffer
        },          
        .depthFormat    = nvvk::findDepthFormat(m_app->getPhysicalDevice()),
        .imageSampler   = m_gBuffersSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    };
    m_gBuffers.init(gBufferInit);
  }

  void create2DTexture(nvvk::Image& image, VkExtent3D extent, VkFormat format){
     // Destroy if already created
    m_alloc.destroyImage(image);

    VkImageCreateInfo ci = DEFAULT_VkImageCreateInfo;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = extent;
    ci.mipLevels = 1;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

    VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;
    vi.image = image.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;

    NVVK_CHECK(m_alloc.createImage(image, ci, vi));

    // Setup sampler to be ortholinear storage with no interpolation and repetition
    VkSamplerCreateInfo si = DEFAULT_VkSamplerCreateInfo;
    si.magFilter    = VK_FILTER_NEAREST;
    si.minFilter    = VK_FILTER_NEAREST;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    NVVK_CHECK(m_samplerPool.acquireSampler(image.descriptor.sampler, si));

    image.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  void createRNGTextures(){
    SCOPED_TIMER(__FUNCTION__);
    
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

      // Noise tex
      VkExtent3D extent = {NOISE_TEX_SIZE,NOISE_TEX_SIZE,1};  // XYZ size
      VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;  // Texel format
      create2DTexture(m_noiseTex, extent, format);
      NVVK_DBG_NAME(m_noiseTex.image);
      
      int size = NOISE_TEX_SIZE*NOISE_TEX_SIZE*4;
      std::vector<float> noise;
      noise.reserve(size);
      for(int i = 0; i < size; i++){
        noise.push_back(randomFloat2());
      }
      NVVK_CHECK(m_stagingUploader.appendImage(m_noiseTex,std::span(noise)));
      
      m_stagingUploader.cmdUploadAppended(cmd);  // Upload the scene information to the GPU

    m_app->submitAndWaitTempCmdBuffer(cmd); 
  }

  void create3DStorageTexture(nvvk::Image& image, VkExtent3D extent, VkFormat format, VkClearColorValue clearColor){
    // Destroy if already created
    m_alloc.destroyImage(image);


    std::array<uint32_t, 1> queueFamilies = {
        m_app->getQueue(0).familyIndex,
    };

    VkImageCreateInfo ci = DEFAULT_VkImageCreateInfo;
    ci.imageType = VK_IMAGE_TYPE_3D;
    ci.format = format;
    ci.extent = extent;
    ci.mipLevels = 1;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT; // Read/write texture
    // ci.queueFamilyIndexCount = 2;
    // ci.pQueueFamilyIndices = queueFamilies; // TODO: Check if necesary

    VkImageViewCreateInfo vi = DEFAULT_VkImageViewCreateInfo;
    vi.image = image.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vi.format = format;

    NVVK_CHECK(m_alloc.createImage(image, ci, vi));

    // Setup sampler to be ortholinear storage with no interpolation or repetition
    VkSamplerCreateInfo si = DEFAULT_VkSamplerCreateInfo;
    si.magFilter    = VK_FILTER_NEAREST;
    si.minFilter    = VK_FILTER_NEAREST;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

    NVVK_CHECK(m_samplerPool.acquireSampler(image.descriptor.sampler, si));

    image.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    nvvk::cmdImageMemoryBarrier(cmd, {image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);

    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Debugging information
    NVVK_DBG_NAME(image.image);
    NVVK_DBG_NAME(image.descriptor.sampler);
    NVVK_DBG_NAME(image.descriptor.imageView);
  }

  void create3DTextures(){
    SCOPED_TIMER(__FUNCTION__);

    // Clipmap
    VkExtent3D extent = {NUM_BRICKS_PER_AXIS,NUM_BRICKS_PER_AXIS,NUM_BRICKS_PER_AXIS*CLIPMAP_LEVELS};  // XYZ size
    VkFormat format = VK_FORMAT_R32_UINT;  // Texel format
    uint32_t clearValueClip = shaderio::UNIFORM_POSITIVE_BRICK_POINTER;
    VkClearColorValue clearColor = {.uint32={clearValueClip,clearValueClip,clearValueClip,clearValueClip}};
    create3DStorageTexture(m_clipMap, extent, format, clearColor);
    NVVK_DBG_NAME(m_clipMap.image);

    // Brick atlas
    const int atlas_axis_size = BRICK_PER_ATLAS_AXIS*BRICK_SIZE;
    extent = {atlas_axis_size,atlas_axis_size,BRICK_SIZE};  // XYZ size
    format = VK_FORMAT_R8_SNORM;  // Texel format
    glm::float32 clearValueF = 1.0f;
    clearColor = {.float32={clearValueF,clearValueF,clearValueF,clearValueF}};
    create3DStorageTexture(m_brickAtlas, extent, format, clearColor);
    NVVK_DBG_NAME(m_brickAtlas.image);

    // Material atlas
    const int mat_atlas_axis_size = BRICK_PER_ATLAS_AXIS*MAT_PER_BRICK_AXIS;
    extent = {mat_atlas_axis_size,mat_atlas_axis_size,MAT_PER_BRICK_AXIS};  // XYZ size
    format = VK_FORMAT_R8G8B8A8_UNORM;  // Texel format
    clearValueF = 1.0f;
    clearColor = {.float32={clearValueF,clearValueF,clearValueF,clearValueF}};
    create3DStorageTexture(m_matAtlas, extent, format, clearColor);
    NVVK_DBG_NAME(m_matAtlas.image);
  }

  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const uint32_t aabbCount){
    nvvk::AccelerationStructureGeometryInfo result = {};

    // Describe buffer as array of VkAabbPostions
    VkAccelerationStructureGeometryAabbsDataKHR aabbs{
      .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
      .data   = {.deviceAddress = m_bLasB.address},
      .stride = sizeof(VkAabbPositionsKHR)
    };

    // Identify the above data as containing opaque triangles.
    result.geometry = VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
        .geometry     = {.aabbs = aabbs},
        .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = aabbCount};

    return result;
  }

  void createAccelerationStructure(VkAccelerationStructureTypeKHR asType,           // The type of acceleration structure (BLAS or TLAS)
                                  nvvk::AccelerationStructure& accelStruct,         // The acceleration structure to create
                                  nvvk::AccelerationStructureGeometryInfo& geoInfo, // The geometry and range to build the acceleration structure from
                                  VkBuildAccelerationStructureFlagsKHR flags,       // Build flags (e.g. prefer fast trace)
                                  nvvk::Buffer& scratchBuffer                       // Scratch buffer that will be used to build de as
  )
  {
    VkDevice device = m_app->getDevice();

    // Helper function to align a value to a given alignment
    auto alignUp = [](auto value, size_t alignment) noexcept { return ((value + alignment - 1) & ~(alignment - 1)); };

    // Fill the build information with the current information, the rest is filled later (scratch buffer and destination AS)
    VkAccelerationStructureBuildGeometryInfoKHR asBuildInfo{
        .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type          = asType,  // The type of acceleration structure (BLAS or TLAS)
        .flags         = flags,   // Build flags (e.g. prefer fast trace)
        .mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,  // Build mode vs update
        .geometryCount = 1,     // Deal with one geometry at a time
        .pGeometries   = &geoInfo.geometry,  // The geometry to build the acceleration structure from
        
    };

    // One geometry at a time (could be multiple)
    std::vector<uint32_t> maxPrimCount(1);
    maxPrimCount[0] = geoInfo.rangeInfo.primitiveCount;

    // Find the size of the acceleration structure and the scratch buffer
    VkAccelerationStructureBuildSizesInfoKHR asBuildSize{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &asBuildInfo,
                                            maxPrimCount.data(), &asBuildSize);

    // Make sure the scratch buffer is properly aligned
    VkDeviceSize scratchSize = alignUp(asBuildSize.buildScratchSize, m_asProperties.minAccelerationStructureScratchOffsetAlignment);

    // Create the scratch buffer to store the temporary data for the build
    m_alloc.destroyBuffer(scratchBuffer);
    NVVK_CHECK(m_alloc.createBuffer(scratchBuffer, scratchSize,
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                            | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));
    // Create the acceleration structure
    VkAccelerationStructureCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .size  = asBuildSize.accelerationStructureSize,  // The size of the acceleration structure
        .type  = asType,  // The type of acceleration structure (BLAS or TLAS)
    };
    NVVK_CHECK(m_alloc.createAcceleration(accelStruct, createInfo));

    // Build the acceleration structure
    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();

      // Fill with new information for the build,scratch buffer and destination AS
      asBuildInfo.dstAccelerationStructure  = accelStruct.accel;
      asBuildInfo.scratchData.deviceAddress = scratchBuffer.address;

      VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo = &geoInfo.rangeInfo;
      vkCmdBuildAccelerationStructuresKHR(cmd, 1, &asBuildInfo, &pBuildRangeInfo);

      m_app->submitAndWaitTempCmdBuffer(cmd);
    }
  }

  void createBottomLevelAS(nvvk::AccelerationStructureGeometryInfo geoInfo){
    //SCOPED_TIMER(__FUNCTION__);

    nvvk::Buffer scratch;
    createAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, m_bLas, geoInfo, 
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | 
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
      scratch);
    m_alloc.destroyBuffer(scratch);
    NVVK_DBG_NAME(m_bLas.accel);
  }

  void createTopLevelAS(){
    SCOPED_TIMER(__FUNCTION__);

    m_alloc.destroyAcceleration(m_tLas);

    nvvk::AccelerationStructureGeometryInfo geoInfo{};
    VkAccelerationStructureGeometryInstancesDataKHR geometryInstances{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
      .data = {.deviceAddress = m_instancesB.address}
    };
    geoInfo.geometry = {.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                        .geometry     = {.instances = geometryInstances}};
    geoInfo.rangeInfo = {.primitiveCount = static_cast<uint32_t>(shaderio::NUM_BRICKS_IN_ATLAS)};

      
    createAccelerationStructure(
      VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, 
      m_tLas, 
      geoInfo,
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | 
      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
      m_tLasB
    );
    NVVK_DBG_NAME(m_tLas.accel);
  }

  void updateTopLevelAS(VkCommandBuffer cmd, bool rebuild){
    const auto profiledSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Accel struct update");

    if(rebuild)
      LOGI("REBUILDING TLAS\n");

    nvvk::AccelerationStructureGeometryInfo geoInfo{};
    VkAccelerationStructureGeometryInstancesDataKHR geometryInstances{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
      .data = {.deviceAddress = m_instancesB.address}
    };
    geoInfo.geometry = {.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                        .geometry     = {.instances = geometryInstances}};
    geoInfo.rangeInfo = {.primitiveCount = static_cast<uint32_t>(shaderio::NUM_BRICKS_IN_ATLAS)};

    VkAccelerationStructureBuildGeometryInfoKHR asBuildInfo{
        .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags         = 
          VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | 
          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
        .mode          = rebuild ?
          VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR: VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR,
        .srcAccelerationStructure = rebuild ? VK_NULL_HANDLE : m_tLas.accel,
        .dstAccelerationStructure = m_tLas.accel,
        .geometryCount = 1,
        .pGeometries   = &geoInfo.geometry,
        .scratchData = {.deviceAddress = m_tLasB.address}
    };

    VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
      .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
    };

    vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      0,
      1, &barrier,
      0, nullptr,
      0, nullptr);

    VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo = &geoInfo.rangeInfo;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &asBuildInfo, &pBuildRangeInfo);
  
    barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
    };

    vkCmdPipelineBarrier(
      cmd,
      VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      0,
      1, &barrier,
      0, nullptr,
      0, nullptr);
  }

  void createInstanceBuffer(){
    const int instanceCount = shaderio::NUM_BRICKS_IN_ATLAS;
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    for(int i = 0; i < instanceCount; i++) {
        VkAccelerationStructureInstanceKHR ray_inst{};
        ray_inst.transform = nvvk::toTransformMatrixKHR(glm::mat4(1.0f));
        ray_inst.accelerationStructureReference = m_bLas.address;
        ray_inst.instanceCustomIndex = i;
        ray_inst.mask = 0x00;
        ray_inst.instanceShaderBindingTableRecordOffset = 0;
        ray_inst.flags = 0;
        tlasInstances.push_back(ray_inst);
    }    

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_stagingUploader.appendBuffer(m_instancesB, 0, std::span(tlasInstances));
      m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createAccelerationStructures(){
    SCOPED_TIMER(__FUNCTION__);

    createBottomLevelAS(primitiveToGeometry(1));
    
    createInstanceBuffer();
    
    createTopLevelAS();
  }

  void updateSceneObjects(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    m_scene.flushDeletedNodes();
    std::vector<nvutils::Bbox> aabbVector = m_scene.getAllBboxes();
    std::vector<shaderio::SceneObject> objectsVector = m_scene.getObjects();
    std::vector<shaderio::Material> materialsVector = m_scene.getMaterials();

    if(aabbVector.size() != objectsVector.size())
        LOGE("Aabb vector is diferent size from objects vector %zu != %zu\n",aabbVector.size(),objectsVector.size());
    if(aabbVector.size()>MAX_SCENE_OBJECTS)
      LOGE("Number of scene objects exceeds maximum %zu > %i\n",aabbVector.size(),MAX_SCENE_OBJECTS);
    if(materialsVector.size()>MAX_MATERIALS)
      LOGE("Number of scene materials exceeds maximum %zu > %i\n",materialsVector.size(),MAX_MATERIALS);

    if(aabbVector.size() == 0){
      aabbVector.push_back({});
      objectsVector.push_back({});
      m_pushConst.numObjects = 0;
    }else{
      m_pushConst.numObjects = aabbVector.size();
    }

    unsigned long size = aabbVector.size() * sizeof(nvutils::Bbox);
    vkCmdUpdateBuffer(cmd, m_sceneAabbB.buffer, 0, size, aabbVector.data());

    size = objectsVector.size() * sizeof(shaderio::SceneObject);
    vkCmdUpdateBuffer(cmd, m_sceneObjectsB.buffer, 0, size, objectsVector.data());
    
    size = materialsVector.size() * sizeof(shaderio::Material);
    vkCmdUpdateBuffer(cmd, m_sceneMaterialsB.buffer, 0, size, materialsVector.data());

    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneAabbB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneObjectsB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
  }

  void updateAOkernels(VkCommandBuffer cmd){
    int size = MAX_NUM_AO_KERNELS;
    std::vector<glm::vec3> kernels;
    kernels.reserve(size);
    for(int i = 0; i < size; i++){
      glm::vec3 sample;
      bool degenerate = true;
      
      while(degenerate) {
        sample = glm::vec3(
            randomFloat2(),
            randomFloat2(),
            randomFloat1()
        );

        if(abs(sample.x) < 0.01f || abs(sample.y) < 0.01f || abs(sample.z) < 0.1f) degenerate = true;
        else degenerate = false;
      }
      
      sample = glm::normalize(sample);
      float scale = float(i) / size;
      scale = glm::mix(0.1f, 1.0f, scale * scale);
      sample *= scale;
      sample *= m_pushConst.lp.aoRadius;
      sample.z = glm::max(1e-4f,sample.z);

      kernels.push_back(sample);
    }
    vkCmdUpdateBuffer(cmd, m_aoKernelsB.buffer, 0, size*sizeof(glm::vec3), kernels.data());
  
    nvvk::cmdBufferMemoryBarrier(cmd, {m_aoKernelsB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
  }
  
  void updateShadowKernels(VkCommandBuffer cmd){
    glm::vec3 lightDir = m_pushConst.lp.lightDir;
    glm::vec3 aux = glm::vec3(0,0,1);
    if(glm::dot(aux,lightDir)>0.99){
      aux = glm::vec3(1,0,0);
    }
    glm::vec3 tangent = glm::normalize(glm::cross(lightDir,aux));
    glm::vec3 bitangent = glm::cross(lightDir, tangent);
    glm::mat3 TBN = glm::mat3(tangent, bitangent, lightDir);

    int size = MAX_NUM_SHADOW_KERNELS;
    std::vector<glm::vec3> kernels;
    kernels.reserve(size);

    kernels.push_back(TBN * glm::vec3(0,0,1));

    for(int i = 1; i < size; i++){
      glm::vec3 sample;

      float alpha = randomFloat1()*2.0*glm::pi<float>();
      float distance = randomFloat1();

      sample = glm::vec3(
        glm::cos(alpha)*distance,
        glm::sin(alpha)*distance,
        m_pushConst.lp.shadowSharpness
      );
        
      sample = glm::normalize(sample);

      sample = TBN*sample;

      kernels.push_back(sample);
    }
    vkCmdUpdateBuffer(cmd, m_shadowKernelsB.buffer, 0, size*sizeof(glm::vec3), kernels.data());
  
    nvvk::cmdBufferMemoryBarrier(cmd, {m_shadowKernelsB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
  }

  void createSimResources(){
    SCOPED_TIMER(__FUNCTION__);
    VkCommandPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = m_app->getQueue(0).familyIndex,
    };

    NVVK_CHECK(vkCreateCommandPool(m_app->getDevice(), &poolInfo, NULL, &m_simCmdPool));

    VkFenceCreateInfo fenceInfo = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };

    NVVK_CHECK(vkCreateFence(m_app->getDevice(), &fenceInfo, NULL, &m_simFence));
  }

  void createScene(){
    SCOPED_TIMER(__FUNCTION__);
    nvvk::ResourceAllocator* allocator = m_stagingUploader.getResourceAllocator();
    
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      // ------------------
      // Scene info buffer
      // ------------------
      NVVK_CHECK(allocator->createBuffer(m_sceneInfoB,
                                     std::span<const shaderio::SceneInfo>(&m_sceneInfo, 1).size_bytes(),
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT));
      NVVK_DBG_NAME(m_sceneInfoB.buffer);

      // ------------------
      // AABB and objects buffers
      // ------------------
      NVVK_CHECK(allocator->createBuffer(m_sceneAabbB,
                                     MAX_SCENE_OBJECTS*sizeof(nvutils::Bbox),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                        ));
      NVVK_DBG_NAME(m_sceneAabbB.buffer);
      
      NVVK_CHECK(allocator->createBuffer(m_sceneObjectsB,
                                     MAX_SCENE_OBJECTS*sizeof(shaderio::SceneObject),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                        ));
      NVVK_DBG_NAME(m_sceneObjectsB.buffer);

      NVVK_CHECK(allocator->createBuffer(m_sceneMaterialsB,
                                     MAX_MATERIALS*sizeof(shaderio::Material),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                        ));
      NVVK_DBG_NAME(m_sceneMaterialsB.buffer);

      int dyn_size = 1;
      std::vector<uint8_t> zeros_dynamic(dyn_size, 0);
      NVVK_CHECK(allocator->createBuffer(m_sceneDynamicObjects.nvbuffer,
                                    MAX_SCENE_DYNAMIC_OBJECTS*sizeof(uint),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VMA_MEMORY_USAGE_AUTO,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                                      | VMA_ALLOCATION_CREATE_MAPPED_BIT));
      m_sceneDynamicObjects.mappedData = m_sceneDynamicObjects.nvbuffer.mapping;
      m_sceneDynamicObjects.count = 1;
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_sceneDynamicObjects.nvbuffer, 0,std::span(zeros_dynamic)));  

      // ------------------
      // Accel structure buffers
      // ------------------
      std::vector<nvutils::Bbox> aabbVector;
      aabbVector.push_back(
        nvutils::Bbox(glm::vec3(0.0),glm::vec3(1)));
      NVVK_CHECK(allocator->createBuffer(m_bLasB,
                                     aabbVector.size()*sizeof(nvutils::Bbox),
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                          | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                        ));
      NVVK_DBG_NAME(m_bLasB.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_bLasB, 0,std::span(aabbVector)));  

      const int instanceCount = shaderio::NUM_BRICKS_IN_ATLAS;
      NVVK_CHECK(m_alloc.createBuffer(m_instancesB,
                                    instanceCount*sizeof(VkAccelerationStructureInstanceKHR),
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                        | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                      ));
      NVVK_DBG_NAME(m_instancesB.buffer);

      // ------------------
      // Job queues
      // ------------------
      VkDeviceSize b_size = shaderio::MAX_NUM_BUILD_JOBS*sizeof(shaderio::BuildJob);
      NVVK_CHECK(allocator->createBuffer(m_buildJobQueue,
                                     b_size,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                        ));
      NVVK_DBG_NAME(m_buildJobQueue.buffer);

      b_size = shaderio::MAX_NUM_BRICK_JOBS*sizeof(shaderio::BrickJob);
      std::vector<uint8_t> zeros2(b_size, 0);
      NVVK_CHECK(allocator->createBuffer(m_brickJobQueue,
                                     b_size,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                        ));
      NVVK_DBG_NAME(m_brickJobQueue.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_brickJobQueue, 0,std::span(zeros2)));  

      // ------------------
      // Counters
      // ------------------
      std::vector<glm::uint32_t> zeros3(3, 0);
      NVVK_CHECK(allocator->createBuffer(m_countersB,
                                     zeros3.size()*sizeof(glm::uint32_t),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                        )); // TODO: Is it necesary the transfer bit?
      NVVK_DBG_NAME(m_countersB.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_countersB, 0,std::span(zeros3)));

      // ------------------
      // Indirect dispatch group counts buffer
      // ------------------
      std::vector<shaderio::DispatchIndirectCommand> zerosIndirect(1);
      NVVK_CHECK(allocator->createBuffer(m_indirectB,
                                     zerosIndirect.size()*sizeof(shaderio::DispatchIndirectCommand),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                        ));
      NVVK_DBG_NAME(m_indirectB.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_indirectB, 0,std::span(zerosIndirect)));

      // ------------------
      // Free list buffer
      // ------------------
      std::vector<uint32_t> freeList(shaderio::NUM_BRICKS_IN_ATLAS);
      std::iota(freeList.begin(), freeList.end(), 0);
      NVVK_CHECK(allocator->createBuffer(m_freeListB,
                                     freeList.size()*sizeof(uint32_t),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                        ));
      NVVK_DBG_NAME(m_freeListB.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_freeListB, 0,std::span(freeList)));

      // ------------------
      // Random
      // ------------------
      NVVK_CHECK(m_alloc.createBuffer(m_aoKernelsB,
                                      MAX_NUM_AO_KERNELS*sizeof(glm::vec3),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                      ));
      NVVK_DBG_NAME(m_aoKernelsB.buffer);

      NVVK_CHECK(m_alloc.createBuffer(m_shadowKernelsB,
                                      MAX_NUM_SHADOW_KERNELS*sizeof(glm::vec3),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT 
                                      ));
      NVVK_DBG_NAME(m_shadowKernelsB.buffer);


      m_stagingUploader.cmdUploadAppended(cmd);  // Upload the scene information to the GPU

    m_app->submitAndWaitTempCmdBuffer(cmd); 


    // Camera setup
    m_cameraManip->setClipPlanes({0.01F, 100.0F});
    m_cameraManip->setLookat({0.0F, 0.5F, 5.0}, {0.F, 0.F, 0.F}, {0.0F, 1.0F, 0.0F});
  }

  //---------------------------------------------------------------------------------------------------------------
  // The Vulkan descriptor set defines the resources that are used by the shaders.
  // Here we add the bindings for the textures.
  void createDescriptorSetLayout(){
    SCOPED_TIMER(__FUNCTION__);

    // TODO: OH GOD, clean this mess!!!!!1!1!!
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(shaderio::BindingPoints::sceneInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    
    bindings.addBinding(shaderio::BindingPoints::renderTarget, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::normalBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::albedoBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::depthBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::shadowBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::shadowSampler, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::shadowScratchBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::shadowScratchSampler, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::positionBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::aoBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::aoScratchBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::aoSample, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::aoScratchSample, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    
    bindings.addBinding(shaderio::BindingPoints::gSampler, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    
    bindings.addBinding(shaderio::BindingPoints::aabbs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::objects, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::dynamicObjects, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    
    bindings.addBinding(shaderio::BindingPoints::tLas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::bLas, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    
    bindings.addBinding(shaderio::BindingPoints::clipMap, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::brickAtlas, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::matAtlas, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    
    bindings.addBinding(shaderio::BindingPoints::buildJobQ, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::brickJobQ, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::counters, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::indirectCommands, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::freeList, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

    bindings.addBinding(shaderio::BindingPoints::noise, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::aoKernels, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::shadowKernels, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);


    // Creating the descriptor set and set layout from the bindings
    NVVK_CHECK(m_descPack.init(bindings, m_app->getDevice(), 1));
    NVVK_DBG_NAME(m_descPack.getLayout());
    NVVK_DBG_NAME(m_descPack.getPool());
    NVVK_DBG_NAME(m_descPack.getSet(0));

    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::sceneInfo), m_sceneInfoB.buffer);
    
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::aabbs), m_sceneAabbB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::objects), m_sceneObjectsB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::materials), m_sceneMaterialsB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::dynamicObjects), m_sceneDynamicObjects.nvbuffer.buffer);
    
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::tLas), m_tLas);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::bLas), m_bLasB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::instances), m_instancesB.buffer);
    
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::clipMap), m_clipMap.descriptor);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::brickAtlas), m_brickAtlas.descriptor);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::matAtlas), m_matAtlas.descriptor);
    
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::buildJobQ), m_buildJobQueue.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::brickJobQ), m_brickJobQueue.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::counters), m_countersB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::indirectCommands), m_indirectB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::freeList), m_freeListB.buffer);
    
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::noise), m_noiseTex.descriptor);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::aoKernels), m_aoKernelsB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::shadowKernels), m_shadowKernelsB.buffer);
    
    vkUpdateDescriptorSets(m_app->getDevice(),  
                        static_cast<uint32_t>(writeContainer.size()),  
                        writeContainer.data(), 0, nullptr);
  }

  void createPipelineLayout(VkPipelineLayout* pipelineLayout){
    // Push constant is used to pass data to the shader at each frame
    const VkPushConstantRange pushConstantsRange{
      .stageFlags = VK_SHADER_STAGE_ALL, 
      .offset = 0, 
      .size = sizeof(shaderio::PushConstant)
    };

    // The pipeline layout is used to pass data to the pipeline, anything with "layout" in the shader
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = m_descPack.getLayoutPtr(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantsRange,
    };
    NVVK_CHECK(vkCreatePipelineLayout(m_app->getDevice(), &pipelineLayoutInfo, nullptr, pipelineLayout));
    NVVK_DBG_NAME(*pipelineLayout);
  }

  VkShaderModuleCreateInfo createShaderModule(VkShaderModule* shaderModule, const std::filesystem::path& filename, const std::span<const uint32_t> spirv){
    vkDestroyShaderModule(m_app->getDevice(), *shaderModule, nullptr);

    // Use pre-compiled shaders by default
    VkShaderModuleCreateInfo shaderCode = nvsamples::getShaderModuleCreateInfo(spirv);

    // Get .slang file and compile to spirv
    std::filesystem::path shaderSource = nvutils::findFile(filename, nvsamples::getShaderDirs());
    if(m_slangCompiler.compileFile(shaderSource)){
      // Using the Slang compiler to compile the shaders
      shaderCode.codeSize = m_slangCompiler.getSpirvSize();
      shaderCode.pCode    = m_slangCompiler.getSpirv();
    }else{
      LOGE("Error compiling shader: %s\n%s\n", shaderSource.string().c_str(),
           m_slangCompiler.getLastDiagnosticMessage().c_str());
    }

    // Create shader module using the pcode
    const uint32_t* spirvPtr = reinterpret_cast<const uint32_t*>(m_slangCompiler.getSpirv());
    size_t spirvWordCount = m_slangCompiler.getSpirvSize() / sizeof(uint32_t);
    //LOGI("Shader %s has %zu words\n",filename.c_str(),spirvWordCount);
    NVVK_CHECK(nvvk::createShaderModule(*shaderModule, m_app->getDevice(), 
    std::span<const uint32_t>(spirvPtr, spirvWordCount)));
    NVVK_DBG_NAME(*shaderModule);
    return shaderCode;
  }

  void compileShaders(){
    SCOPED_TIMER(__FUNCTION__);

    createShaderModule(&m_tracingPipeline.shader,"compute_tracing.slang",compute_tracing_slang);
    createShaderModule(&m_lightingPipeline.shader,"lighting.slang",lighting_slang);
    createShaderModule(&m_rtPipeline.shader,"raytracing.slang",raytracing_slang);
    createShaderModule(&m_brickJobPipeline.shader,"brick.slang",brick_slang);
    createShaderModule(&m_buildJobPipeline.shader,"build.slang",build_slang);
    createShaderModule(&m_aoPipeline.shader,"ao.slang",ao_slang);
    createShaderModule(&m_bilateralHPipeline.shader,"bilateral_h.slang",bilateral_h_slang);
    createShaderModule(&m_bilateralVPipeline.shader,"bilateral_v.slang",bilateral_v_slang);
    createShaderModule(&m_simulationPipeline.shader,"simulation.slang",simulation_slang);
  }

  void createPipelines(){
    SCOPED_TIMER(__FUNCTION__);

    createComputePipeline(&m_tracingPipeline);
    createComputePipeline(&m_lightingPipeline);
    createRTPipeline(&m_rtPipeline);
    createComputePipeline(&m_brickJobPipeline);
    createComputePipeline(&m_buildJobPipeline);
    createComputePipeline(&m_aoPipeline);
    createComputePipeline(&m_bilateralHPipeline);
    createComputePipeline(&m_bilateralVPipeline);
    createComputePipeline(&m_simulationPipeline);
  }

  void createComputePipeline(Pipeline* pl){
    destroyPipeline(pl);
    createPipelineLayout(&pl->layout);

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = pl->shader;
    stage.pName  = "computeMain";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = pl->layout;

    NVVK_CHECK(vkCreateComputePipelines(
        m_app->getDevice(),
        VK_NULL_HANDLE,
        1,
        &cpci,
        nullptr,
        &pl->pipeline));
  }

  void createRTPipeline(Pipeline* pl){
    destroyPipeline(pl);

    createPipelineLayout(&pl->layout);

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eMissShadow,
      eClosestHit,
      eClosestHitShadow,
      eIntersection,
      eIntersectionShadow,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    for(auto& s : stages){
      s = {};
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    }

    stages[eRaygen].module              = pl->shader;
    stages[eRaygen].pName               = "rgenMain";
    stages[eRaygen].stage               = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].module                = pl->shader;
    stages[eMiss].pName                 = "rmissMain";
    stages[eMiss].stage                 = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMissShadow].module          = pl->shader;
    stages[eMissShadow].pName           = "rmissShadow";
    stages[eMissShadow].stage           = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].module          = pl->shader;
    stages[eClosestHit].pName           = "rchitMain";
    stages[eClosestHit].stage           = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHitShadow].module    = pl->shader;
    stages[eClosestHitShadow].pName     = "rchitMain";
    stages[eClosestHitShadow].stage     = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eIntersection].module        = pl->shader;
    stages[eIntersection].pName         = "rintMain";
    stages[eIntersection].stage         = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    stages[eIntersectionShadow].module  = pl->shader;
    stages[eIntersectionShadow].pName   = "rintShadow";
    stages[eIntersectionShadow].stage   = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    
    
    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    // Raygen
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shader_groups.push_back(group);

    // Miss
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shader_groups.push_back(group);

    // Shadow miss
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMissShadow;
    shader_groups.push_back(group);

    // Closest hit shader
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.intersectionShader = eIntersection;
    shader_groups.push_back(group);

    // Shadow intersection
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    group.closestHitShader = eClosestHitShadow;
    group.intersectionShader = eIntersectionShadow;
    shader_groups.push_back(group);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rtPipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    rtPipelineInfo.pStages                      = stages.data();
    rtPipelineInfo.groupCount                   = static_cast<uint32_t>(shader_groups.size());
    rtPipelineInfo.pGroups                      = shader_groups.data();
    rtPipelineInfo.maxPipelineRayRecursionDepth = std::max(3U, m_rtProperties.maxRayRecursionDepth);  // Ray depth
    rtPipelineInfo.layout                       = pl->layout;
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR (m_app->getDevice(), {}, {}, 1, &rtPipelineInfo, nullptr, &pl->pipeline));

    // Create the shader binding table for this pipeline
    createShaderBindingTable(pl, rtPipelineInfo);
  }

  void createShaderBindingTable(Pipeline* pl, const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo){
    SCOPED_TIMER(__FUNCTION__);

    m_alloc.destroyBuffer(m_sbtB);
    // Calculate required SBT buffer size
    size_t bufferSize = m_sbtGen.calculateSBTBufferSize(pl->pipeline, rtPipelineInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(m_alloc.createBuffer(m_sbtB, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                        m_sbtGen.getBufferAlignment()));

    NVVK_DBG_NAME(m_sbtB.buffer);

    // Populate the SBT buffer with shader handles and data using the CPU-mapped memory pointer
    NVVK_CHECK(m_sbtGen.populateSBTBuffer(m_sbtB.address, bufferSize, m_sbtB.mapping));
  }


  // Recompiles and waits for idle time to swap it into the pipeline
  void reloadShaders(){
    SCOPED_TIMER(__FUNCTION__);
    compileShaders();
    vkDeviceWaitIdle(m_app->getDevice());
    createPipelines();
  }


  void updateSceneBuffer(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    const glm::mat4& viewMatrix = m_cameraManip->getViewMatrix();
    const glm::mat4& projMatrix = m_cameraManip->getPerspectiveMatrix();
    m_currCamId0 = glm::floor(m_cameraManip->getEye()/shaderio::BRICK_SIZES[0]);
    const glm::vec3& id0Pos = glm::vec3(m_currCamId0)*shaderio::BRICK_SIZES[0];

    m_sceneInfo.viewMatrix = viewMatrix;
    m_sceneInfo.projMatrix = projMatrix;
    m_sceneInfo.viewMatrixInv = glm::inverse(viewMatrix);
    m_sceneInfo.projMatrixInv = glm::inverse(projMatrix);
    m_sceneInfo.cameraPosition = glm::vec4(m_cameraManip->getEye(),0.0);
    m_sceneInfo.cameraId0 = glm::ivec4(m_currCamId0,0);
    m_sceneInfo.cameraId0Pos = glm::vec4(id0Pos,0);

    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneInfoB.buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT});
    vkCmdUpdateBuffer(cmd, m_sceneInfoB.buffer, 0, sizeof(shaderio::SceneInfo), &m_sceneInfo);
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneInfoB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
  }

  void readAndUpdateDynamicObjects(VkCommandBuffer cmd){

    if(!m_firstFrame){
      vkWaitForFences(m_app->getDevice(), 1, &m_simFence, VK_TRUE, UINT64_MAX);
      vkResetFences(m_app->getDevice(), 1, &m_simFence);

      assert(rwbuff.mappedData != nullptr);
/* 
      nvvk::cmdBufferMemoryBarrier(cmd, {
        m_sceneDynamicObjects.nvbuffer.buffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_2_HOST_BIT
      });
 */
      // Read buffer
      uint* rdata = reinterpret_cast<uint*>(m_sceneDynamicObjects.mappedData);

      for(uint i = 0; i < m_sceneDynamicObjects.count; i++){
        m_test = rdata[i];
        LOGI("CPU Read %u @ %i\n",m_test,m_pushConst.frameCount);
      }
      // Process data
/* 
      nvvk::cmdBufferMemoryBarrier(cmd, {
        m_sceneDynamicObjects.nvbuffer.buffer,
        VK_PIPELINE_STAGE_2_HOST_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
      });
       */
    }
    m_sceneDynamicObjects.count = 1;
    // Write buffer
    /* 
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneDynamicObjects.nvbuffer.buffer,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT});
 */
    //vkCmdUpdateBuffer(cmd, m_sceneDynamicObjects.nvbuffer.buffer, 0, sizeof(uint), &m_test);
    /* 
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneDynamicObjects.nvbuffer.buffer,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
                                   */
    LOGI("CPU Write %u @ %i\n",m_test,m_pushConst.frameCount);

  }

  // Accessor for camera manipulator
  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() const { return m_cameraManip; }

private:
  // Vulkan and app variables
  nvapp::Application*     m_app{};              // The application framework
  nvvk::ResourceAllocator m_alloc{};            // Resource allocator for Vulkan resources, used for buffers and images
  nvvk::StagingUploader   m_stagingUploader{};  // Utility to upload data to the GPU, used for staging buffers and images
  nvvk::SamplerPool       m_samplerPool{};      // Texture sampler pool, used to acquire texture samplers for images
  VkSampler               m_gBuffersSampler{};  // Sampler used for the gBuffers
  nvvk::GBuffer           m_gBuffers{};         // The G-Buffers
  nvslang::SlangCompiler  m_slangCompiler{};    // The Slang compiler used to compile the shaders
  nvvk::DescriptorPack    m_descPack{};         // The descriptor bindings used to create the descriptor set layout and descriptor sets
  VkCommandPool           m_simCmdPool{};       // Command pool for simulation
  VkFence                 m_simFence{};         // Fence for the end of simulation pass

  // Pipelines
  Pipeline m_tracingPipeline{};     // Tracing pipeline, fills the gbuffers with info
  Pipeline m_lightingPipeline{};    // Lighting pipeline, uses the gbuffers to paint th viewport
  Pipeline m_rtPipeline{};          // Hardware accelerated ray tracing pipeline
  Pipeline m_buildJobPipeline{};    // Build job pipeline
  Pipeline m_brickJobPipeline{};    // Brick job pipeline
  Pipeline m_aoPipeline{};          // Ambient occlussion generation pipeline
  Pipeline m_bilateralHPipeline{};  // Bilateral blur horizontal pass
  Pipeline m_bilateralVPipeline{};  // Bilateral blur vertical pass
  Pipeline m_simulationPipeline{};  // Simluation pass

  // Shader binding table management
  nvvk::SBTGenerator    m_sbtGen;             // SBT manager
  nvvk::Buffer          m_sbtB;               // Buffer for shader binding table

  // Ray Tracing Properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  // Push constants to send 
  shaderio::PushConstant m_pushConst = {.time = 0.0f};

  // Scene information
  shaderio::SceneInfo   m_sceneInfo{};            // Struct containing the scene information
  nvvk::Buffer          m_sceneInfoB{};           // Buffer binded to the UBO of scene info
  nvvk::Buffer          m_sceneAabbB{};           // Buffer binded to the scene aabbs array
  nvvk::Buffer          m_sceneObjectsB{};        // Buffer binded to the scene objects array
  nvvk::Buffer          m_sceneMaterialsB{};      // Buffer binded to the scene materials array
  RWBuffer              m_sceneDynamicObjects;   // Buffers binded to the scene dynamic objects array

  // Acceleration structure buffers and components
  nvvk::AccelerationStructure   m_tLas{};       // Top-level acceleration structure
  nvvk::AccelerationStructure   m_bLas{};       // Bottom-level acceleration structure
  nvvk::Buffer                  m_tLasB{};      // Top-level acceleration structures scratch buffer
  nvvk::Buffer                  m_bLasB{};      // Bottom-level acceleration structures buffer
  nvvk::Buffer                  m_instancesB{}; // Instances buffer
  
  // Job queues and utils
  nvvk::Buffer m_buildJobQueue{};   // Queue for the Build jobs
  nvvk::Buffer m_brickJobQueue{};   // Queue for the Brick jobs
  nvvk::Buffer m_countersB{};       // Diferent counters used by the shaders
  nvvk::Buffer m_indirectB{};       // Indirect dispatch group counts buffer
  nvvk::Buffer m_freeListB{};       // List of free pointers to the brick atlas

  // RNG
  nvvk::Image  m_noiseTex{};        // Rgb noise texture
  nvvk::Buffer m_aoKernelsB{};  // Buffer containing random vectors on +Y hemisphere
  nvvk::Buffer m_shadowKernelsB{};  // Buffer containing random unit vectors on +Y hemisphere

  // 3D textures
  nvvk::Image m_clipMap{};          // 3D map of pointers to the brick atlas
  nvvk::Image m_brickAtlas{};       // Atlas where all the bricks are stored
  nvvk::Image m_matAtlas{};         // Atlas where the materials indeces of the allocated bricks are stored

  // Pre-built components
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip{std::make_shared<nvutils::CameraManipulator>()}; // Camera manipulator
  nvshaders::Tonemapper    m_tonemapper{};      // Tonemapper for post-processing effects
  shaderio::TonemapperData m_tonemapperData{};  // Tonemapper data used to pass parameters to the tonemapper shader

  // Scene
  Scene m_scene;
  glm::ivec3 m_currCamId0 = glm::ivec3(0);
  glm::ivec3 m_prevCamId0 = glm::ivec3(0);
  float m_prevTime = -1;

  // UI params
  bool m_debugActive = false;
  int m_debugMode = 0;
  bool m_rtxON = false;
  bool m_rebuildTlas = false;
  bool m_refreshAOkernels = false;
  bool m_refreshShadowKernels = false;
  bool m_updateTlas = false;
  bool m_firstFrame = true;
  glm::vec3 m_zenithColor = glm::vec3(0.644, 0.635, 0.608);
  glm::vec3 m_horizonColor = glm::vec3(0.628, 0.495, 0.279);

  // Test TODO: Remove
  uint m_test = 0;

  // Startup managers for profiler and paramter registry
  Info m_info;

  // Profiler
  nvutils::ProfilerTimeline* m_graphicsTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;
};


int main(int argc, char** argv)
{
  initRandom();

  nvutils::ProfilerManager   profilerManager;
  nvutils::ParameterRegistry parameterRegistry;
  nvutils::ParameterParser   parameterParser;

  // setup sample element
  AppElement::Info sampleInfo = {
      .profilerManager   = &profilerManager,
      .parameterRegistry = &parameterRegistry,
  };
  std::shared_ptr<AppElement> appElement = std::make_shared<AppElement>(sampleInfo);

  // setup logger element, `true` means shown by default
  // we add it early so outputs are captured early on, you might want to defer this to a later timer.
  std::shared_ptr<nvapp::ElementLogger> elementLogger = std::make_shared<nvapp::ElementLogger>(true);
  nvutils::Logger::getInstance().setLogCallback([&](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    elementLogger->addLog(logLevel, "%s", text.c_str());
  });

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};

  nvvk::ValidationSettings validationSettings;  
  validationSettings.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);  

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {
        {VK_KHR_SWAPCHAIN_EXTENSION_NAME},
        {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},                                    // Required for premade modules, like the tonemapper
        {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},     // Build acceleration structures
        {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},  // Use vkCmdTraceRaysKHR
        {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},                           // Required by ray tracing pipeline
        {VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME}
      },
      .instanceCreateInfoExt = validationSettings.buildPNextChain(),
      .enableValidationLayers = true
  };

  // let's add a command-line option to enable/disable validation layers
  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"verbose"}, &vkSetup.verbose);
  // as well as an option to force the vulkan device based on canonical index
  parameterRegistry.add({"forcedevice"}, &vkSetup.forceGPU);

  // add all parameters to the parser
  parameterParser.add(parameterRegistry);

  // and then parse command line
  parameterParser.parse(argc, argv);

  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name           = "Victor's TFG"; //TODO: Change to a *cooler* name
  appInfo.useMenu        = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();
  appInfo.dockSetup      = [](ImGuiID viewportID) {
    ImGuiID centerNode = viewportID;

    ImGuiID settingID = ImGui::DockBuilderSplitNode(centerNode, ImGuiDir_Right, 0.12f, nullptr, &centerNode);
    ImGuiID sceneID   = ImGui::DockBuilderSplitNode(centerNode, ImGuiDir_Left,  0.2f, nullptr, &centerNode);
    ImGuiID objectID   = ImGui::DockBuilderSplitNode(sceneID, ImGuiDir_Down,  0.5f, nullptr, &sceneID);
    ImGuiID materialID   = ImGui::DockBuilderSplitNode(sceneID, ImGuiDir_Down,  0.2f, nullptr, &sceneID);
    ImGuiID loggerID  = ImGui::DockBuilderSplitNode(centerNode, ImGuiDir_Down,  0.3f, nullptr, &centerNode);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.5f, nullptr, &loggerID);

    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGui::DockBuilderDockWindow("Scene", sceneID);
    ImGui::DockBuilderDockWindow("Object", objectID);
    ImGui::DockBuilderDockWindow("Material", materialID);
    ImGui::DockBuilderDockWindow("Log", loggerID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // add camera element
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  auto camManip = appElement->getCameraManipulator();
  elemCamera->setCameraManipulator(camManip);
  app.addElement(elemCamera);
  // add the app element
  app.addElement(appElement);
  // add the window element
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());
  // add profiler element
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager));
  // add logger element
  app.addElement(elementLogger);
  
  // Initial camera params
  // Set camera to start at position (0,0,0) looking along -Z axis with Y up  
  nvutils::CameraManipulator::Camera camera;  
  camera.eye = glm::vec3(0.0f, 0.0f, 0.0f);  // Camera position  
  camera.ctr = glm::vec3(0.0f, 0.0f, -1.0f); // Look at point (forward)  
  camera.up  = glm::vec3(0.0f, 1.0f, 0.0f);  // Up vector  
  camera.fov = 60.0f;                         // Field of view in degrees  
    
  // Set this as the home camera  
  nvgui::SetHomeCamera(camera);  
  camManip->setCamera(camera, true);  // Apply immediately
  camManip->setMode(nvutils::CameraManipulator::Modes::Fly);

  // enter the main loop
  app.run();

  // Cleanup in reverse order
  app.deinit();
  vkContext.deinit();

  return 0;
}
