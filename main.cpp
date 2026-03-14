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
#include "glm/vector_relational.hpp"
#include "nvvk/resources.hpp"
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

#include "_autogen/tracing.slang.h"
#include "_autogen/lighting.slang.h"
#include "_autogen/raytracing.slang.h"
#include "_autogen/brick.slang.h"
#include "_autogen/build.slang.h"
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
#include "slang.h"

const char* DebugModes[] = {
    "Debug color",
    "Albedo",
    "Normal",
    "Depth",
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

class AppElement : public nvapp::IAppElement
{
  enum
  {
    eImgNormal,
    eImgAlbedo,
    eImgRendered,
    eImgTonemapped,
  };

public:
  struct Info
  {
    nvutils::ProfilerManager*   profilerManager{};
    nvutils::ParameterRegistry* parameterRegistry{};
  };


  AppElement(const Info& info)
      : m_info(info)
  {
    // Add run parameter example
    //m_info.parameterRegistry->add({"animate"}, &m_animate);
  }

  ~AppElement() override = default;

  void onAttach(nvapp::Application* app) override{
    m_app                                = app;

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
    m_asBuilder.init(&m_alloc, &m_stagingUploader, m_app->getQueue(0)); // TODO: Is this the correct queue?
    m_sbtGen.init(app->getDevice(),m_rtProperties);



    // TODO set back to on when proper lighting solution is made
    // Set tonemapping off by default
    m_tonemapperData.isActive = 0;

    setupSlangCompiler();           // Setup slang compiler with correct build config flags
    createScene();                  // Create the scene and fill it up with sdfs
    setupGBuffers();                // Set up the GBuffers to render to
    create3DTextures();             // Creates the different 3d textures used to store voxel grid data
    createAccelerationStructures(); // Creates the bLas and tLas needed for the rt pipeline 
    createDescriptorSetLayout();    // Create the descriptor set layout for the pipelines
    createPipelineLayouts();        // Create the pipelines layouts
    compileAndCreateShaders();      // Compile the shaders and create the shader modules
    createPipelines();              // Create the pipelines using the layouts and the shaders

    // Initialize the tonemapper with proe-compiled shader
    m_tonemapper.init(&m_alloc, std::span<const uint32_t>(tonemapper_slang));

    // TODO: Figure out how to use the profiler tool
    // Init profiler with a single queue
    m_profilerTimeline = m_info.profilerManager->createTimeline({"graphics"});
    m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  }

  //-------------------------------------------------------------------------------
  // Destroy all elements that were created
  // - Called when the application is shutting down
  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

    VkDevice device = m_app->getDevice();

    m_descPack.deinit();
    vkDestroyPipeline(device,m_tracingPipeline,nullptr);
    vkDestroyPipeline(device,m_lightingPipeline,nullptr);
    vkDestroyPipeline(device,m_rtPipeline,nullptr);
    vkDestroyPipeline(device,m_brickJobPipeline,nullptr);
    vkDestroyPipeline(device,m_buildJobPipeline,nullptr);
    vkDestroyPipelineLayout(device,m_tracingLayout,nullptr);
    vkDestroyPipelineLayout(device,m_lightingLayout,nullptr);
    vkDestroyPipelineLayout(device,m_rtPipelineLayout,nullptr);
    vkDestroyPipelineLayout(device,m_brickJobLayout,nullptr);
    vkDestroyPipelineLayout(device,m_buildJobLayout,nullptr);
    vkDestroyShaderModule(device,m_tracingModule,nullptr);
    vkDestroyShaderModule(device,m_lightingModule,nullptr);
    vkDestroyShaderModule(device,m_rtModule,nullptr);
    vkDestroyShaderModule(device,m_brickJobModule,nullptr);
    vkDestroyShaderModule(device,m_buildJobModule,nullptr);

    m_alloc.destroyBuffer(m_sceneInfoB);
    m_alloc.destroyBuffer(m_sceneAabbB);
    m_alloc.destroyBuffer(m_sceneObjectsB);
    m_alloc.destroyBuffer(m_buildJobQueue);
    m_alloc.destroyBuffer(m_brickJobQueue);
    m_alloc.destroyBuffer(m_freeListCounter);
    m_alloc.destroyImage(m_globalGrid);
    m_alloc.destroyImage(m_clipMap);
    m_alloc.destroyImage(m_brickAtlas);
    m_alloc.destroyBuffer(m_sbtBuffer);
    m_asBuilder.deinitAccelerationStructures();

    m_gBuffers.deinit();
    m_sbtGen.deinit();
    m_stagingUploader.deinit();
    m_asBuilder.deinit();
    //m_skySimple.deinit();
    m_tonemapper.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
    m_profilerGpuTimer.deinit();
    m_info.profilerManager->destroyTimeline(m_profilerTimeline);
  }

  //---------------------------------------------------------------------------------------------------------------
  // Rendering all UI elements, this includes the image of the GBuffer, the camera controls, and the sky parameters.
  // - Called every frame
  void onUIRender() override
  { 
    ImGui::Begin("Settings");
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);

    // Add window information
    const VkExtent2D& viewportSize = m_app->getViewportSize();
    ImGui::Text("Viewport Size: %d x %d", viewportSize.width, viewportSize.height);

    if(ImGui::CollapsingHeader("Camera"))
        nvgui::CameraWidget(m_cameraManip);

    if(ImGui::CollapsingHeader("Tonemapper"))
        nvgui::tonemapperWidget(m_tonemapperData);

    if(!ImGui::CollapsingHeader("Tracing")){
      ImGui::Checkbox("Hardware RTX", &m_RTX_ON);
    }

    if(ImGui::CollapsingHeader("Lighting data")){
      ImGui::Text("Directional Light");
      ImGui::SliderFloat3("Direction", &m_pushConst.lp.lightDir.x, -1.0f, 1.0f);
      ImGui::ColorEdit3("Light Color", &m_pushConst.lp.lightColor.x);

      ImGui::Separator();
      ImGui::Text("Ambient Hemispheric");
      ImGui::ColorEdit3("Ambient Top", &m_pushConst.lp.ambientTop.x);
      ImGui::ColorEdit3("Ambient Bottom", &m_pushConst.lp.ambientBottom.x);

      ImGui::Separator();
      ImGui::Text("Fog");
      ImGui::SliderFloat("Fog Density", &m_pushConst.lp.fogDensity, 0.0f, 0.2f);
      ImGui::ColorEdit3("Fog Color", &m_pushConst.lp.fogColor.x);
    }
    
    if(!ImGui::CollapsingHeader("Debug colors")){
      ImGui::Checkbox("Active", &m_debugActive);
      ImGui::Combo("Mode", &m_debugMode, DebugModes, IM_ARRAYSIZE(DebugModes));
      ImGui::Combo("Palette", &m_pushConst.debug.palette, DebugPalettes, IM_ARRAYSIZE(DebugPalettes));
      if(m_debugActive){
        m_pushConst.debug.mode = m_debugMode + 1;
      }else{
        m_pushConst.debug.mode = 0;
      }
      // TODO: temporary
      if(ImGui::Button("Refresh grid")){
        m_scene.m_needsRefresh = true;
      }

      //auto aabbs = m_scene.getBboxes();
      //auto jobs = m_scene.getBuildJobs(aabbs);
      if(ImGui::Button("Test")){
        auto aabbs = m_scene.getBboxes();
        int x;
        auto jobs = m_scene.getBuildJobs(aabbs,x);
        m_testSize = jobs.size();
        m_testMed = glm::ivec3(0);
        for(auto& job: jobs){
          auto num_b = glm::ivec3(job.num_b_level);
          if(glm::any(glm::greaterThan(num_b, glm::ivec3(MAX_BUILD_JOB_SIZE)))){
            LOGI("OH NO! %i,%i,%i\n",num_b.x,num_b.y,num_b.z);
          }
          m_testMed += num_b;
        }
        m_testMed /= m_testSize;
      }
      ImGui::Text("Test size: %i",m_testSize);
      ImGui::Text("Test Median: %f,%f,%f",m_testMed.x,m_testMed.y,m_testMed.z);
    }

    ImGui::End();

    // Draw scene tree and object tab
    m_scene.draw();

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(eImgTonemapped), ImGui::GetContentRegionAvail());
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

  void onPreRender() override { m_profilerTimeline->frameAdvance(); }

  //---------------------------------------------------------------------------------------------------------------
  // When the viewport is resized, the GBuffer must be resized
  // - Called when the Window "viewport is resized
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { 
    NVVK_CHECK(m_gBuffers.update(cmd, size)); 

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
    
    // Update data TODO: Encapsulate
    m_pushConst.time = static_cast<float>(ImGui::GetTime());
    m_pushConst.lp.lightDir = glm::normalize(m_pushConst.lp.lightDir);
    updateSceneBuffer(cmd);

    if(m_scene.m_needsRefresh){
      updateSceneObjects(cmd);
      updateTextureData(cmd);
      m_scene.m_needsRefresh = false;
    }

    if(m_RTX_ON){
      raytracingPass(cmd);
    }else{
      tracingPass(cmd);
    }

    lightingPass(cmd);
    
    postProcess(cmd);
  }

  void tracingPass(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracingPipeline);  
    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracingLayout,
                            0, 1, m_descPack.getSetPtr(), 0, nullptr);  
    // Push constants
    vkCmdPushConstants(cmd, m_tracingLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
    // Dispatch
    VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE_2D);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
  }

  void lightingPass(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    // Update data TODO: Encapsulate
    shaderio::PushConstant pc{};
    pc.time = static_cast<float>(ImGui::GetTime());
    updateSceneBuffer(cmd);

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_lightingPipeline);  
    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_lightingLayout,
                            0, 1, m_descPack.getSetPtr(), 0, nullptr);  
    // Push constants
    vkCmdPushConstants(cmd, m_lightingLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
    // Dispatch
    VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE_2D);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
  }

  void raytracingPass(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);  
    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout,
                            0, 1, m_descPack.getSetPtr(), 0, nullptr);  
    // Push constants
    vkCmdPushConstants(cmd, m_rtPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
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
  }

  void executeBrickJobs(VkCommandBuffer cmd, int num_brick_jobs){
    NVVK_DBG_SCOPE(cmd);

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_brickJobPipeline);  

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_brickJobLayout,
                            0, 1, m_descPack.getSetPtr(), 0, nullptr);  
    // Push constants
    vkCmdPushConstants(cmd, m_brickJobLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
    
    // Dispatch
    vkCmdDispatch(cmd, 
      BRICK_JOB_GROUP_X_DISPATCH_SIZE, 
      glm::ceil(float(num_brick_jobs)/float(BRICK_JOB_GROUP_X_DISPATCH_SIZE)),
      1);
  
    nvvk::cmdImageMemoryBarrier(cmd, {m_brickAtlas.image, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_GENERAL});
  }

  void generationPass(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    int num_bricks;
    std::vector<nvutils::Bbox> aabbVector = m_scene.getBboxes();
    std::vector<shaderio::BuildJob> buildJobs = m_scene.getBuildJobs(aabbVector,num_bricks);

    if(buildJobs.size() > MAX_NUM_BUILD_JOBS)
      LOGE("Not enough space in build job queue to allocale %i jobs\n",num_bricks);
    if(num_bricks > shaderio::MAX_NUM_BRICK_JOBS)
      LOGE("Not enough space in brick job queue to allocale %i jobs\n",num_bricks);

    m_pushConst.numBrickJobs = num_bricks;

    unsigned long size = buildJobs.size() * sizeof(shaderio::BuildJob);
    vkCmdUpdateBuffer(cmd, m_buildJobQueue.buffer, 0, size, buildJobs.data());

    nvvk::cmdBufferMemoryBarrier(cmd, {m_buildJobQueue.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_buildJobPipeline);  

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_buildJobLayout,
                            0, 1, m_descPack.getSetPtr(), 0, nullptr);  
    // Push constants
    vkCmdPushConstants(cmd, m_buildJobLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
    
    // Dispatch
    vkCmdDispatch(cmd, 1, 1, buildJobs.size());
  
    nvvk::cmdBufferMemoryBarrier(cmd, {m_brickJobQueue.buffer, 
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_SHADER_READ_BIT}); 

    executeBrickJobs(cmd, num_bricks);
  }

  // Apply post-processing
  void postProcess(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    // Wait for render target to be done
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(eImgRendered), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_GENERAL});

    // No img layout transition needed

    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(eImgRendered),
                            m_gBuffers.getDescriptorImageInfo(eImgTonemapped));
  }

  void setupSlangCompiler(){
    m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_slangCompiler.defaultTarget();
    m_slangCompiler.defaultOptions();
#ifdef NDEBUG
    LOGI("Slang compiler: RELEASE configuration\n");
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
    LOGI("Slang compiler: DEBUG configuration\n");
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
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // Create the G-Buffers
    nvvk::GBufferInitInfo gBufferInit{
        .allocator      = &m_alloc,
        .colorFormats   = {
          VK_FORMAT_A2B10G10R10_UNORM_PACK32, // Normal buffer, alpha = Material flag
          VK_FORMAT_R8G8B8A8_UNORM,           // Albedo buffer
          VK_FORMAT_R32G32B32A32_SFLOAT,      // Render target
          VK_FORMAT_R8G8B8A8_UNORM},          // Tonemapped
        .depthFormat    = nvvk::findDepthFormat(m_app->getPhysicalDevice()),
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    };
    m_gBuffers.init(gBufferInit);
  }

  void create3DStorageTexture(nvvk::Image& image, VkExtent3D extent, VkFormat format, VkClearColorValue clearColor){
    // Destroy if already created
    m_alloc.destroyImage(image);


    std::array<uint32_t, 1> queueFamilies = {
        m_app->getQueue(0).familyIndex,
    };

    VkImageCreateInfo ci = DEFAULT_VkImageCreateInfo;
    ci.imageType = VK_IMAGE_TYPE_3D;
    ci.format = format; // TODO: Optimize to only use 8bit unorm
    ci.extent = extent;
    ci.mipLevels = 1; // TODO: Learn how to use MIP levels
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

    // Global grid 
    VkExtent3D extent = {shaderio::NUM_VALUES_PER_AXIS,shaderio::NUM_VALUES_PER_AXIS,shaderio::NUM_VALUES_PER_AXIS};  // XYZ size
    VkFormat format = VK_FORMAT_R16_SNORM;  // Texel format
    glm::float32 clearValueF = 1.0f;
    VkClearColorValue clearColor = {.float32={clearValueF,clearValueF,clearValueF,clearValueF}};
    create3DStorageTexture(m_globalGrid, extent, format, clearColor);

    // Clipmap
    extent = {NUM_BRICKS_PER_AXIS,NUM_BRICKS_PER_AXIS,NUM_BRICKS_PER_AXIS*CLIPMAP_LEVELS};  // XYZ size
    format = VK_FORMAT_R32_UINT;  // Texel format
    uint32_t clearValueClip = shaderio::UNIFORM_POSITIVE_BRICK_POINTER;
    clearColor = {.uint32={clearValueClip,clearValueClip,clearValueClip,clearValueClip}};
    create3DStorageTexture(m_clipMap, extent, format, clearColor);

    // Brick atlas
    const int brick_size = BRICK_SIZE;
    const int atlas_axis_size = BRICK_PER_ATLAS_AXIS*BRICK_SIZE;
    extent = {atlas_axis_size,atlas_axis_size,brick_size};  // XYZ size
    format = VK_FORMAT_R16_SNORM;  // Texel format
    clearValueF = 1.0f;
    clearColor = {.float32={clearValueF,clearValueF,clearValueF,clearValueF}};
    create3DStorageTexture(m_brickAtlas, extent, format, clearColor);
    
  }

  // Warning: Deprecated
  void updateTextureDataCPU(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    assert(m_globalGrid.image);
    std::vector<float> imageData = m_scene.generateDenseGrid();
    assert(m_stagingUploader.isAppendedEmpty());
    nvvk::SemaphoreState cmdSemaphoreState{};
    NVVK_CHECK(m_stagingUploader.appendImage(m_globalGrid, std::span(imageData), m_globalGrid.descriptor.imageLayout, cmdSemaphoreState));
    m_stagingUploader.cmdUploadAppended(cmd);
  }

  void updateTextureData(VkCommandBuffer cmd){
    //updateTextureDataCPU(cmd);
    generationPass(cmd);
  }

  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const uint32_t aabbCount){
    nvvk::AccelerationStructureGeometryInfo result = {};

    // Describe buffer as array of VkAabbPostions
    VkAccelerationStructureGeometryAabbsDataKHR aabbs{
      .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
      .data   = {.deviceAddress = m_sceneAabbB.address},
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

  void createBottomLevelAS(const std::vector<nvvk::AccelerationStructureGeometryInfo> geoInfos){
    m_asBuilder.blasSubmitBuildAndWait(geoInfos, 
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    // TODO: Check out more flags
  }

  void createTopLevelAS(const int instanceCount){
    // TODO: This is hardwired to one global instance
    const glm::mat4 transformM = glm::mat4(1.0f);

    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    for(int i = 0; i<instanceCount; i++){
      VkAccelerationStructureInstanceKHR ray_inst{};
      ray_inst.transform = nvvk::toTransformMatrixKHR(transformM);
      ray_inst.accelerationStructureReference = m_asBuilder.blasSet[i].address;
      ray_inst.instanceCustomIndex = i;
      ray_inst.mask = 0xFF;
      ray_inst.instanceShaderBindingTableRecordOffset = 0;
      ray_inst.flags = 0;
      // TODO: You can put flags here
      tlasInstances.push_back(ray_inst);
    }

    m_asBuilder.tlasSubmitBuildAndWait(tlasInstances, 
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
    // TODO: Check out more flags
  }

  void createAccelerationStructures(){
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    updateSceneObjects(cmd);  // Call update scene after the initial reation of buffers
    m_app->submitAndWaitTempCmdBuffer(cmd); 

    std::vector<nvutils::Bbox> aabbVector = m_scene.getBboxes();

    std::vector<nvvk::AccelerationStructureGeometryInfo> geoInfos;
    VkDeviceSize stride = sizeof(VkAabbPositionsKHR);
    VkDeviceSize currentOffset = 0;
    for(int i = 0; i < aabbVector.size(); i++){
      nvvk::AccelerationStructureGeometryInfo geoInfo = {};

      // Describe buffer as array of VkAabbPostions
      VkAccelerationStructureGeometryAabbsDataKHR aabbs{
        .sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
        .data   = {.deviceAddress = m_sceneAabbB.address + currentOffset},
        .stride = stride
      };

      // Identify the above data as containing opaque triangles.
      geoInfo.geometry = VkAccelerationStructureGeometryKHR{
          .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
          .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
          .geometry     = {.aabbs = aabbs},
          .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
      };

      geoInfo.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{
        .primitiveCount = 1,
        .firstVertex = 0
      };

      geoInfos.push_back(geoInfo);
      currentOffset += stride;
    }
    createBottomLevelAS(geoInfos);
    createTopLevelAS(aabbVector.size());
  }

  void updateSceneObjects(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    std::vector<nvutils::Bbox> aabbVector = m_scene.getBboxes();
    std::vector<shaderio::SceneObject> objectsVector = m_scene.getObjects();
    if(aabbVector.size() != objectsVector.size())
        LOGE("Aabb vector is diferent size from objects vector %zu != %zu\n",aabbVector.size(),objectsVector.size());
    if(aabbVector.size()>MAX_SCENE_OBJECTS)
      LOGE("Number of scene objects exceeds maximum %zu > %i\n",aabbVector.size(),MAX_SCENE_OBJECTS);

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
    
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneAabbB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneObjectsB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
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
                                     VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
      NVVK_DBG_NAME(m_sceneInfoB.buffer);

      // ------------------
      // AABB and objects buffers
      // ------------------
      NVVK_CHECK(allocator->createBuffer(m_sceneAabbB,
                                     MAX_SCENE_OBJECTS*sizeof(nvutils::Bbox),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT 
                                          | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                        ));
      NVVK_DBG_NAME(m_sceneAabbB.buffer);
      
      NVVK_CHECK(allocator->createBuffer(m_sceneObjectsB,
                                     MAX_SCENE_OBJECTS*sizeof(shaderio::SceneObject),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT 
                                        ));
      NVVK_DBG_NAME(m_sceneObjectsB.buffer);

      // ------------------
      // Job queues
      // ------------------
      VkDeviceSize b_size = MAX_NUM_BUILD_JOBS*sizeof(shaderio::BuildJob);
      NVVK_CHECK(allocator->createBuffer(m_buildJobQueue,
                                     b_size,
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT 
                                        ));
      NVVK_DBG_NAME(m_buildJobQueue.buffer);

      b_size = shaderio::MAX_NUM_BRICK_JOBS*sizeof(shaderio::BrickJob);
      std::vector<uint8_t> zeros2(b_size, 0);
      NVVK_CHECK(allocator->createBuffer(m_brickJobQueue,
                                     b_size,
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT 
                                        )); // TODO: Is it necesary the transfer bit?
      NVVK_DBG_NAME(m_brickJobQueue.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_brickJobQueue, 0,std::span(zeros2)));  

      // ------------------
      // Free list
      // ------------------
      std::vector<glm::uint32_t> zeros3(1, 0);
      NVVK_CHECK(allocator->createBuffer(m_freeListCounter,
                                     sizeof(glm::uint32_t),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT 
                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT 
                                        )); // TODO: Is it necesary the transfer bit?
      NVVK_DBG_NAME(m_freeListCounter.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_freeListCounter, 0,std::span(zeros3)));

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
    nvvk::DescriptorBindings bindings;
    // Add bindings here, if needed
    bindings.addBinding(shaderio::BindingPoints::sceneInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::renderTarget, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::normalBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::albedoBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::depthBuffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::globalGrid, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::aabbs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::objects, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::tLas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::clipMap, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::brickAtlas, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::buildJobQ, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::brickJobQ, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(shaderio::BindingPoints::freeListCounter, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

    // Creating the descriptor set and set layout from the bindings
    // TODO: You can put flags here, maybe this is important
    NVVK_CHECK(m_descPack.init(bindings, m_app->getDevice(), 1));
    NVVK_DBG_NAME(m_descPack.getLayout());
    NVVK_DBG_NAME(m_descPack.getPool());
    NVVK_DBG_NAME(m_descPack.getSet(0));

    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::sceneInfo), m_sceneInfoB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::globalGrid), m_globalGrid.descriptor);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::aabbs), m_sceneAabbB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::objects), m_sceneObjectsB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::tLas), m_asBuilder.tlas);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::clipMap), m_clipMap.descriptor);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::brickAtlas), m_brickAtlas.descriptor);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::buildJobQ), m_buildJobQueue.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::brickJobQ), m_brickJobQueue.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::freeListCounter), m_freeListCounter.buffer);
    vkUpdateDescriptorSets(m_app->getDevice(),  
                        static_cast<uint32_t>(writeContainer.size()),  
                        writeContainer.data(), 0, nullptr);
  }

  void createPipelineLayouts(){
    createPipelineLayout(&m_tracingLayout);
    createPipelineLayout(&m_lightingLayout);
    createPipelineLayout(&m_rtPipelineLayout);
    createPipelineLayout(&m_brickJobLayout);
    createPipelineLayout(&m_buildJobLayout);
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

  void compileAndCreateShaders(){
    // Destroy the previous shader module, if it exist
    vkDestroyShaderModule(m_app->getDevice(), m_tracingModule, nullptr);
    vkDestroyShaderModule(m_app->getDevice(), m_lightingModule, nullptr);
    vkDestroyShaderModule(m_app->getDevice(), m_rtModule, nullptr);
    vkDestroyShaderModule(m_app->getDevice(), m_brickJobModule, nullptr);
    vkDestroyShaderModule(m_app->getDevice(), m_buildJobModule, nullptr);

    createShaderModule(&m_tracingModule,"tracing.slang",tracing_slang);
    createShaderModule(&m_lightingModule,"lighting.slang",lighting_slang);
    createShaderModule(&m_rtModule,"raytracing.slang",raytracing_slang);
    createShaderModule(&m_brickJobModule,"brick.slang",brick_slang);
    createShaderModule(&m_buildJobModule,"build.slang",build_slang);

  }

  void createPipelines(){
    createComputePipeline(&m_tracingPipeline,&m_tracingLayout,&m_tracingModule);
    createComputePipeline(&m_lightingPipeline,&m_lightingLayout,&m_lightingModule);
    createRTPipeline(&m_rtPipeline,&m_rtPipelineLayout);
    createComputePipeline(&m_brickJobPipeline,&m_brickJobLayout,&m_brickJobModule);
    createComputePipeline(&m_buildJobPipeline,&m_buildJobLayout,&m_buildJobModule);
  }

  void createComputePipeline(VkPipeline* pipeline, VkPipelineLayout* pipelineLayout, VkShaderModule* shaderModule){
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = *shaderModule;
    stage.pName  = "computeMain";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = *pipelineLayout;

    NVVK_CHECK(vkCreateComputePipelines(
        m_app->getDevice(),
        VK_NULL_HANDLE,
        1,
        &cpci,
        nullptr,
        pipeline));
    NVVK_DBG_NAME(*pipeline);
  }

  void createRTPipeline(VkPipeline* pipeline, VkPipelineLayout* pipelineLayout){
    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eIntersection,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    for(auto& s : stages){
      s = {};
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    }

    stages[eRaygen].module        = m_rtModule;
    stages[eRaygen].pName         = "rgenMain";
    stages[eRaygen].stage         = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].module          = m_rtModule;
    stages[eMiss].pName           = "rmissMain";
    stages[eMiss].stage           = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].module    = m_rtModule;
    stages[eClosestHit].pName     = "rchitMain";
    stages[eClosestHit].stage     = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eIntersection].module  = m_rtModule;
    stages[eIntersection].pName   = "rintMain";
    stages[eIntersection].stage   = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    
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

    // closest hit shader
    group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.intersectionShader = eIntersection;
    shader_groups.push_back(group);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rtPipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    rtPipelineInfo.pStages                      = stages.data();
    rtPipelineInfo.groupCount                   = static_cast<uint32_t>(shader_groups.size());
    rtPipelineInfo.pGroups                      = shader_groups.data();
    rtPipelineInfo.maxPipelineRayRecursionDepth = std::max(3U, m_rtProperties.maxRayRecursionDepth);  // Ray depth
    rtPipelineInfo.layout                       = m_rtPipelineLayout;
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR (m_app->getDevice(), {}, {}, 1, &rtPipelineInfo, nullptr, &m_rtPipeline));
    NVVK_DBG_NAME(m_rtPipeline);

    // Create the shader binding table for this pipeline
    createShaderBindingTable(rtPipelineInfo);
  }

  void createShaderBindingTable(const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo){
    SCOPED_TIMER(__FUNCTION__);
    m_alloc.destroyBuffer(m_sbtBuffer);
    // Calculate required SBT buffer size
    size_t bufferSize = m_sbtGen.calculateSBTBufferSize(m_rtPipeline, rtPipelineInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(m_alloc.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                        m_sbtGen.getBufferAlignment()));

    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    // Populate the SBT buffer with shader handles and data using the CPU-mapped memory pointer
    NVVK_CHECK(m_sbtGen.populateSBTBuffer(m_sbtBuffer.address, bufferSize, m_sbtBuffer.mapping));
  }


  // Recompiles and waits for idle time to swap it into the pipeline
  void reloadShaders(){
    compileAndCreateShaders();
    vkDeviceWaitIdle(m_app->getDevice());
    vkDestroyPipeline(m_app->getDevice(),m_tracingPipeline,nullptr);
    vkDestroyPipeline(m_app->getDevice(),m_lightingPipeline,nullptr);
    vkDestroyPipeline(m_app->getDevice(),m_rtPipeline,nullptr);
    vkDestroyPipeline(m_app->getDevice(),m_brickJobPipeline,nullptr);
    vkDestroyPipeline(m_app->getDevice(),m_buildJobPipeline,nullptr);
    createPipelines();
  }


  void updateSceneBuffer(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    const glm::mat4& viewMatrix = m_cameraManip->getViewMatrix();
    const glm::mat4& projMatrix = m_cameraManip->getPerspectiveMatrix();

    m_sceneInfo.viewMatrix = glm::inverse(viewMatrix);
    m_sceneInfo.projMatrix = glm::inverse(projMatrix);
    m_sceneInfo.viewProjMatrix = glm::inverse(projMatrix * viewMatrix);
    m_sceneInfo.cameraPosition = m_cameraManip->getEye();  // Get the camera position

    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneInfoB.buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT});
    vkCmdUpdateBuffer(cmd, m_sceneInfoB.buffer, 0, sizeof(shaderio::SceneInfo), &m_sceneInfo);
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneInfoB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
  }

  // Accessor for camera manipulator
  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() const { return m_cameraManip; }

private:
  // Vulkan variables
  nvapp::Application*     m_app{};            // The application framework
  nvvk::ResourceAllocator m_alloc{};          // Resource allocator for Vulkan resources, used for buffers and images
  nvvk::StagingUploader   m_stagingUploader{};// Utility to upload data to the GPU, used for staging buffers and images
  nvvk::SamplerPool       m_samplerPool{};    // Texture sampler pool, used to acquire texture samplers for images
  nvvk::GBuffer           m_gBuffers{};       // The G-Buffer: color + depth
  nvslang::SlangCompiler  m_slangCompiler{};  // The Slang compiler used to compile the shaders
  nvvk::DescriptorPack    m_descPack;         // The descriptor bindings used to create the descriptor set layout and descriptor sets

  // Tracing Pipeline
  VkPipeline            m_tracingPipeline{};  // Compute pipeline
  VkPipelineLayout      m_tracingLayout{};    // Compute pipeline layout
  VkShaderModule        m_tracingModule{};    // Compute shader module for tracing

  // Lighting Pipeline (Deferred)
  VkPipeline            m_lightingPipeline{}; // Compute pipeline
  VkPipelineLayout      m_lightingLayout{};   // Compute pipeline layout
  VkShaderModule        m_lightingModule{};   // Compute shader module for lighting

  // RT Pipeline
  VkPipeline            m_rtPipeline{};       // Ray tracing pipeline
  VkPipelineLayout      m_rtPipelineLayout{}; // Ray tracing pipeline layout
  VkShaderModule        m_rtModule{};         // Raytracing shader module for rt pipeline

  // Brick generation Pipeline
  VkPipeline            m_brickJobPipeline{};  // Compute pipeline
  VkPipelineLayout      m_brickJobLayout{};    // Compute pipeline layout
  VkShaderModule        m_brickJobModule{};    // Compute shader module for brick generation

  // Build job Pipeline
  VkPipeline            m_buildJobPipeline{};  // Compute pipeline
  VkPipelineLayout      m_buildJobLayout{};    // Compute pipeline layout
  VkShaderModule        m_buildJobModule{};    // Compute shader module for build jobs

  // Acceleration Structure Components
  nvvk::AccelerationStructureHelper        m_asBuilder{};

  // Direct SBT management
  nvvk::SBTGenerator    m_sbtGen;
  nvvk::Buffer          m_sbtBuffer;         // Buffer for shader binding table

  // Ray Tracing Properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  // Push constants to send 
  shaderio::PushConstant m_pushConst = {.time = 0.0f};

  // Scene information. TODO: encapsulate this into a class
  shaderio::SceneInfo   m_sceneInfo{};          // Struct containing the scene information
  nvvk::Buffer          m_sceneInfoB{};         // Buffer binded to the UBO of scene info
  nvvk::Buffer          m_sceneAabbB{};         // Buffer binded to the scene aabbs array
  nvvk::Buffer          m_sceneObjectsB{};      // Buffer binded to the scene objects array

  // Job queues 
  nvvk::Buffer m_buildJobQueue{};       // Queue for the Build jobs
  nvvk::Buffer m_brickJobQueue{};       // Queue for the Brick jobs
  nvvk::Buffer m_freeListCounter{};   // Index of next free pointer in the free list

  // 3D textures
  nvvk::Image m_globalGrid;   // Dense value grid
  nvvk::Image m_clipMap;      // 3D map of pointers to the brick atlas
  nvvk::Image m_brickAtlas;   // Atlas where all the bricks are stored

  // Pre-built components
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip{std::make_shared<nvutils::CameraManipulator>()}; // Camera manipulator
  nvshaders::Tonemapper    m_tonemapper{};      // Tonemapper for post-processing effects
  shaderio::TonemapperData m_tonemapperData{};  // Tonemapper data used to pass parameters to the tonemapper shader

  // UI params
  bool m_debugActive = false;
  int m_debugMode = 0;
  bool m_RTX_ON = false;

  // Scene
  Scene m_scene;

  // Test variables TODO: Remove after debugging
  int m_testSize = 0;
  glm::vec3 m_testMed;

  // Startup managers for profiler and paramter registry
  Info m_info;

  // TODO: Figure out how to use the profiler tool
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;
};


int main(int argc, char** argv)
{
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

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {
        {VK_KHR_SWAPCHAIN_EXTENSION_NAME},
        {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},                                    // Required for premade modules, like the tonemapper
        {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},     // Build acceleration structures
        {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},  // Use vkCmdTraceRaysKHR
        {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},                           // Required by ray tracing pipeline
      },
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
    ImGuiID objectID   = ImGui::DockBuilderSplitNode(sceneID, ImGuiDir_Down,  0.33f, nullptr, &sceneID);
    ImGuiID loggerID  = ImGui::DockBuilderSplitNode(centerNode, ImGuiDir_Down,  0.3f, nullptr, &centerNode);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.5f, nullptr, &loggerID);

    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGui::DockBuilderDockWindow("Scene", sceneID);
    ImGui::DockBuilderDockWindow("Object", objectID);
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
  

  // enter the main loop
  app.run();

  // Cleanup in reverse order
  app.deinit();
  vkContext.deinit();

  return 0;
}
