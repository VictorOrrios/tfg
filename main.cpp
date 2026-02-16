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


#include "glm/detail/type_vec2.hpp"
#include "glm/fwd.hpp"
#include "glm/geometric.hpp"
#include "glm/matrix.hpp"
#include "nvvkgltf/tinygltf_utils.hpp"
#include <cstdint>
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

#include "_autogen/tracing.slang.h"
#include "_autogen/lighting.slang.h"
#include "_autogen/sky_simple.slang.h"
#include "_autogen/tonemapper.slang.h"

#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>              // Timers for profiling
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
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

const char* DebugModes[] = {
    "Debug color",
    "Albedo",
    "Normal",
    "Depth",
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

  void onAttach(nvapp::Application* app) override
  {
    m_app                                = app;
    VmaAllocatorCreateInfo allocatorInfo = {
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4, 
    };

    // Initialize core components
    NVVK_CHECK(m_alloc.init(allocatorInfo));
    m_samplerPool.init(app->getDevice());
    m_stagingUploader.init(&m_alloc, true);

    // TODO set back to on when proper lighting solution is made
    // Set tonemapping off by default
    m_tonemapperData.isActive = 0;

    setupSlangCompiler();         // Setup slang compiler with correct build config flags
    createScene();                // Create the scene and fill it up with sdfs
    setupGBuffers();              // Set up the GBuffers to render to
    create3DTextures();           // Creates the different 3d textures used to store voxel grid data
    createDescriptorSetLayout();  // Create the descriptor set layout for the pipeline
    createPipelineLayouts();      // Create the pipeline layouts
    compileAndCreateShaders();    // Compile the shaders and create the shader modules
    createPipelines();      // Create the pipelines using the layouts and the shaders

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
    vkDestroyPipelineLayout(device,m_tracingLayout,nullptr);
    vkDestroyPipelineLayout(device,m_lightingLayout,nullptr);
    vkDestroyShaderModule(device,m_tracingModule,nullptr);
    vkDestroyShaderModule(device,m_lightingModule,nullptr);

    m_alloc.destroyBuffer(m_sceneInfoB);
    m_alloc.destroyImage(m_globalGrid);

    m_gBuffers.deinit();
    m_stagingUploader.deinit();
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
      ImGui::Separator();
      ImGui::Text("Debug colors");
      ImGui::Checkbox("Active", &m_debugActive);
      ImGui::Combo("Mode", &m_debugMode, DebugModes, IM_ARRAYSIZE(DebugModes));
      ImGui::Combo("Palette", &m_pushConst.debug.palette, DebugPalettes, IM_ARRAYSIZE(DebugPalettes));
      if(m_debugActive){
        m_pushConst.debug.mode = m_debugMode + 1;
      }else{
        m_pushConst.debug.mode = 0;
      }
    }

    ImGui::End();

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

    tracingPass(cmd);
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
    VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE);
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
    vkCmdPushConstants(cmd, m_tracingLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
    // Dispatch
    VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
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
        { slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_HIGH }
    });
    m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation,
        { slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_NONE }
    });
    m_slangCompiler.addOption({slang::CompilerOptionName::WarningsAsErrors,
        { slang::CompilerOptionValueKind::Int, 1 }
    });
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

  void create3DTextures(){
    SCOPED_TIMER(__FUNCTION__);

    // Destroy if already created
    m_alloc.destroyImage(m_globalGrid);

    // ===============
    // Global grid paramters
    VkExtent3D extent = {100,100,100};  // XYZ size
    VkFormat format = VK_FORMAT_R32_SFLOAT;                   // Texel format
    glm::float32 clearValue = 10000.0f;

    std::array<uint32_t, 1> queueFamilies = {
        m_app->getQueue(0).familyIndex, // Compute queue
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
    vi.image = m_globalGrid.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vi.format = format;

    NVVK_CHECK(m_alloc.createImage(m_globalGrid, ci, vi));
    NVVK_DBG_NAME(m_globalGrid.image);
    NVVK_DBG_NAME(m_globalGrid.descriptor.imageView);

    // Setup sampler to be ortholinear storage with no interpolation or repetition
    VkSamplerCreateInfo si = DEFAULT_VkSamplerCreateInfo;
    si.magFilter    = VK_FILTER_NEAREST;
    si.minFilter    = VK_FILTER_NEAREST;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

    NVVK_CHECK(m_samplerPool.acquireSampler(m_globalGrid.descriptor.sampler, si));

    m_globalGrid.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    nvvk::cmdImageMemoryBarrier(cmd, {m_globalGrid.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkClearColorValue       clearColor = {{clearValue,clearValue,clearValue,clearValue}};
    vkCmdClearColorImage(cmd, m_globalGrid.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);

    updateTextureData(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Debugging information
    NVVK_DBG_NAME(m_globalGrid.image);
    NVVK_DBG_NAME(m_globalGrid.descriptor.sampler);
    NVVK_DBG_NAME(m_globalGrid.descriptor.imageView);
  }

  // TODO: Temporary, it fills out a grid with a sdf for testing
  inline float opUnion(float a, float b) { return glm::min(a, b); }
  inline float opSubtraction(float a, float b) { return glm::max(-a, b); }
  inline float opIntersection(float a, float b) { return glm::max(a, b); }
  inline float opXor(float a, float b) {
    return glm::max(glm::min(a, b), -glm::max(a, b));
  }
  inline float opSmoothUnion(float a, float b, float k) {
    k *= 4.0f;
    float h = glm::max(k - glm::abs(a - b), 0.0f);
    return glm::min(a, b) - h * h * 0.25f / k;
  }
  inline float opSmoothSubtraction(float a, float b, float k) {
    return -opSmoothUnion(a, -b, k);
  }
  inline float opSmoothIntersection(float a, float b, float k) {
    return -opSmoothUnion(-a, -b, k);
  }
  inline float sdSphere(const glm::vec3 &p, float s) {
    return glm::length(p) - s;
  }
  inline float sdPlane(const glm::vec3 &p, const glm::vec3 &n, float h) {
    return glm::dot(p, n) + h;
  }
  inline float sdCapsule(const glm::vec3 &p, const glm::vec3 &a,
                         const glm::vec3 &b, float r) {
    glm::vec3 pa = p - a;
    glm::vec3 ba = b - a;
    float h = glm::clamp(glm::dot(pa, ba) / glm::dot(ba, ba), 0.0f, 1.0f);
    return glm::length(pa - ba * h) - r;
  }
  inline float sdRoundedCylinder(const glm::vec3 &p, float ra, float rb,
                                 float h) {
    glm::vec2 d(glm::length(glm::vec2(p.x, p.z)) - ra + rb,
                glm::abs(p.y) - h + rb);
    return glm::min(glm::max(d.x, d.y), 0.0f) +
           glm::length(glm::max(d, glm::vec2(0.0f))) - rb;
  }
  float sdSnowMan(const glm::vec3 &p) {
    float r = sdSphere(p, 1.0f);
    r = opSmoothUnion(r, sdSphere(p - glm::vec3(0, 1.5f, 0), 0.6f), 0.1f);

    r = opSmoothUnion(r, sdSphere(p - glm::vec3(0.3f, 1.6f, 0.5f), 0.1f),
                      0.01f);
    r = opSmoothUnion(r, sdSphere(p - glm::vec3(-0.3f, 1.6f, 0.5f), 0.1f),
                      0.01f);
    r = opSmoothUnion(r, sdCapsule(p, glm::vec3(0.0f), glm::vec3(1.6f, 0.8f, 0.0f), 0.15f),
        0.05f);
    r = opSmoothUnion(r, sdCapsule(p, glm::vec3(0.0f), glm::vec3(-1.6f, 0.8f, 0.0f), 0.15f),
        0.05f);
    r = opSmoothUnion(r,sdCapsule(p, glm::vec3(0.0f, 1.4f, 0.0f),
                                glm::vec3(0.0f, 1.3f, 0.8f), 0.05f),
                      0.01f);
    r = opUnion(r, sdRoundedCylinder(p - glm::vec3(0.0f, 2.1f, 0.0f), 0.7f,
                                     0.05f, 0.1f));
    r = opUnion(r, sdRoundedCylinder(p - glm::vec3(0.0f, 2.5f, 0.0f), 0.4f,
                                     0.05f, 0.5f));
    return r;
  }
  std::vector<float> generateSDFGrid(int size, float scale){
    glm::vec3 center = glm::vec3(0.5,0.5,0.5);
    std::vector<float> data(size*size*size);
    for (int x = 0; x < size; x++) {
      for (int y = 0; y < size; y++) {
        for (int z = 0; z < size; z++) {
          glm::vec3 point = (glm::vec3(x+0.5f,y+0.5f,z+0.5f)/float(size) - center);
          // Sphere
          //float d = sdSphere(point/scale, 0.2);
          // Snowman
          float d = sdSnowMan(point/scale);
          data[x*size*size+y*size+z] = d*scale;
        }
      }
    }
    return data;
  }

  void updateTextureData(VkCommandBuffer cmd){
    NVVK_DBG_SCOPE(cmd);

    // TODO: This only works on start up because there is no concurrency problems

    assert(m_globalGrid.image);
    std::vector<float> imageData = generateSDFGrid(100, 0.15);
    assert(m_stagingUploader.isAppendedEmpty());
    nvvk::SemaphoreState cmdSemaphoreState{};
    NVVK_CHECK(m_stagingUploader.appendImage(m_globalGrid, std::span(imageData), m_globalGrid.descriptor.imageLayout, cmdSemaphoreState));
    m_stagingUploader.cmdUploadAppended(cmd);
  }

  void createScene(){
    SCOPED_TIMER(__FUNCTION__);
    nvvk::ResourceAllocator* allocator = m_stagingUploader.getResourceAllocator();
    
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

      NVVK_CHECK(allocator->createBuffer(m_sceneInfoB,
                                     std::span<const shaderio::SceneInfo>(&m_sceneInfo, 1).size_bytes(),
                                     VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
      NVVK_DBG_NAME(m_sceneInfoB.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_sceneInfoB, 0,
                                          std::span<const shaderio::SceneInfo>(&m_sceneInfo, 1)));

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

    // Creating the descriptor set and set layout from the bindings
    // TODO: You can put flags here, maybe this is important
    NVVK_CHECK(m_descPack.init(bindings, m_app->getDevice(), 1));
    NVVK_DBG_NAME(m_descPack.getLayout());
    NVVK_DBG_NAME(m_descPack.getPool());
    NVVK_DBG_NAME(m_descPack.getSet(0));

    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::sceneInfo), m_sceneInfoB.buffer);
    writeContainer.append(m_descPack.makeWrite(shaderio::BindingPoints::globalGrid), m_globalGrid.descriptor);
    vkUpdateDescriptorSets(m_app->getDevice(),  
                        static_cast<uint32_t>(writeContainer.size()),  
                        writeContainer.data(), 0, nullptr);
  }

  void createPipelineLayouts(){
    createPipelineLayout(&m_tracingLayout);
    createPipelineLayout(&m_lightingLayout);
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
    NVVK_DBG_NAME(m_tracingLayout);
  }

  void createShaderModule(VkShaderModule* shaderModule, const std::filesystem::path& filename, const std::span<const uint32_t> spirv){

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
    NVVK_CHECK(nvvk::createShaderModule(*shaderModule, m_app->getDevice(), 
    std::span<const uint32_t>(spirvPtr, spirvWordCount)));
    NVVK_DBG_NAME(*shaderModule);
  }

  void compileAndCreateShaders(){
    // Destroy the previous shader module, if it exist
    vkDestroyShaderModule(m_app->getDevice(), m_tracingModule, nullptr);
    vkDestroyShaderModule(m_app->getDevice(), m_lightingModule, nullptr);

    createShaderModule(&m_tracingModule,"tracing.slang",tracing_slang);
    createShaderModule(&m_lightingModule,"lighting.slang",lighting_slang);

  }

  void createPipelines(){
    createComputePipeline(&m_tracingPipeline,&m_tracingLayout,&m_tracingModule);
    createComputePipeline(&m_lightingPipeline,&m_lightingLayout,&m_lightingModule);
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

  // Recompiles and waits for idle time to swap it into the pipeline
  void reloadShaders(){
    compileAndCreateShaders();
    vkDeviceWaitIdle(m_app->getDevice());
    vkDestroyPipeline(m_app->getDevice(),m_tracingPipeline,nullptr);
    vkDestroyPipeline(m_app->getDevice(),m_lightingPipeline,nullptr);
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

    // Making sure the scene information buffer is updated before rendering
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneInfoB.buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT});
    vkCmdUpdateBuffer(cmd, m_sceneInfoB.buffer, 0, sizeof(shaderio::SceneInfo), &m_sceneInfo);
    nvvk::cmdBufferMemoryBarrier(cmd, {m_sceneInfoB.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT});
  }

  // Accessor for camera manipulator
  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() const { return m_cameraManip; }

private:

  nvapp::Application*     m_app{};            // The application framework
  nvvk::ResourceAllocator m_alloc{};          // Resource allocator for Vulkan resources, used for buffers and images
  nvvk::StagingUploader  m_stagingUploader{}; // Utility to upload data to the GPU, used for staging buffers and images
  nvvk::SamplerPool      m_samplerPool{};     // Texture sampler pool, used to acquire texture samplers for images
  nvvk::GBuffer          m_gBuffers{};        // The G-Buffer: color + depth
  nvslang::SlangCompiler m_slangCompiler{};   // The Slang compiler used to compile the shaders
  nvvk::DescriptorPack  m_descPack;           // The descriptor bindings used to create the descriptor set layout and descriptor sets

  // Tracing Pipeline
  VkPipeline            m_tracingPipeline{};  // Compute pipeline
  VkPipelineLayout      m_tracingLayout{};    // Compute pipeline layout
  VkShaderModule        m_tracingModule{};    // Compute shader module

  // Lighting Pipeline (Deferred)
  VkPipeline            m_lightingPipeline{}; // Compute pipeline
  VkPipelineLayout      m_lightingLayout{};   // Compute pipeline layout
  VkShaderModule        m_lightingModule{};   // Compute shader module for lighting

  // Push constants to send 
  shaderio::PushConstant m_pushConst = {.time = 0.0f};

  // Scene information. TODO: encapsulate this into a class
  shaderio::SceneInfo   m_sceneInfo{};        // Struct containing the scene information
  nvvk::Buffer          m_sceneInfoB{};       // Buffer binded to the UBO of scene info

  // 3D textures
  nvvk::Image m_globalGrid;

  // Pre-built components
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip{std::make_shared<nvutils::CameraManipulator>()}; // Camera manipulator
  nvshaders::Tonemapper    m_tonemapper{};      // Tonemapper for post-processing effects
  shaderio::TonemapperData m_tonemapperData{};  // Tonemapper data used to pass parameters to the tonemapper shader

  // UI params
  bool m_debugActive = false;
  int m_debugMode = 0;

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

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {
        {VK_KHR_SWAPCHAIN_EXTENSION_NAME},
        {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
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
    // right side panel container
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);

    // bottom panel container
    ImGuiID loggerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Log", loggerID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.4F, nullptr, &loggerID);
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
