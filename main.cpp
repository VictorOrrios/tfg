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
#include <nvgui/sky.hpp>                   // Sky widget
#include <nvgui/tonemapper.hpp>            // Tonemapper widget
#include <nvshaders_host/sky.hpp>          // Sky shader
#include <nvshaders_host/tonemapper.hpp>   // Tonemapper shader
#include <nvgui/camera.hpp>                // Camera widget
#include <nvvk/formats.hpp>
#include <nvvk/shaders.hpp>
#include <nvvk/pipeline.hpp>
#include <nvvk/compute_pipeline.hpp>


class AppElement : public nvapp::IAppElement
{
public:
  struct Info
  {
    nvutils::ProfilerManager*   profilerManager{};
    nvutils::ParameterRegistry* parameterRegistry{};
  };


  AppElement(const Info& info)
      : m_info(info)
  {
    // let's add a command-line option to toggle animation
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

    setupSlangCompiler();         // Setup slang compiler with correct build config flags
    setupGBuffers();              // Set up the GBuffers to render to
    createScene();                // Create the scene and fill it up with sdfs
    createDescriptorSetLayout();  // Create the descriptor set layout for the pipeline
    createPipelineLayout();       // Create the pipeline layout
    compileAndCreateShaders();    // Compile the shaders and create the shader modules
    createComputePipeline();      // Create the pipeline using the layout and the shaders

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
    vkDestroyPipeline(device,m_computePipeline,nullptr);
    vkDestroyPipelineLayout(device,m_pipelineLayout,nullptr);
    vkDestroyShaderModule(device,m_shaderModule,nullptr);

    m_gBuffers.deinit();
    m_stagingUploader.deinit();
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

    ImGui::End();

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    // TODO: Insert gbuffer
    ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  //---------------------------------------------------------------------------------------------------------------
  // This renders the toolbar of the window
  // - Called when the ImGui menu is rendered
  void onUIMenu() override
  {
    bool vsync = m_app->isVsync();

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
    VkWriteDescriptorSet m_writeSet = m_descPack.makeWrite(shaderio::BindingPoints::gBuffers);
    writeContainer.append(m_writeSet, m_gBuffers.getDescriptorImageInfo());
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

    // Update data
    shaderio::PushConstant pc{};
    pc.time = static_cast<float>(ImGui::GetTime());

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);  
    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout,
                            0, 1, m_descPack.getSetPtr(), 0, nullptr);  
    // Push constants
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &pc);
    // Dispatch
    VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
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
    // Acquiring the texture sampler which will be used for displaying the GBuffer
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // Create the G-Buffers
    nvvk::GBufferInitInfo gBufferInit{
        .allocator      = &m_alloc,
        .colorFormats   = {VK_FORMAT_R32G32B32A32_SFLOAT,VK_FORMAT_R8G8B8A8_UNORM},  // Render target, tonemapped
        .depthFormat    = nvvk::findDepthFormat(m_app->getPhysicalDevice()),
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    };
    m_gBuffers.init(gBufferInit);
  }

  void createScene(){
    // TODO
  }

  //---------------------------------------------------------------------------------------------------------------
  // The Vulkan descriptor set defines the resources that are used by the shaders.
  // Here we add the bindings for the textures.
  void createDescriptorSetLayout(){
    nvvk::DescriptorBindings bindings;
    // Add bindings here, if needed
    bindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);

    // Creating the descriptor set and set layout from the bindings
    // TODO: You can put flags here, maybe this is important
    NVVK_CHECK(m_descPack.init(bindings, m_app->getDevice(), 1));
    NVVK_DBG_NAME(m_descPack.getLayout());
    NVVK_DBG_NAME(m_descPack.getPool());
    NVVK_DBG_NAME(m_descPack.getSet(0));
  }

  void createPipelineLayout(){
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
    NVVK_CHECK(vkCreatePipelineLayout(m_app->getDevice(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
    NVVK_DBG_NAME(m_pipelineLayout);
  }

  void compileAndCreateShaders(){
    // Destroy the previous shader module, if it exist
    vkDestroyShaderModule(m_app->getDevice(), m_shaderModule, nullptr);

    // Get .slang file and compile to spirv
    std::filesystem::path shaderSource = nvutils::findFile("compute.slang", nvsamples::getShaderDirs());
    if(!m_slangCompiler.compileFile(shaderSource)){
      LOGE("Error compiling shader: %s\n%s\n", shaderSource.string().c_str(),
           m_slangCompiler.getLastDiagnosticMessage().c_str());
    }

    // Create shader module using the pcode
    const uint32_t* spirvPtr = reinterpret_cast<const uint32_t*>(m_slangCompiler.getSpirv());
    size_t spirvWordCount = m_slangCompiler.getSpirvSize() / sizeof(uint32_t);
    LOGI("Compiled SPIRV word count:%zu\n",spirvWordCount);
    NVVK_CHECK(nvvk::createShaderModule(m_shaderModule, m_app->getDevice(), 
    std::span<const uint32_t>(spirvPtr, spirvWordCount)));
    NVVK_DBG_NAME(m_shaderModule);
  }

  void createComputePipeline(){
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = m_shaderModule;
    stage.pName  = "computeMain";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = stage;
    cpci.layout = m_pipelineLayout;

    NVVK_CHECK(vkCreateComputePipelines(
        m_app->getDevice(),
        VK_NULL_HANDLE,
        1,
        &cpci,
        nullptr,
        &m_computePipeline));
    NVVK_DBG_NAME(m_computePipeline);
  }


  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
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

  // Pipeline
  VkPipeline            m_computePipeline;    // Compute shader module
  VkShaderModule        m_shaderModule{};     // Compute shader module
  VkPipelineLayout      m_pipelineLayout{};   // Compute pipeline layout
  nvvk::DescriptorPack  m_descPack;           // The descriptor bindings used to create the descriptor set layout and descriptor sets

  // Push constants to send 
  shaderio::PushConstant m_pushConst = {.time = 0.0f};

  // Pre-built components
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip{std::make_shared<nvutils::CameraManipulator>()}; // Camera manipulator

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
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
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

  // add the sample main element
  app.addElement(appElement);
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
