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

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_vulkan.h>

#include "shaders/shaderio.h"           // Shared between host and device

#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/logger.hpp>
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
#include <nvgui/camera.hpp>                // Camera widget
#include "shaders/shaderio.h"                 


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
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    };

    // Initialize core components
    NVVK_CHECK(m_alloc.init(allocatorInfo));
    m_samplerPool.init(app->getDevice());
    m_stagingUploader.init(&m_alloc, true);

    // TODO: Figure out how to use the profiler tool
    // Init profiler with a single queue
    m_profilerTimeline = m_info.profilerManager->createTimeline({"graphics"});
    m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

    m_stagingUploader.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
    m_profilerGpuTimer.deinit();
    m_info.profilerManager->destroyTimeline(m_profilerTimeline);
  }

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
    ImGui::End();
  }

  void onPreRender() override { m_profilerTimeline->frameAdvance(); }

  void onRender(VkCommandBuffer cmd) override
  {
    
  }

  // Called if showMenu is true
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

private:

  nvapp::Application*     m_app{};            // The application framework
  nvvk::ResourceAllocator m_alloc{};          // Resource allocator for Vulkan resources, used for buffers and images
  nvvk::StagingUploader  m_stagingUploader{}; // Utility to upload data to the GPU, used for staging buffers and images
  nvvk::SamplerPool      m_samplerPool{};     // Texture sampler pool, used to acquire texture samplers for images
  nvvk::GBuffer          m_gBuffers{};        // The G-Buffer: color + depth
  nvslang::SlangCompiler m_slangCompiler{};   // The Slang compiler used to compile the shaders

  // Pipeline
  VkShaderEXT m_shader{};                         // Compute shader module
  VkPipelineLayout m_pipelineLayout{};            // Compute pipeline layout
  nvvk::DescriptorPack m_descPack;                // The descriptor bindings used to create the descriptor set layout and descriptor sets

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
