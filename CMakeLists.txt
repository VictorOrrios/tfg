cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Prior adding the package one can disable certain libraries
# e.g.
option(NVPRO2_ENABLE_nvgl "" OFF)
option(NVPRO2_ENABLE_nvgpu_monitor "" OFF)

# Call FindNvproCore2 to download nvpro_core2 or find it if it is already
# downloaded, or is located as a subdirectory or up to two levels up.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)


#####################################################################################
# Compile shaders
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/_autogen")
file(GLOB SHADER_SLANG_FILES "shaders/*.slang")
list(APPEND SHADER_SLANG_FILES 
    ${NVSHADERS_DIR}/nvshaders/sky_simple.slang
    ${NVSHADERS_DIR}/nvshaders/tonemapper.slang
)
set(SHADER_INCLUDE_FLAGS "-I${NVSHADERS_DIR}")
compile_slang(
    "${SHADER_SLANG_FILES}"
    "${SHADER_OUTPUT_DIR}"
    GENERATED_SHADER_HEADERS
    EXTRA_FLAGS ${SHADER_INCLUDE_FLAGS}
)


#####################################################################################
# Source files for this project
#

# Utils
file(GLOB UTILS_SOURCES CONFIGURE_DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h
)



#####################################################################################
# Executable

add_executable(${PROJECT_NAME}
  main.cpp
)

target_sources(${PROJECT_NAME}
  PRIVATE
    ${UTILS_SOURCES}
    ${GENERATED_SHADER_HEADERS}
)

source_group(
  TREE ${CMAKE_CURRENT_SOURCE_DIR}/utils
  PREFIX "Utils"
  FILES ${UTILS_SOURCES}
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvimageformats
  nvpro2::nvslang
  nvpro2::nvvk
  nvpro2::nvshaders_host
)

add_project_definitions(${PROJECT_NAME})

# This sample doesn't need addtional files, but one might need to
# copy required dlls, additional commands etc. through this command
copy_to_runtime_and_install(${PROJECT_NAME} 
  FILES ${Slang_GLSLANG} ${Slang_GLSL_MODULE} ${Slang_DLL}
  PROGRAMS ${Slang_SLANGD_EXECUTABLE}
  AUTO
)
