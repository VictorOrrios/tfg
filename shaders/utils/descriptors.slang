#pragma once

#include "../shaderio.h"    // Shared definitions with CPU
#include "bbox.slang"       // AABB helpers

//----------------------------
// External memory definition
//----------------------------

// Push constants
[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

// GBuffers
[[vk::binding(BindingPoints::renderTarget)]] RWTexture2D<float4> outImage;
[[vk::binding(BindingPoints::normalBuffer)]] RWTexture2D<float4> normalImage;
[[vk::binding(BindingPoints::albedoBuffer)]] RWTexture2D<float4> albedoImage;
[[vk::binding(BindingPoints::depthBuffer)]] RWTexture2D<float> depthImage;
[[vk::binding(BindingPoints::shadowBuffer)]] RWTexture2D<float> shadowImage;
[[vk::binding(BindingPoints::shadowSampler)]] Texture2D<float> shadowSample;
[[vk::binding(BindingPoints::shadowScratchBuffer)]] RWTexture2D<float> shadowScratchImage;
[[vk::binding(BindingPoints::shadowScratchSampler)]] Texture2D<float> shadowScratchSample;
[[vk::binding(BindingPoints::positionBuffer)]] RWTexture2D<float4> positionImage;
[[vk::binding(BindingPoints::aoBuffer)]] RWTexture2D<float> aoImage;
[[vk::binding(BindingPoints::aoScratchBuffer)]] RWTexture2D<float> aoScratchImage;
[[vk::binding(BindingPoints::aoSample)]] Texture2D<float> aoSample;
[[vk::binding(BindingPoints::aoScratchSample)]] Texture2D<float> aoScratchSample;
[[vk::binding(BindingPoints::gSampler)]] SamplerState gSampler;

// Scene
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;
[[vk::binding(BindingPoints::aabbs)]] StructuredBuffer<Bbox> aabbs;
[[vk::binding(BindingPoints::objects)]] StructuredBuffer<SceneObject> objects;
[[vk::binding(BindingPoints::materials)]] StructuredBuffer<Material> materials;
[[vk::binding(BindingPoints::dynamicObjects)]] RWStructuredBuffer<DynamicObject> dynamic_objects;

// Grid
[[vk::binding(BindingPoints::clipMap)]] RWTexture3D<uint32_t> clipMap;
[[vk::binding(BindingPoints::brickAtlas)]] RWTexture3D<float> brickAtlas;
[[vk::binding(BindingPoints::matAtlas)]] RWTexture3D<float4> matAtlas;

// Generation
[[vk::binding(BindingPoints::buildJobQ)]] StructuredBuffer<BuildJob> build_job_queue;
[[vk::binding(BindingPoints::brickJobQ)]] RWStructuredBuffer<BrickJob> brick_job_queue;
[[vk::binding(BindingPoints::counters)]] RWStructuredBuffer<uint32_t> counters;
[[vk::binding(BindingPoints::indirectCommands)]] RWStructuredBuffer<DispatchIndirectCommand> indirect_commands;
[[vk::binding(BindingPoints::freeList)]] RWStructuredBuffer<uint32_t> free_list;

// Raytracing
[[vk::binding(BindingPoints::tLas)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(BindingPoints::instances)]] RWStructuredBuffer<InstanceData> instances;

// RNG
[[vk::binding(BindingPoints::noise)]] Texture2D<float4> noiseImage;
[[vk::binding(BindingPoints::aoKernels)]] StructuredBuffer<float3> aoKernels;
[[vk::binding(BindingPoints::shadowKernels)]] StructuredBuffer<float3> shadowKernels;