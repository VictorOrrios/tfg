#pragma once
/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SHADERIO_H
#define SHADERIO_H

#ifdef __cplusplus
#define CHECK_STRUCT_ALIGNMENT(_s) static_assert(sizeof(_s) % 8 == 0);
#define CHECK_GRID_ALIGNMENT(_s) static_assert((_s) % 8 == 0);
#define CHECK_ARRAY_SIZE(_a) static_assert(int(sizeof(_a) / sizeof((_a)[0])) == CLIPMAP_LEVELS)
#elif defined(__SLANG__)
#define CHECK_STRUCT_ALIGNMENT(_s)
#define CHECK_GRID_ALIGNMENT(_s)
#define CHECK_ARRAY_SIZE(_a)
#endif

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

// Dispatch group counts
#define WORKGROUP_SIZE_1D 64
#define WORKGROUP_SIZE_2D 16
#define WORKGROUP_SIZE_3D 8

// Buffers static max limit
#define MAX_SCENE_OBJECTS 1024
#define MAX_SCENE_DYNAMIC_OBJECTS 512
#define MAX_MATERIALS 32
#define BRICK_PER_ATLAS_AXIS 512
const static int NUM_BRICKS_IN_ATLAS = BRICK_PER_ATLAS_AXIS*BRICK_PER_ATLAS_AXIS;

// Global grid parameters
#define NUM_BRICKS_PER_AXIS 64  // How many bricks per axis per level in clip map
#define L0_AXIS_WORLD_SIZE  2.0 // Axis size of the first clip map level
#define CLIPMAP_LEVELS      8   // How many levels are in the clip map | WARNING: If updated then all sizes MUST BE UPDATED TO
#define BRICK_SIZE          8   // How many values per axis does a brick store  
#define MAT_PER_BRICK_AXIS  4   // How many materials are stored per brick axis
CHECK_GRID_ALIGNMENT(NUM_BRICKS_PER_AXIS) // Power of two needed for faster calculations

// Extent calculations
const static int NUM_VOXELS_PER_AXIS = NUM_BRICKS_PER_AXIS*(BRICK_SIZE-1);
const static int NUM_VALUES_PER_AXIS = NUM_BRICKS_PER_AXIS*BRICK_SIZE;

    
// Sizes definition
#define S_AXIS(level) (L0_AXIS_WORLD_SIZE * (1<<level))
#define S_BRICK(level) (S_AXIS(level) / NUM_BRICKS_PER_AXIS)
#define S_VOXEL(level) (S_BRICK(level) / (BRICK_SIZE - 1))
#define MAX_VOXEL_V(level) (float(2.5 * sqrt(3.0 * S_VOXEL(level) * S_VOXEL(level))))
#define MAX_BRICK_V(level) (float(sqrt(3.0 * S_BRICK(level) * S_BRICK(level)) / 2.0 + MAX_VOXEL_V(level)))

const static float AXIS_SIZES[CLIPMAP_LEVELS] = {
  S_AXIS(0),S_AXIS(1),S_AXIS(2),S_AXIS(3),S_AXIS(4),S_AXIS(5),S_AXIS(6),S_AXIS(7)};
const static float BRICK_SIZES[CLIPMAP_LEVELS] = {
  S_BRICK(0),S_BRICK(1),S_BRICK(2),S_BRICK(3),S_BRICK(4),S_BRICK(5),S_BRICK(6),S_BRICK(7)};
const static float VOXEL_SIZES[CLIPMAP_LEVELS] = {
  S_VOXEL(0),S_VOXEL(1),S_VOXEL(2),S_VOXEL(3),S_VOXEL(4),S_VOXEL(5),S_VOXEL(6),S_VOXEL(7)};
const static float MAX_VOXEL_VALUES[CLIPMAP_LEVELS] = {
  MAX_VOXEL_V(0),MAX_VOXEL_V(1),MAX_VOXEL_V(2),MAX_VOXEL_V(3),MAX_VOXEL_V(4),MAX_VOXEL_V(5),MAX_VOXEL_V(6),MAX_VOXEL_V(7)};
const static float MAX_BRICK_VALUES[CLIPMAP_LEVELS] = {
  MAX_BRICK_V(0),MAX_BRICK_V(1),MAX_BRICK_V(2),MAX_BRICK_V(3),MAX_BRICK_V(4),MAX_BRICK_V(5),MAX_BRICK_V(6),MAX_BRICK_V(7)};

/* 
const static float AXIS_SIZES[CLIPMAP_LEVELS] = {
  S_AXIS(0),S_AXIS(1)};
const static float BRICK_SIZES[CLIPMAP_LEVELS] = {
  S_BRICK(0),S_BRICK(1)};
const static float VOXEL_SIZES[CLIPMAP_LEVELS] = {
  S_VOXEL(0),S_VOXEL(1)};
const static float MAX_VOXEL_VALUES[CLIPMAP_LEVELS] = {
  MAX_VOXEL_V(0),MAX_VOXEL_V(1)};
const static float MAX_BRICK_VALUES[CLIPMAP_LEVELS] = {
  MAX_BRICK_V(0),MAX_BRICK_V(1)};
 */
/*    
   const static float AXIS_SIZES[CLIPMAP_LEVELS] = {
  S_AXIS(0)};
const static float BRICK_SIZES[CLIPMAP_LEVELS] = {
  S_BRICK(0)};
const static float VOXEL_SIZES[CLIPMAP_LEVELS] = {
  S_VOXEL(0)};
const static float MAX_VOXEL_VALUES[CLIPMAP_LEVELS] = {
  MAX_VOXEL_V(0)};
const static float MAX_BRICK_VALUES[CLIPMAP_LEVELS] = {
  MAX_BRICK_V(0)};
  
 */
// Build & Brick jobs constants
#define MAX_BUILD_JOB_SIZE 8
#define BRICK_JOB_GROUP_X_DISPATCH_SIZE 256
const static uint MAX_NUM_BUILD_JOBS = 512*512;
const static uint MAX_NUM_BRICK_JOBS = MAX_NUM_BUILD_JOBS*MAX_BUILD_JOB_SIZE*MAX_BUILD_JOB_SIZE;

// Dirty bit definitions for mutual exclusion
#define DIRTY_BIT 0x80000000        // Most significant bit of a 32 bit variable
#define NOT_DIRTY_BIT (~DIRTY_BIT) 

// Magic pointer indicating unirform values in brick (not stored in atlas)
const static uint UNIFORM_POSITIVE_BRICK_POINTER = NUM_BRICKS_IN_ATLAS+1;
const static uint UNIFORM_NEGATIVE_BRICK_POINTER = UNIFORM_POSITIVE_BRICK_POINTER+1;

// Rng buffers and images sizes
#define NOISE_TEX_SIZE 1024
#define MAX_NUM_AO_KERNELS 256
#define MAX_NUM_SHADOW_KERNELS 24

// User constants
#define MAX_SHININESS 100

// Shared between Host and Device
enum BindingPoints{
  sceneInfo = 0,
  renderTarget,
  normalBuffer,
  albedoBuffer,
  depthBuffer,
  shadowBuffer,
  shadowSampler,
  shadowScratchBuffer,
  shadowScratchSampler,
  positionBuffer,
  aoBuffer,
  aoScratchBuffer,
  aoSample,
  aoScratchSample,
  gSampler,
  aabbs,
  objects,
  materials,
  dynamicObjects,
  tLas,
  bLas,
  instances,
  clipMap,
  brickAtlas,
  matAtlas,
  buildJobQ,
  brickJobQ,
  counters,
  indirectCommands,
  freeList,
  noise,
  aoKernels,
  shadowKernels,
};

enum Counters{
  nextBrickJob = 0,
  freeCounter = 1,
  allocCounter = 2
};

enum DebugModes{
  dmNone = 0,
  dmDebug,
  dmAlbedo,
  dmNormal,
  dmDepth,
  dmShadow,
  dmPosition,
  dmAO,
  dmBoundingBox
};
  
enum class PrimType { Empty=0, Box, Sphere, Torus, Snowman, Plane };

struct LightinParams{
  float3 lightDir         = normalize(float3(0.9f,0.2f,0.2f));
  float3 lightColor       = float3(0.644, 0.635, 0.608);
  float  lightPower       = 4.0f;
  float3 ambientTop       = float3(0.3f, 0.35f, 0.5f);
  float3 ambientBottom    = float3(0.1f, 0.1f, 0.1f);
  float3 fogColor         = float3(0.5f, 0.6f, 0.7f);
  float  fogDensity       = 0.03F;
  float  aoRadius         = 0.5f;
  float  aoBias           = 0.001f;
  int    aoSamples        = 80;
  int    aoTexelSize      = 2;
  int    shadowSamples    = 2;
  int    shadowTexelSize  = 4;
  float  shadowSharpness  = 50.0f;
};

struct DebugParams{
  int mode; // Refers to shaderio::DebugModes
  int palette;
  bool brickPercent;
};

struct PhysicsParams{
  float dts;
  float time_dilation = 0.0;
  int sub_steps = 3;
  float3 gravity = float3(0.0f,-9.8f,0.0f);
};

struct PushConstant{
  float time;
  DebugParams debug;
  LightinParams lp;
  PhysicsParams pyp;
  int numObjects;
  int numDynamicObjects;
  uint frameCount = 0;
};

struct SceneInfo{
  float4x4  viewMatrixInv;
  float4x4  projMatrixInv;
  float4x4  viewMatrix;
  float4x4  projMatrix;
  float4    cameraPosition;
  int4      cameraId0;
  float4    cameraId0Pos;
};
CHECK_STRUCT_ALIGNMENT(SceneInfo)

struct SceneObject{
  float4x4 tInv;
  float4 spacing;
  float4 defP;
  float4 terrain;
  int4 limit_octaves;
  int type;
  int combOp;
  int repOp;
  int defOp;
  int morphPrim;
  float scale;
  float roundness;
  float smoothness;
  float morph;
  uint mat;
  bool physicsActive;
  uint _pad;
};
CHECK_STRUCT_ALIGNMENT(SceneObject)

struct Material{
  float4 albedo_shininess;
  float2 alpha_metalness;
};
CHECK_STRUCT_ALIGNMENT(Material)

struct BuildJob{
  int4 min_id_level;
  int4 num_b;
};
CHECK_STRUCT_ALIGNMENT(BuildJob)

struct BrickJob{
  int4 id_level;
};
CHECK_STRUCT_ALIGNMENT(BrickJob)

struct DynamicObject{
  float4x4 tInv;
  float4 position;
  float4 rotation;
  float4 prev_position;
  float4 inv_rotation;
  float4 prev_rotation;
  float4 vel;
  float4 omega;
  float4 inv_inertia;
  float4 pos_diff;
  float4 pos_delta;
  float4 omega_delta;
  int type;
  float scale;
  float inv_mass;
  int id;
};  
CHECK_STRUCT_ALIGNMENT(DynamicObject)

struct DispatchIndirectCommand {
  uint x;
  uint y;
  uint z;
  uint _pad;
};
CHECK_STRUCT_ALIGNMENT(DispatchIndirectCommand)

struct InstanceData{
  float3x4 transform; // Row-major
  uint instanceCustomIndex : 24;
  uint mask                : 8;
  uint instanceSBTOffset   : 24;
  uint flags               : 8;
  uint64_t blasAddress;
};
CHECK_STRUCT_ALIGNMENT(InstanceData)


NAMESPACE_SHADERIO_END()

#endif
