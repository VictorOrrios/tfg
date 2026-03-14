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
#elif defined(__SLANG__)
#define CHECK_STRUCT_ALIGNMENT(_s)
#define CHECK_GRID_ALIGNMENT(_s);
#endif

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

// Dispatch group counts
#define WORKGROUP_SIZE_2D 32
#define WORKGROUP_SIZE_3D 8

// Buffers static max limit
#define MAX_SCENE_OBJECTS  1024
#define BRICK_PER_ATLAS_AXIS 1024

#define NUM_BRICKS_PER_AXIS  64
CHECK_GRID_ALIGNMENT(NUM_BRICKS_PER_AXIS)
#define CLIPMAP_LEVELS 2
const static int UNIFORM_POSITIVE_BRICK_POINTER = BRICK_PER_ATLAS_AXIS*BRICK_PER_ATLAS_AXIS+1;
const static int UNIFORM_NEGATIVE_BRICK_POINTER = UNIFORM_POSITIVE_BRICK_POINTER+1;

#define L0_AXIS_WORLD_SIZE 2.0

#define BRICK_SIZE  8
const static float BRICK_AXIS_SIZE = L0_AXIS_WORLD_SIZE/NUM_BRICKS_PER_AXIS;
const static int NUM_VOXELS_PER_AXIS = NUM_BRICKS_PER_AXIS*(BRICK_SIZE-1);
const static int NUM_VALUES_PER_AXIS = NUM_BRICKS_PER_AXIS*BRICK_SIZE;
const static float VOXEL_SIZE = BRICK_AXIS_SIZE/(BRICK_SIZE-1);
const static float MAX_VOXEL_VALUE = 2.5*sqrt(3.0*VOXEL_SIZE*VOXEL_SIZE);
const static float MAX_BRICK_CENTER_VALUE = sqrt(3.0*BRICK_AXIS_SIZE*BRICK_AXIS_SIZE)/2.0+MAX_VOXEL_VALUE;

#define MAX_BUILD_JOB_SIZE 8
#define MAX_NUM_BUILD_JOBS 2048
#define BRICK_JOB_GROUP_X_DISPATCH_SIZE 256
const static int MAX_NUM_BRICK_JOBS = MAX_NUM_BUILD_JOBS*MAX_BUILD_JOB_SIZE*MAX_BUILD_JOB_SIZE*MAX_BUILD_JOB_SIZE;

#define DIRTY_BIT 0x80000000 // Most significant bit of a 32 bit variable
#define NOT_DIRTY_BIT (~DIRTY_BIT) 

// TODO: Clean this file and iclude de std packing used per buffer

// Shared between Host and Device
enum BindingPoints
{
  sceneInfo = 0,
  renderTarget = 1,
  normalBuffer = 2,
  albedoBuffer = 3,
  depthBuffer  = 4,
  globalGrid = 5,
  aabbs = 6,
  objects = 7,
  tLas = 8,
  clipMap = 9,
  brickAtlas = 10,
  buildJobQ = 11,
  brickJobQ = 12,
  freeListCounter = 13,
};

struct LightinParams{
  float3 lightDir       = normalize(float3(-0.4f,-1.0f,-0.2f));
  float3 lightColor     = float3(1.0f, 0.95f, 0.8f);
  float3 ambientTop     = float3(0.3f, 0.35f, 0.5f);
  float3 ambientBottom  = float3(0.1f, 0.1f, 0.1f);
  float3 fogColor       = float3(0.5f, 0.6f, 0.7f);
  float  fogDensity     = 0.03F;
};

struct DebugParams{
  int mode;     // 0: Off, 1: Debug color, 2: Albedo, 3: Normal, 4:Depth, 5:BBox
  int palette;
};

struct PushConstant{
  float time;
  DebugParams debug;
  LightinParams lp;
  int numObjects;
  int numBrickJobs;
};

struct SceneInfo{
  float4x4  viewProjMatrix;
  float4x4  viewMatrix;
  float4x4  projMatrix;
  float3    cameraPosition;
  float     _pad;
};
CHECK_STRUCT_ALIGNMENT(SceneInfo)

struct SceneObject{
  float4x4 tInv;
  float4 position;
  float4 rotation;
  float4 spacing;
  float4 defP;
  int4 limit;
  int type;
  int combOp;
  int repOp;
  int defOp;
  float scale;
  float roundness;
  float smoothness;
  float _padding;
};
CHECK_STRUCT_ALIGNMENT(SceneObject)

struct BuildJob{
  int4 min_b_Q_offset;
  int4 num_b_level;
};
CHECK_STRUCT_ALIGNMENT(BuildJob)

struct BrickJob{
  int4 id;
  int2 valid_level;
};
CHECK_STRUCT_ALIGNMENT(BrickJob)


NAMESPACE_SHADERIO_END()

#endif
