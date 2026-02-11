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
#elif defined(__SLANG__)
#define CHECK_STRUCT_ALIGNMENT(_s)
#endif

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define WORKGROUP_SIZE 32

// Shared between Host and Device
enum BindingPoints
{
  sceneInfo = 0,
  renderTarget = 1,
  normalBuffer = 2,
  albedoBuffer = 3,
  depthBuffer  = 4,
};

struct LightinParams{
  float3 lightDir       = normalize(float3(-1.0,-1.0,-1.0));
  float3 lightColor     = float3(1.0f, 0.95f, 0.8f);
  float3 ambientTop     = float3(0.3f, 0.35f, 0.5f);
  float3 ambientBottom  = float3(0.1f, 0.1f, 0.1f);
  float3 fogColor       = float3(0.5f, 0.6f, 0.7f);
  float  fogDensity     = 0.05F;
};

struct PushConstant{
  float time;
  LightinParams lp;
};

struct SceneInfo{
  float4x4  viewProjMatrix;
  float4x4  viewMatrix;
  float4x4  projMatrix;
  float3    cameraPosition;     // Camera position in world space
  float     _pad;
};
CHECK_STRUCT_ALIGNMENT(SceneInfo)



NAMESPACE_SHADERIO_END()

#endif
