/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"

[[vk::binding(0)]] RWTexture2D<float4> outImage;
[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;


[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float2 fragCoord = threadIdx.xy;
  uint2 iResolution;
  outImage.GetDimensions(iResolution.x, iResolution.y); //DispatchRaysDimensions();
  
  float3 finalColor = float3(fragCoord.x/iResolution.x,fragCoord.y/iResolution.y,0.5F);
  outImage[int2(fragCoord)] = float4(finalColor, 1.0F);
}
