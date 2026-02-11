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
// TODO: Change banner

#include "shaderio.h"

[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

[[vk::binding(BindingPoints::renderTarget)]] RWTexture2D<float4> outImage;
[[vk::binding(BindingPoints::normalBuffer)]] RWTexture2D<float4> normalImage;
[[vk::binding(BindingPoints::albedoBuffer)]] RWTexture2D<float4> albedoImage;
[[vk::binding(BindingPoints::depthBuffer)]] RWTexture2D<float> depthImage;


[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float2 fragCoord = threadIdx.xy;
  uint2 iResolution;
  outImage.GetDimensions(iResolution.x, iResolution.y);
  float2 uv = fragCoord/iResolution;
  
  float3 normal = normalImage[int2(fragCoord)].xyz;
  float3 albedo = albedoImage[int2(fragCoord)].xyz;
  float depth = depthImage[int2(fragCoord)];

  float3 sunDir = normalize(float3(1.0,1.0,0.5));
  float3 sunColor = float3(0.839, 0.921, 1);
  float3 shadowColor = float3(0, 0.172, 0.341);

  float3 finalColor;

  if(depth >= 0.0){
    finalColor = lerp(shadowColor,sunColor,dot(normal,sunDir));
  }else{
    finalColor = float3(0.0);
  }


  outImage[int2(fragCoord)] = float4(finalColor, 1.0F);
}
