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
#include "utils/palettes.slang"   // Debug color palette

[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

[[vk::binding(BindingPoints::renderTarget)]] RWTexture2D<float4> outImage;
[[vk::binding(BindingPoints::normalBuffer)]] RWTexture2D<float4> normalImage;
[[vk::binding(BindingPoints::albedoBuffer)]] RWTexture2D<float4> albedoImage;
[[vk::binding(BindingPoints::depthBuffer)]] RWTexture2D<float> depthImage;
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;

#ifdef NDEBUG
const static bool debugActive = false;
#else
const static bool debugActive = true;
#endif

float3 getV(float2 uv){
  float2 ndc = uv * 2.0 - 1.0;
  float4 rayClip = float4(ndc.x, ndc.y, -1.0, 1.0);

  float4 view = mul(rayClip,sceneInfo.projMatrix);
  float3 world = mul(float4(normalize(view.xyz),0.0),sceneInfo.viewMatrix).xyz;
  return normalize(world);
}

float3 scalarToColor(float x){
  return scalarToColor(x, pushConst.debug.palette);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float2 fragCoord = threadIdx.xy;
  uint2 iResolution;
  outImage.GetDimensions(iResolution.x, iResolution.y);
  float2 uv = fragCoord/iResolution;
  
  float3 N = normalImage[int2(fragCoord)].xyz;
  float3 albedo = albedoImage[int2(fragCoord)].xyz;
  float depth = depthImage[int2(fragCoord)];
  float3 finalColor;

  if(depth >= 0.0){
    float3 V = getV(uv);
    float3 L = -pushConst.lp.lightDir;
    float3 H = normalize(L+V);

    float NdotL = saturate(dot(N, L));
    float3 diffuse = albedo * pushConst.lp.lightColor * NdotL;

    // Hemispheric ambient
    float hemi = N.y * 0.5 + 0.5;
    float3 ambient = lerp(pushConst.lp.ambientBottom, pushConst.lp.ambientTop, hemi) * albedo;

    // Bling-Phong specular
    float spec = pow(saturate(dot(N, H)), 64);
    float3 specular = pushConst.lp.lightColor * spec * 0.3;

    // Rim lighting
    float rim = pow(1 - saturate(dot(N, V)), 3);
    float3 rimLight = pushConst.lp.lightColor * rim * 0.25;

    // Depth based fog
    float fog = 1 - exp(-depth * pushConst.lp.fogDensity);

    // Color calculation
    float3 color = diffuse + ambient + specular + rimLight;
    finalColor = lerp(color, pushConst.lp.fogColor, fog);
  }else{
    float fog = 1 - exp(-100.0F * pushConst.lp.fogDensity);
    finalColor = float3(pushConst.lp.fogColor * fog);
  }

  if(debugActive){
    if((pushConst.debug.mode == 1 || pushConst.debug.mode == 4) && uv.x > 0.98){
      // Palette visualizer
      if(uv.x - uv.y > 0.99){
        outImage[int2(fragCoord)] = float4(scalarToColor(-1),1.0);
      }else if(uv.x + uv.y > 1.99){
        outImage[int2(fragCoord)] = float4(scalarToColor(2),1.0);
      }else{
        outImage[int2(fragCoord)] = float4(scalarToColor(uv.y),1.0);
      }
    }else{
      switch(pushConst.debug.mode){
        default:
        case 0: outImage[int2(fragCoord)] = float4(finalColor, 1.0F); break;
        case 1: case 2: outImage[int2(fragCoord)] = float4(albedo, 1.0F); break;
        case 3: outImage[int2(fragCoord)] = float4(N, 1.0F); break;
        case 4: outImage[int2(fragCoord)] = float4(scalarToColor(depth), 1.0F); break;
      }
    }
  }else{
    outImage[int2(fragCoord)] = float4(finalColor, 1.0F);
  }

}
