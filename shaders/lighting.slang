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
[[vk::binding(BindingPoints::normalBuffer)]] Texture2D<float4> normalImage;
[[vk::binding(BindingPoints::albedoBuffer)]] Texture2D<float4> albedoImage;
[[vk::binding(BindingPoints::depthBuffer)]] Texture2D<float> depthImage;
[[vk::binding(BindingPoints::shadowBuffer)]] Texture2D<float> shadowImage;
[[vk::binding(BindingPoints::positionBuffer)]] Texture2D<float4> positionImage;
[[vk::binding(BindingPoints::noise)]] Texture2D<float4> noiseImage;
[[vk::binding(BindingPoints::randHemiVecs)]] ConstantBuffer<float3> randHemiVecs;
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;
[[vk::binding(BindingPoints::counters)]] StructuredBuffer<uint32_t> counters;

#ifdef NDEBUGSHADER
// TODO: Change to false for final release and boost performance
const static bool debugActive = true;
#else
const static bool debugActive = true;
#endif

static uint2 iResolution;

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

int2 worldToFragCoor(float3 worldPos){
  float4 viewPos = mul(float4(worldPos, 1.0), sceneInfo.viewMatrix);
  float4 clipPos = mul(viewPos, sceneInfo.projMatrix);
  float3 ndc = clipPos.xyz / clipPos.w;
  return int2(iResolution*(ndc.xy * 0.5 + 0.5));
}

float ssao(float2 fragCoord, float3 p, float depth, float3 N){
  float3 noise = float3(noiseImage[int2(fragCoord)%NOISE_TEX_SIZE].xy,0.0);
  float3 tangent = normalize(noise - N * dot(noise,N));
  float3 bitangent = cross(N, tangent);
  float3x3 TBN = float3x3(tangent, bitangent, N); 

  float occlusion = 0.0;
  for(int i = 0; i < NUM_RAN_HEMI_VECS; i++){
    float3 samplePos = mul(TBN,randHemiVecs[i]);
    samplePos = p + samplePos*pushConst.lp.ssaoRadius;
    int2 ssPos = worldToFragCoor(samplePos);
    float sampleTrueDepth = depthImage[ssPos];
/* 
    float4 offset = float4(samplePos,1.0);
    offset = mul(sceneInfo.projMatrix,offset);
    offset.xyz /= offset.w;
    offset.xyz = offset.xyz * 0.5 + 0.5;
    float sampleDepth = depthImage[int2(offset.xy)];
     */
    float sampleDepth = length(sceneInfo.cameraPosition.xyz-samplePos);
    occlusion += (sampleTrueDepth >= sampleDepth + pushConst.lp.ssaoBias ? 1.0 : 0.0);
  }

  occlusion = 1.0 - (occlusion / NUM_RAN_HEMI_VECS);
  return occlusion;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float2 fragCoord = threadIdx.xy;
  outImage.GetDimensions(iResolution.x, iResolution.y);
  float2 uv = fragCoord/iResolution;
  
  float3 N = normalImage[int2(fragCoord)].xyz;
  float3 albedo = albedoImage[int2(fragCoord)].xyz;
  float depth = depthImage[int2(fragCoord)];
  float shadow = shadowImage[int2(fragCoord)];
  float3 position = positionImage[int2(fragCoord)].xyz;
  float3 finalColor;

  if(depth >= 0.0){
    float3 V = getV(uv);
    float3 L = pushConst.lp.lightDir;
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
    float3 direct = (diffuse + specular) * shadow;
    float3 indirect = ambient + rimLight;

    float3 color = direct + indirect;

    finalColor = lerp(color, pushConst.lp.fogColor, fog);
  }else{
    float fog = 1 - exp(-100.0F * pushConst.lp.fogDensity);
    finalColor = float3(pushConst.lp.fogColor * fog);
  }

  if(debugActive){
    if(pushConst.debug.brickPercent && uv.x<=0.01){
      float percent = (counters[Counters::allocCounter] - counters[Counters::freeCounter]) / float(NUM_BRICKS_IN_ATLAS);
      if(percent > (1.0-uv.y)){
        outImage[int2(fragCoord)] = float4(1.0);
      }else{
        outImage[int2(fragCoord)] = float4(0.0);
      }
    }else if(uv.x > 0.98 && 
      (pushConst.debug.mode == 1 || pushConst.debug.mode == 4 || pushConst.debug.mode == 5 || pushConst.debug.mode == 7)){
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
        case 5: outImage[int2(fragCoord)] = float4(scalarToColor(shadow),1.0F); break;
        case 6: outImage[int2(fragCoord)] = float4(position,1.0F); break;
      }
    }
  }else{
    outImage[int2(fragCoord)] = float4(finalColor, 1.0F);
  }

  float3 noise = float3(noiseImage[int2(fragCoord)%NOISE_TEX_SIZE].xy,0.0);
  float3 tangent = normalize(noise - N * dot(noise,N));
  float3 bitangent = cross(N, tangent);
  float3x3 TBN = float3x3(tangent, bitangent, N); 

  int2 ssPos;

  float occlusion = 0.0;
  for(int i = 0; i < NUM_RAN_HEMI_VECS; i++){
    float3 samplePos = mul(TBN,randHemiVecs[i]);
    samplePos = position + samplePos*pushConst.lp.ssaoRadius;
    ssPos = worldToFragCoor(samplePos);
    float sampleTrueDepth = depthImage[ssPos];
/* 
    float4 offset = float4(samplePos,1.0);
    offset = mul(sceneInfo.projMatrix,offset);
    offset.xyz /= offset.w;
    offset.xyz = offset.xyz * 0.5 + 0.5;
    float sampleDepth = depthImage[int2(offset.xy)];
     */
    float sampleDepth = length(sceneInfo.cameraPosition.xyz-samplePos);
    occlusion += (sampleTrueDepth >= sampleDepth + pushConst.lp.ssaoBias ? 1.0 : 0.0);
  }

  occlusion = 1.0 - (occlusion / NUM_RAN_HEMI_VECS);

  //outImage[int2(fragCoord)] = float4(occlusion,occlusion,occlusion,1.0);
  outImage[int2(fragCoord)] = float4(float2(worldToFragCoor(position))/iResolution,0.0,1.0);
}
