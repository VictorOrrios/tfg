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
[[vk::binding(BindingPoints::randHemiVecs)]] StructuredBuffer<float3> randHemiVecs;
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

  float4 view = mul(rayClip,sceneInfo.projMatrixInv);
  float3 world = mul(float4(normalize(view.xyz),0.0),sceneInfo.viewMatrixInv).xyz;
  return normalize(world);
}

float3 scalarToColor(float x){
  return scalarToColor(x, pushConst.debug.palette);
}

int2 worldToFragCoor(float3 worldP){
  float4 viewPos = mul(float4(worldP, 1.0), sceneInfo.viewMatrix);
  float4 clipPos = mul(viewPos, sceneInfo.projMatrix);
  float3 ndc = clipPos.xyz / clipPos.w;
  float2 uv = saturate(ndc.xy * 0.5 + 0.5);
  return int2(uv * iResolution);
}

int2 viewPointToScreenCoors(float3 viewP){
  float4 clipPos = mul(float4(viewP,1.0), sceneInfo.projMatrix);
  float3 ndc = clipPos.xyz / clipPos.w;
  float2 uv = ndc.xy * 0.5 + 0.5;
  return int2(uv * iResolution);
}

float3 worldVecToViewSpace(float3 worldP){
  return mul(float4(worldP, 0.0),sceneInfo.viewMatrix).xyz;
}

float ssao(float2 fragCoord, float3 N, float3 fragPos){
  float3 NView = worldVecToViewSpace(N);
  float3 noise = float3(noiseImage[int2(fragCoord)%NOISE_TEX_SIZE].xy,0.0);
  float3 tangent = normalize(noise - NView * dot(noise,NView));
  float3 bitangent = cross(NView, tangent);
  float3x3 TBN = float3x3(tangent, bitangent, NView); 

  float3 samplePos;
  float occlusion = 0.0;
  for(int i = 0; i < NUM_RAN_HEMI_VECS; i++){
    float3 sampleOffset = mul(randHemiVecs[i], TBN) * pushConst.lp.ssaoRadius;
    samplePos = fragPos + sampleOffset;
    int2 sampleCoor = viewPointToScreenCoors(samplePos);
    float sampleDepth = positionImage[sampleCoor].z;
    float rangeCheck = smoothstep(0.0, 1.0, pushConst.lp.ssaoRadius / abs(fragPos.z - sampleDepth));
    occlusion += (sampleDepth >= samplePos.z + pushConst.lp.ssaoBias ? 1.0 : 0.0) * rangeCheck;
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
  float3 fragPos = positionImage[int2(fragCoord)].xyz;
  float3 finalColor;
  float occlussion = 0.0;

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

    // SSAO
    occlussion = ssao(fragCoord,N,fragPos);

    // Depth based fog
    float fog = 1 - exp(-depth * pushConst.lp.fogDensity);

    // Color calculation
    float3 direct = (diffuse + specular) * shadow;
    float3 indirect = (ambient + rimLight);

    float3 color = (direct + indirect) * occlussion;

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
      (pushConst.debug.mode == DebugModes::dmDebug || 
        pushConst.debug.mode == DebugModes::dmDepth || 
        pushConst.debug.mode == DebugModes::dmShadow || 
        DebugModes::dmBoundingBox)){
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
        case DebugModes::dmNone: 
        case DebugModes::dmBoundingBox:
          outImage[int2(fragCoord)] = float4(finalColor, 1.0F); break;
        case DebugModes::dmDebug: 
        case DebugModes::dmAlbedo: 
          outImage[int2(fragCoord)] = float4(albedo, 1.0F); break;
        case DebugModes::dmNormal: outImage[int2(fragCoord)] = float4(N, 1.0F); break;
        case DebugModes::dmDepth: outImage[int2(fragCoord)] = float4(scalarToColor(depth), 1.0F); break;
        case DebugModes::dmShadow: outImage[int2(fragCoord)] = float4(scalarToColor(shadow),1.0F); break;
        case DebugModes::dmPosition: outImage[int2(fragCoord)] = float4(fragPos*float3(1,1,-1 ),1.0F); break;
        case DebugModes::dmSSAO: outImage[int2(fragCoord)] = float4(float3(occlussion),1.0F); break;
      }
    }
  }else{
    outImage[int2(fragCoord)] = float4(finalColor, 1.0F);
  }
/* 
  float3 NView = worldVecToViewSpace(N);
  float3 noise = float3(noiseImage[int2(fragCoord)%NOISE_TEX_SIZE].xy,0.0);
  
  float3 altN = float3(1,0,0);
  if(abs(dot(altN,NView))>0.99){
    altN = float3(0,1,0);
  }
  float3 tangent = cross(altN,NView);
  
  //float3 tangent = normalize(noise - NView * dot(noise,NView));
  float3 bitangent = cross(NView, tangent);
  float3x3 TBN = float3x3(tangent, bitangent, NView); 

  float debug;
  float3 samplePos;
  float occlusion = 0.0;
  for(int i = 0; i < NUM_RAN_HEMI_VECS; i++){
    samplePos = mul(randHemiVecs[i],TBN);
    samplePos = mul(float3(0,0,1),TBN);
    samplePos = fragPos + samplePos*pushConst.lp.ssaoRadius;
    int2 sampleCoor = viewPointToScreenCoors(samplePos);
    float sampleDepth = positionImage[sampleCoor].z;
    float rangeCheck = smoothstep(0.0, 1.0, pushConst.lp.ssaoRadius / abs(fragPos.z - sampleDepth));
    occlusion += (sampleDepth >= samplePos.z + pushConst.lp.ssaoBias ? 1.0 : 0.0) * rangeCheck;
    debug = occlusion;
  }

  occlusion = 1.0 - (occlusion / NUM_RAN_HEMI_VECS);

  float3 final = occlusion;
  outImage[int2(fragCoord)] = float4(final,1.0);
 */

}
