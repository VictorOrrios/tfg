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

#include "shaderio.h"                 // Shared definitions with CPU
#include "utils/palettes.slang"       // Debug color palette
#include "utils/descriptors.slang"    // External memory definitions
#include "utils/transforms.slang"     // World/View/Clip transforms

static uint2 iResolution;

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
  float occlussion = aoImage[int2(fragCoord)];
  float3 finalColor;

  if(depth >= 0.0){
    float3 V = uv2worldDir(uv);
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
    float3 indirect = (ambient + rimLight);

    float3 color = (direct + indirect) * occlussion;

    finalColor = lerp(color, pushConst.lp.fogColor, fog);
  }else{
    float fog = 1 - exp(-100.0F * pushConst.lp.fogDensity);
    finalColor = float3(pushConst.lp.fogColor * fog);
  }

  if(debugActive){
    if(pushConst.debug.brickPercent && uv.x<=0.01){
      float freeUv = fract(counters[Counters::freeCounter] / float(NUM_BRICKS_IN_ATLAS));
      float allocUv = fract(counters[Counters::allocCounter] / float(NUM_BRICKS_IN_ATLAS));
      float percent = (counters[Counters::allocCounter] -counters[Counters::freeCounter]) / float(NUM_BRICKS_IN_ATLAS);
      if(((uv.y <= allocUv && uv.y >= freeUv) && (allocUv > freeUv)) ||
         ((uv.y <= allocUv || uv.y >= freeUv) && (allocUv < freeUv))){
        outImage[int2(fragCoord)] = float4(scalarToColor(percent),1.0);
      }else{
        outImage[int2(fragCoord)] = float4(0,0,0,1);
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
  float3 final;

  final = occlusion;
  outImage[int2(fragCoord)] = float4(final,1.0);
 */
 
}
