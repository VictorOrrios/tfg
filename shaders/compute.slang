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

[[vk::binding(BindingPoints::gBuffers)]] RWTexture2D<float4> outImage;
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;

struct Ray {
  float3 orig;
  float3 dir;
};

float sdSphere(float3 p, float s){
    return length(p) - s;
}

float sdf_global(float3 p){
  return sdSphere(p, 0.5);
}

float rayMarch(Ray ray)
{
    const float MAX_DIST = 100.0;
    const float MIN_DIST = 0.001;
    
    float depth = 0.0;
    
    for(int i = 0; i < 100; i++){
        float3 p = ray.orig + ray.dir * depth;
        float dist = sdf_global(p);
        
        if(abs(dist) < MIN_DIST){
            return depth;  // Hit
        }
        
        if(depth > MAX_DIST){
            return -1.0;  // No hit
        }
        
        depth += dist;
    }
    
    return -1.0;
}

float3 cast(Ray r){
  float d = rayMarch(r);
  if(d >= 0.0){
    return float3(1.0,0.0,0.0);
  }else{
    return r.dir;
  }
}


Ray get_ray(float2 uv){
  // Calculate offsets
  float2 ndc = uv * 2.0 - 1.0;
  float4 rayClip = float4(ndc.x, ndc.y, -1.0, 1.0);

  float4 view = mul(rayClip,sceneInfo.projMatrix);
  float3 world = mul(float4(normalize(view.xyz),0.0),sceneInfo.viewMatrix).xyz;
  float3 dir = normalize(world);

  Ray ray;
  ray.orig = sceneInfo.cameraPosition;
  ray.dir = dir;

  return ray;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float2 fragCoord = threadIdx.xy;
  uint2 iResolution;
  outImage.GetDimensions(iResolution.x, iResolution.y);
  float2 uv = fragCoord/iResolution;
  Ray r = get_ray(uv);
  float3 finalColor = cast(r);
  //finalColor = r.dir;
  //finalColor = float3(uv,0.0);

  outImage[int2(fragCoord)] = float4(finalColor, 1.0F);
}
