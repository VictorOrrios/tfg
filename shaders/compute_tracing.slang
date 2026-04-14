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

#include "shaderio.h"               // Shared definitions with CPU
#include "utils/common.slang"       // Shared definitions with shaders
#include "utils/palettes.slang"     // Debug color palette
#include "utils/descriptors.slang"  // External memory definitions
#include "utils/transforms.slang"   // World/View/Clip transforms
#include "utils/tracing.slang"      // Ray/Sphere tracing functions




[shader("compute")]
[numthreads(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID){
  float depth, shadow; float3 albedo, normal, position, debug;

  float2 fragCoord = threadIdx.xy;
  uint2 iResolution;
  normalImage.GetDimensions(iResolution.x, iResolution.y);
  float2 uv = fragCoord/iResolution;
  
  if(any(fragCoord >= iResolution))
    return;

  Ray r = get_ray(uv);

  float3 p1, p2;
  float3 hitPoint;
  // Scene bounding box
  float bboxSize = AXIS_SIZES[CLIPMAP_LEVELS-1];
  float3 bboxPos = float3(0.0,0.0,0.0);
  Bbox bbox = unitBBox;
  bbox.bMin = bbox.bMin*bboxSize+bboxPos;
  bbox.bMax = bbox.bMax*bboxSize+bboxPos;
  
  //depth = rayMarchAmaWoo(r,bbox,normal);
  depth = traceScene(r,normal,albedo,debug);

  bool hit = depth >= 0.0;
  if(!hit){
    albedo = float3(0);
    debug = r.dir;
    position = float3(0.0,0.0,-1000000.0);
  }else{
    position = worldPoint2viewSpace(r.orig + r.dir*depth);

    if(all(fragCoord % pushConst.lp.shadowTexelSize == pushConst.lp.shadowTexelSize/2)){
      if(dot(normal,pushConst.lp.lightDir) > -0.3){
        const float k = 180.0;
        float3 p = r.orig + r.dir*depth;
        p += normal * 0.03;
        //shadow = evalShadow(p,k);
        shadow = traceShadow(p,k);
      }else{
        shadow = 0.0;
      }
      shadowImage[int2(fragCoord/pushConst.lp.shadowTexelSize)] = shadow;
    }
  }

  if(debugColor){
    albedo = debug;
  }else if(bboxColor){
    for(int i = 0; i<pushConst.numObjects; i++){
      const float eps = 0.05;
      float3 near, far, dir = r.dir;
      Bbox bboxObj = aabbs[i];
      bool hit = intersectCube(r, bboxObj, near, far);
      if(!hit) continue;
      float nearD = length(near-r.orig);
      if((depth < 0 || depth+eps > nearD) && length(near-far) < eps){
      //if((depth < 0 || depth+eps > nearD)){
        albedo = scalarToColor(float(i)/float(pushConst.numObjects));
        depth = nearD;
      }
    }
  }

  albedoImage[int2(fragCoord)] = float4(albedo,1.0F);
  normalImage[int2(fragCoord)] = float4(normal, 0.0F);
  depthImage[int2(fragCoord)] = depth;
  positionImage[int2(fragCoord)] = float4(position,0.0);
}
