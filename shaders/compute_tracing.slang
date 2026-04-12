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
  depth = traceScene(r,normal,albedo,shadow,debug);

  bool hit = depth >= 0.0;
  if(!hit){
    albedo = float3(0);
    debug = r.dir;
    position = float3(0.0,0.0,-1000000.0);
  }else{
    position = worldPoint2viewSpace(r.orig + r.dir*depth);
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
  shadowImage[int2(fragCoord)] = shadow;
  positionImage[int2(fragCoord)] = float4(position,0.0);

  if(all(threadIdx == 0) && false){
    const int W = 64;
    for(int i = -5; i<20; i++){
      int3 camId0 = sceneInfo.cameraId0.xyz;
      float3 p = float3(2.000927,-1.688187,1.994390);
      p.x += VOXEL_SIZES[0]*i;
      int3 pId0 = int3(floor(p / BRICK_SIZES[0]));

      
      int3 diff = pId0 - camId0;
      diff += max(0,sign(diff));
      diff = abs(diff);
      int maxDiff = max(diff.x,max(diff.y,diff.z));
      
      const int HALF = W/2;
      const float GRID_OFFSET = 2.0 / W;
      float logVal = log2((float)(maxDiff) * GRID_OFFSET);
      int level = int(ceil(max(0,logVal)));

      level = clamp(level,0,CLIPMAP_LEVELS - 1);
      int3 camId = camId0 >> level;
      int3 pId = pId0 >> level;
      int3 rel_id = pId - camId + HALF;
      if (all(rel_id >= 0) && all(rel_id < W)) {
      }else{
        //level = min(level+1,CLIPMAP_LEVELS - 1);
      }
      printf("iteration %i | p %f,%f,%f | pId0 %i,%i,%i | pId %i,%i,%i | camId %i,%i,%i | rel_id %i,%i,%i | level %i\n",
        i,
        p.x,p.y,p.z,
        pId0.x,pId0.y,pId0.z,
        pId.x,pId.y,pId.z,
        camId.x,camId.y,camId.z,
        rel_id.x,rel_id.y,rel_id.z,
        level);
    }
  }
}
