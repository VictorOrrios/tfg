#include "shaderio.h"               // Shared definitions with CPU
#include "utils/common.slang"       // Shared definitions with shaders
#include "utils/palettes.slang"     // Debug color palette
#include "utils/descriptors.slang"  // External memory definitions
#include "utils/transforms.slang"   // World/View/Clip transforms
#include "utils/tracing.slang"      // Ray/Sphere tracing functions


[shader("compute")]
[numthreads(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID){
  float depth, shadow, shininess, alpha, metalness; float3 albedo, normal, position, debug; Material mat;

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
  depth = traceScene(r,normal,mat,debug);

  bool hit = depth >= 0.0;
  if(!hit){
    albedo = float3(0);
    shininess = 0.0;
    alpha = 0.0;
    metalness = 0.0;
    debug = r.dir;
    position = worldPoint2viewSpace(r.orig + r.dir);
  }else{
    albedo = mat.albedo_shininess.xyz;
    shininess = mat.albedo_shininess.w;
    alpha = mat.alpha_metalness.x;
    metalness = mat.alpha_metalness.y;
    position = worldPoint2viewSpace(r.orig + r.dir*depth);

    // If the pixel needs to compute shadow
    if(all(fragCoord % pushConst.lp.shadowTexelSize == pushConst.lp.shadowTexelSize/2)){
      if(pushConst.lp.shadowSamples == 0){
      shadow = 1.0;
      }else if(dot(normal,pushConst.lp.lightDir) > -0.3){
        const float k = 180.0;
        float3 p = r.orig + r.dir*depth;
        p += normal * 0.03;
        if(pushConst.lp.tracingMode == TracingModes::compute){
          shadow = traceShadow(p,k);
        }else{
          shadow = evalShadow(p,k);
        }
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

  albedoImage[int2(fragCoord)] = float4(albedo, alpha);
  normalImage[int2(fragCoord)] = float4(normal, metalness);
  depthImage[int2(fragCoord)] = depth;
  positionImage[int2(fragCoord)] = float4(position,0.0);
}
