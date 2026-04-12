#pragma once

#include "../shaderio.h"        // Shared definitions with CPU
#include "common.slang"         // Shared definitions with shaders
#include "descriptors.slang"    // External memory definitions

Ray get_ray(float2 uv){
  // Calculate offsets
  float2 ndc = uv * 2.0 - 1.0;
  float4 rayClip = float4(ndc.x, ndc.y, -1.0, 1.0);

  float4 view = mul(rayClip,sceneInfo.projMatrixInv);
  float3 world = mul(float4(normalize(view.xyz),0.0),sceneInfo.viewMatrixInv).xyz;
  float3 dir = normalize(world);

  Ray ray;
  ray.orig = sceneInfo.cameraPosition.xyz;
  ray.dir = dir;

  return ray;
}

float3 uv2worldDir(float2 uv){
  float2 ndc = uv * 2.0 - 1.0;
  float4 rayClip = float4(ndc.x, ndc.y, -1.0, 1.0);

  float4 view = mul(rayClip,sceneInfo.projMatrixInv);
  float3 world = mul(float4(normalize(view.xyz),0.0),sceneInfo.viewMatrixInv).xyz;
  return normalize(world);
}

int2 viewPoint2screenCoors(float3 viewP, uint2 iResolution){
  float4 clipPos = mul(float4(viewP,1.0), sceneInfo.projMatrix);
  float3 ndc = clipPos.xyz / clipPos.w;
  if (any(ndc.xy < -1.0) || any(ndc.xy > 1.0) || ndc.z < 0.0 || ndc.z > 1.0) 
    return int2(-1, -1);
  float2 uv =ndc.xy * 0.5 + 0.5;
  return int2(uv * iResolution);
}

float3 worldVec2viewSpace(float3 worldP){
  return mul(float4(worldP, 0.0),sceneInfo.viewMatrix).xyz;
}

float3 worldPoint2viewSpace(float3 worldP){
  return mul(float4(worldP, 1.0),sceneInfo.viewMatrix).xyz;
}