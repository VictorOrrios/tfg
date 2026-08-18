
#include "shaderio.h"                 // Shared definitions with CPU
#include "utils/descriptors.slang"    // External memory definitions
#include "utils/transforms.slang"     // World/View/Clip transforms
#include "utils/grid.slang"           // Grid functions
#include "utils/sdf.slang"            // SDF functions, operations and primitives

static uint2 iResolution;

float ssao(float2 fragCoord, float3 N, float3 fragPos){
  float3 NView = worldVec2viewSpace(N);
  float3 noise = noiseImage[(int2(fragCoord))%NOISE_TEX_SIZE].xyz;
  float3 tangent = normalize(cross(noise,N));
  float3 bitangent = cross(NView, tangent);
  float3x3 TBN = float3x3(tangent, bitangent, NView);

  float occlusion = 0.0;
  for(int i = 0; i < pushConst.lp.aoSamples; i++){
    float3 randHemiVec = aoKernels[i];
    float3 sampleOffset = mul(randHemiVec, TBN);
    float3 samplePos = fragPos + sampleOffset;
    int2 sampleCoor = viewPoint2screenCoors(samplePos, iResolution);
    if(any(sampleCoor < 0)) continue;
    float sampleDepth = positionImage[sampleCoor].z;
    float rangeCheck = smoothstep(0.0, 1.0, pushConst.lp.aoRadius / abs(fragPos.z - sampleDepth));
    occlusion += (sampleDepth >= samplePos.z + pushConst.lp.aoBias ? 1.0 : 0.0) * rangeCheck;
  }

  return 1.0 - (occlusion / pushConst.lp.aoSamples);
}

float gridAo(float2 fragCoord, float3 N, float3 fragPos){
  float3 fragWorldPos = viewPoint2worldPoint(fragPos);
  float3 noise = noiseImage[(int2(fragCoord))%NOISE_TEX_SIZE].xyz;
  float3 tangent = normalize(cross(noise,N));
  // cross() degenerates when the noise vector is parallel to N; pick another axis.
  if(!(dot(tangent, tangent) > 0.0)){
    tangent = normalize(abs(N.y) < 0.9 ? cross(float3(0,1,0), N) : cross(float3(1,0,0), N));
  }
  float3 bitangent = normalize(cross(N, tangent));
  // Offset the sample origin off the surface by at least half a voxel, otherwise
  // samples near the surface read their own geometry as an occluder.
  float surfaceBias = max(pushConst.lp.aoBias, VOXEL_SIZES[getLevelOfPoint(fragWorldPos)] * 0.5);
  float3 aoOrigin = fragWorldPos + N * surfaceBias;

  float occlusion = 0.0;
  for(int i = 0; i < pushConst.lp.aoSamples; i++){
    float3 randHemiVec = aoKernels[i];
    float3 sampleOffset = tangent * randHemiVec.x + bitangent * randHemiVec.y + N * randHemiVec.z;
    float3 samplePos = aoOrigin + sampleOffset;
    float sdfValue = sampleInterpolatedPoint(samplePos);
    // Ramp by penetration depth rather than a binary test, so the result does not
    // collapse to fully occluded or fully open.
    float sampleDistance = max(length(sampleOffset), 1e-4);
    occlusion += saturate((-sdfValue - surfaceBias) / sampleDistance);
  }

  return 1.0 - (occlusion / pushConst.lp.aoSamples);
}

float evalAo(float2 fragCoord, float3 N, float3 fragPos){
  float3 fragWorldPos = viewPoint2worldPoint(fragPos);
  float3 noise = noiseImage[(int2(fragCoord))%NOISE_TEX_SIZE].xyz;
  float3 tangent = normalize(cross(noise,N));
  float3 bitangent = cross(N, tangent);
  float3x3 TBN = float3x3(tangent, bitangent, N);

  float occlusion = 0.0;
  for(int i = 0; i < pushConst.lp.aoSamples; i++){
    float3 randHemiVec = aoKernels[i];
    float3 sampleOffset = mul(randHemiVec, TBN);
    float3 samplePos = fragWorldPos + sampleOffset;
    float sdfValue = map(samplePos,0);
    occlusion += (sdfValue <= pushConst.lp.aoBias ? 1.0 : 0.0);
  }

  return 1.0 - (occlusion / pushConst.lp.aoSamples);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  int2 fragCoord = threadIdx.xy;
  outImage.GetDimensions(iResolution.x, iResolution.y);

  // fragCoord indexes the (possibly downsampled) AO image, so it must be bounds
  // checked against that resolution, not against the full res output image.
  int2 aoResolution = (int2(iResolution) + pushConst.lp.aoTexelSize - 1) / pushConst.lp.aoTexelSize;

  if(any(fragCoord >= aoResolution))
    return;

  int2 sampleCoord = min(fragCoord * pushConst.lp.aoTexelSize + (pushConst.lp.aoTexelSize/2), int2(iResolution) - 1);

  if(pushConst.lp.aoSamples == 0){
    aoImage[fragCoord] = 1.0;
    return;
  }

  float occlussion = 1.0;
  float depth = depthImage[int2(sampleCoord)];
  if(depth >= 0){

    float3 N = normalImage[int2(sampleCoord)].xyz;
    N = dot(N, N) > 1e-8 ? normalize(N) : float3(0.0, 1.0, 0.0);
    float3 fragPos = positionImage[int2(sampleCoord)].xyz;

    //occlussion = ssao(sampleCoord,N,fragPos);
    if(pushConst.lp.tracingMode == TracingModes::sphere){
      occlussion = evalAo(sampleCoord,N,fragPos);
    }else{
      //occlussion = ssao(sampleCoord,N,fragPos);
      occlussion = gridAo(sampleCoord,N,fragPos);
    }
  }

  aoImage[fragCoord] = saturate(occlussion);

}