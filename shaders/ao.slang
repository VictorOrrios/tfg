
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
  float3 bitangent = cross(N, tangent);
  float3x3 TBN = float3x3(tangent, bitangent, N);

  float occlusion = 0.0;
  for(int i = 0; i < pushConst.lp.aoSamples; i++){
    float3 randHemiVec = aoKernels[i];
    float3 sampleOffset = mul(randHemiVec, TBN);
    float3 samplePos = fragWorldPos + sampleOffset;
    float sdfValue = sampleInterpolatedPoint(samplePos);
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

  int2 sampleCoord = fragCoord * pushConst.lp.aoTexelSize + (pushConst.lp.aoTexelSize/2);

  if(any(fragCoord >= iResolution))
    return;

  float occlussion = 1.0;
  float depth = depthImage[int2(sampleCoord)];
  if(depth >= 0){

    float3 N = normalImage[int2(sampleCoord)].xyz;
    float3 fragPos = positionImage[int2(sampleCoord)].xyz;

    //occlussion = ssao(sampleCoord,N,fragPos);
    occlussion = gridAo(sampleCoord,N,fragPos);
  }

  aoImage[fragCoord] = occlussion;

}