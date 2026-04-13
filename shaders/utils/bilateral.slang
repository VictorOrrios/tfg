#pragma once

#include "../shaderio.h"      // Shared definitions with CPU
#include "descriptors.slang"  // External memory definitions

const static int KERNEL_RADIUS = 4;
const static float kernel[5] = float[](
  0.227027,
  0.1945946,
  0.1216216,
  0.054054,
  0.016216
);

const static float depthKernel[10] = float[](
  1.000000,
  0.606531,
  0.367879,
  0.223130,
  0.135335,
  0.082085,
  0.049787,
  0.030197,
  0.018316,
  0.011109
);


float gaussian(float x, float sigma){
  return exp(-(x * x) / (2.0 * sigma * sigma));
}

float depthWeight(float depth_center, float depth_sample){
  float dz = abs(depth_center - depth_sample);
  return depthKernel[int(dz*10.0)];
}

float applyBilateralBlurPass(int2 fragCoord, uint2 iResolution, bool horizontal){
  float sample_center = aoImage[fragCoord];
  float depth_center = depthImage[fragCoord];

  int2 texelOffset = (horizontal? int2(1,0) : int2(0,1)) * pushConst.lp.ssaoTexelSize;

  float result = sample_center * kernel[0];
  float weightSum = kernel[0];

  for (int i = 1; i <= KERNEL_RADIUS; i++) {
    for (int s = -1; s <= 1; s += 2) {

      int2 coor = fragCoord + texelOffset*i*s;

      if(any(coor < 0) || any(coor >= iResolution))
        break;

      coor -= coor%pushConst.lp.ssaoTexelSize;

      float sample = horizontal? aoImage[coor] : aoScratchImage[coor];
      float depth = depthImage[coor];
      float3 normal = normalImage[coor].xyz;

      float w = kernel[i] * depthWeight(depth_center, depth);

      result += sample * w;
      weightSum += w;
    }
  }

  return result / weightSum;
}