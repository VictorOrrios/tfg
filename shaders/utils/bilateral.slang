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

const static float depthKernel[11] = float[](
  1.000000,
  0.606531,
  0.367879,
  0.223130,
  0.135335,
  0.082085,
  0.049787,
  0.030197,
  0.018316,
  0.011109,
  0.0,
);


float gaussian(float x, float sigma){
  return exp(-(x * x) / (2.0 * sigma * sigma));
}

float depthWeight(float depth_center, float depth_sample){
  float dz = abs(depth_center - depth_sample);
  return depthKernel[int(dz*1.0)];
}

float applyBilateralBlurPass(
  int2 center_coor, bool horizontal, uint2 iResolution, int texelSize,
  Texture2D<float> horizImage, Texture2D<float> vertImage
){
  float2 uv_scale = horizontal? float2(1.0/texelSize) : float2(1.0,1.0);

  float2 center_uv = (float2(center_coor)/iResolution) * uv_scale;

  int2 coor_step = horizontal? int2(texelSize,0) : int2(0,texelSize);
  float2 uv_step = (horizontal? float2(1.0/float(iResolution.x),0.0) : float2(0.0,1.0/float(iResolution.y))) * uv_scale * texelSize;
  
  float sample_center = horizontal? horizImage.Sample(gSampler,center_uv) : vertImage.Sample(gSampler,center_uv);
  float depth_center = depthImage[center_coor];

  float result = sample_center * kernel[0];
  float weightSum = kernel[0];

  for (int i = 1; i <= KERNEL_RADIUS; i++) {
    for (int s = -1; s <= 1; s += 2) {

      int2 coor = center_coor + coor_step*i*s;

      if(any(coor < 0) || any(coor >= iResolution))
        continue;

      float2 uv = center_uv + uv_step*i*s;

      float sample = horizontal? horizImage.Sample(gSampler,uv) : vertImage.Sample(gSampler,uv);
      float depth = depthImage[coor];

      float w = kernel[i] * depthWeight(depth_center, depth);

      result += sample * w;
      weightSum += w;
    }
  }

  return result / weightSum;
}