#pragma once

#include "../shaderio.h"      // Shared definitions with CPU
#include "descriptors.slang"  // External memory definitions

const static int KERNEL_RADIUS = 2;
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
  if(!(dz == dz))
    return 0.0;
  int index = clamp(int(dz * 4.0), 0, 10);
  return depthKernel[index];
}

float applyBilateralBlurPass(
  int2 center_coor, bool horizontal, uint2 iResolution, int texelSize,
  Texture2D<float> horizImage, Texture2D<float> vertImage
){
  int safeTexelSize = max(texelSize, 1);
  int2 fullRes = int2(iResolution);

  if(any(fullRes <= 0))
    return 1.0;

  int2 centerFull = clamp(center_coor, int2(0), fullRes - int2(1));
  if(safeTexelSize <= 1){
    float sample = horizontal ? horizImage.Load(int3(centerFull.x, centerFull.y, 0))
                              : vertImage.Load(int3(centerFull.x, centerFull.y, 0));
    return sample == sample ? saturate(sample) : 1.0;
  }

  float sample_center;
  float depth_center;
  float result;
  float weightSum;

  if(horizontal){
    int2 lowRes = max((fullRes + safeTexelSize - 1) / safeTexelSize, int2(1));
    int2 centerLow = clamp(center_coor / safeTexelSize, int2(0), lowRes - int2(1));
    int2 centerDepth = min(centerLow * safeTexelSize + safeTexelSize / 2, fullRes - int2(1));

    sample_center = horizImage.Load(int3(centerLow.x, centerLow.y, 0));
    depth_center = depthImage[centerDepth];
    result = sample_center * kernel[0];
    weightSum = kernel[0];

    for(int i = 1; i <= KERNEL_RADIUS; i++){
      for(int s = -1; s <= 1; s += 2){
        int2 sampleLow = centerLow + int2(i * s, 0);

        if(any(sampleLow < 0) || any(sampleLow >= lowRes))
          continue;

        int2 depthCoor = min(sampleLow * safeTexelSize + safeTexelSize / 2, fullRes - int2(1));
        float sample = horizImage.Load(int3(sampleLow.x, sampleLow.y, 0));
        float depth = depthImage[depthCoor];
        float w = kernel[i] * depthWeight(depth_center, depth);

        result += sample * w;
        weightSum += w;
      }
    }
  }else{
    sample_center = vertImage.Load(int3(centerFull.x, centerFull.y, 0));
    depth_center = depthImage[centerFull];
    result = sample_center * kernel[0];
    weightSum = kernel[0];

    for(int i = 1; i <= KERNEL_RADIUS; i++){
      for(int s = -1; s <= 1; s += 2){
        int2 coor = centerFull + int2(0, safeTexelSize * i * s);

        if(any(coor < 0) || any(coor >= fullRes))
          continue;

        float sample = vertImage.Load(int3(coor.x, coor.y, 0));
        float depth = depthImage[coor];
        float w = kernel[i] * depthWeight(depth_center, depth);

        result += sample * w;
        weightSum += w;
      }
    }
  }

  float blurred = weightSum > 1e-8 ? result / weightSum : sample_center;
  return blurred == blurred ? saturate(blurred) : saturate(sample_center);
}
