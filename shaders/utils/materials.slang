#pragma once

#include "../shaderio.h"      // Shared definitions with CPU
#include "rng.slang"          // Random numbers and hashes
#include "descriptors.slang"  // External memory definitions

Material lerpMaterial(Material matA, Material matB, float b){
  matB.albedo_shininess = lerp(matA.albedo_shininess,matB.albedo_shininess,b);
  matB.alpha_metalness = lerp(matA.alpha_metalness,matB.alpha_metalness,b);
  matB.type = min(matA.type,matB.type);
  return matB;
}

float3 getTerrainColor(float3 p, float slope) {
  const float3 grass1 = float3(0.15, 0.3, 0.1);
  const float3 grass2 = float3(0.4, 0.5, 0.2);
  const float3 dirt   = float3(0.6, 0.5, 0.4);
  const float3 snow   = float3(1.0, 1.0, 1.0);
  const float3 rock   = float3(0.22, 0.2, 0.2);
  const float3 sand   = float3(0.8, 0.7, 0.6);
  //const float3 sand   = float3(0.1);
  const float3 water  = float3(0.122, 0.451, 0.788);
  //const float3 water  = float3(0.0);
  const float3 foam   = float3(1.0, 1.0, 1.0);

  float height = p.y;
  float3 random = noiseImage.Sample(gSampler,p.xz/1000.0).xyz;
  float3 random2 = noiseImage.Sample(gSampler,p.xz/10000.0).xyz;
  float breakup01 = abs(length(random)/1.73);
  float breakup11 = breakup01*2.0-1.0;
  float breakupbig = abs(length(random2)/1.73);


  float3 grassColor = lerp(
      grass1,
      grass2,
      smoothstep(0.2, 0.8, breakup01+height/3.0)
  );
  float3 color;
  const float snow_height_offset = 0.0;
  const float snow_diff = 10.0;
  float snow_percent = smoothstep(snow_height_offset+breakupbig*snow_diff, snow_diff+snow_height_offset, height);
  float3 grass_snow = lerp(grassColor, snow, snow_percent);
  float3 water_color = water*lerp(0.5,1.0,breakupbig);
  float3 grass_snow_rock = lerp(grass_snow, rock, smoothstep(0.3, 0.4, slope+breakup11/10.0));
  color = lerp(water_color, sand, smoothstep(0.0, 0.15, height+breakup11/8.0));
  color = lerp(color, grass_snow_rock, smoothstep(0.15, 0.2, height+breakup11/7.0));


  return color;

}

void updateTopK(uint id, float score, inout uint4 ids, inout float4 scores){
  int replaceIdx = -1;
  float worstScore = score;
  for(int i = 0; i < 4; i++){
    if(ids[i] == id){
      scores[i] = min(scores[i],score);
      replaceIdx = -2;
      break;
    }
    
    if(scores[i] > worstScore){
      worstScore = scores[i];
      replaceIdx = i;
    }
  }

  if(replaceIdx >= 0){
    ids[replaceIdx] = id;
    scores[replaceIdx] = score;
  }
}

float4 scores2Weight(uint4 ids, float4 scores){
  float4 weights = float4(0.0);
  float sum = 0.0;
  float epsilon = 0.0001;
  float minScore = min(0,min(scores.x,min(scores.y,min(scores.z,scores.w))));
  
  [unroll]
  for(int i = 0; i < 4; i++){
    if(ids[i] >= 0){
      float score = scores[i]-minScore;
      weights[i] = rcp(score * score + epsilon);
      sum += weights[i];
    }
  }
  
  if(sum > 0.0)
    weights *= rcp(sum);
  
  return weights;
}