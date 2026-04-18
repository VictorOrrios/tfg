#pragma once

#include "../shaderio.h"      // Shared definitions with CPU

Material lerpMaterial(Material matA, Material matB, float b){
  matB.albedo_shininess = lerp(matA.albedo_shininess,matB.albedo_shininess,b);
  matB.alpha_metalness = lerp(matA.alpha_metalness,matB.alpha_metalness,b);
  return matB;
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