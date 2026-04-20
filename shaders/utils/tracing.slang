#pragma once

#include "../shaderio.h"      // Shared definitions with CPU
#include "common.slang"       // Shared definitions with shaders
#include "descriptors.slang"  // External memory definitions
#include "grid.slang"         // Grid functions
#include "sdf.slang"          // SDF functions, operations and primitives
#include "palettes.slang"     // Debug color palette
#include "evalFuncs.slang"    // Scene evaluation functions

void fetchBrickValue(int pointer, int3 voxel_offset, float max_voxel_value, out float4 sBottom, out float4 sTop){
  int3 brickAtlasIndex = int3(pointer%BRICK_PER_ATLAS_AXIS,pointer/BRICK_PER_ATLAS_AXIS,0);
  int3 voxelIndex = brickAtlasIndex*BRICK_SIZE + voxel_offset;
  sBottom = float4(
    brickAtlas[voxelIndex + int3(0,0,0)],  // s000
    brickAtlas[voxelIndex + int3(1,0,0)],  // s100
    brickAtlas[voxelIndex + int3(1,0,1)],  // s101
    brickAtlas[voxelIndex + int3(0,0,1)]   // s001
  ) * max_voxel_value;

  sTop = float4(
    brickAtlas[voxelIndex + int3(0,1,0)],  // s010
    brickAtlas[voxelIndex + int3(1,1,0)],  // s110
    brickAtlas[voxelIndex + int3(1,1,1)],  // s111
    brickAtlas[voxelIndex + int3(0,1,1)]   // s011
  ) * max_voxel_value;
}


bool sphereTraceBrick(Ray r, int pointer, float3 brickPos, int level, out float depth){
  const int MAX_ITERATIONS = BRICK_SIZE * 5;
  depth = 0.0;

  const float brick_size = BRICK_SIZES[level];
  const float voxel_size = VOXEL_SIZES[level];
  const float max_voxel_value = MAX_VOXEL_VALUES[level];
  float3 p = r.orig;
  float3 diff = p - brickPos;
  float4 sBottom,sTop;
  int3 prev_voxel_offset = int3(-1);

  for(int step = 0; step<MAX_ITERATIONS; step++){
    if(any(diff<0.0) || any(diff>brick_size))
      break;
    
    float3 diffNorm = diff/voxel_size;
    int3 voxel_offset = int3(floor(diffNorm));
    float3 voxel_coor = frac(diffNorm);

    if(step == 0 || any(prev_voxel_offset != voxel_offset)){
      fetchBrickValue(pointer,voxel_offset,max_voxel_value,sBottom,sTop);
      prev_voxel_offset = voxel_offset;
    }

    float t = triLerp(voxel_coor,sBottom,sTop);

    if(t < voxel_size/10){
      depth = length(r.orig - p);
      return true;
    }

    p += r.dir*t;

    diff = p - brickPos;
  }

  return false;
}

bool sphereTraceBrickShadow(Ray r, int pointer, float3 brickPos, int level){
  const int MAX_ITERATIONS = BRICK_SIZE * 2;

  const float brick_size = BRICK_SIZES[level];
  const float voxel_size = VOXEL_SIZES[level];
  const float max_voxel_value = MAX_VOXEL_VALUES[level];
  float3 p = r.orig;
  float3 diff = p - brickPos;
  float4 sBottom,sTop;
  int3 prev_voxel_offset = int3(-1);

  for(int step = 0; step<MAX_ITERATIONS; step++){
    if(any(diff<0.0) || any(diff>brick_size))
      break;
    
    float3 diffNorm = diff/voxel_size;
    int3 voxel_offset = int3(floor(diffNorm));
    float3 voxel_coor = frac(diffNorm);

    if(step == 0 || any(prev_voxel_offset != voxel_offset)){
      fetchBrickValue(pointer,voxel_offset,max_voxel_value,sBottom,sTop);
      prev_voxel_offset = voxel_offset;
    }

    float t = triLerp(voxel_coor,sBottom,sTop);

    if(t < voxel_size/10){
      return true;
    }

    p += r.dir * max(t, voxel_size * 0.5f);

    diff = p - brickPos;
  }

  return false;
}

// Pre: r.orig must be inside bbox
float sphereTraceBbox(Ray r, Bbox bbox, out float3 debug){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS * int(CLIPMAP_LEVELS*0.5);

  // Debug visualizers
  const bool drawSilhouette   = debugColor && true;   // Draw the silhouette of the grid
  const bool drawSteps        = debugColor && false;  // Draw the numer of steps taken
  const bool drawLevel        = debugColor && true;  // Draw the clip map level at hit point
  const bool drawHitIndex     = debugColor && false;  // Draw the grid brick index at the hit point
  const bool drawBrickPointer = debugColor && false;   // Draw the pointer value of the brick at hit point

  debug = float3(0.0);
  float3 p = r.orig, dir = r.dir;
  float4 sBottom, sTop;
  int level;

  for(int step = 0; step < MAX_ITERATIONS; step++){
    level = getLevelOfPoint(p);

    const float brick_size = BRICK_SIZES[level];
    const float voxel_size = VOXEL_SIZES[level];

    // Scale normalization
    float3 pBrickNorm = p/brick_size;
    // Truncate normalized coordinates to get the id
    int3 brickId = int3(floor(pBrickNorm));
    // Get coordinates inside brick
    float3 brickCoor = frac(pBrickNorm);

    // Get de voxel params
    float3 relVoxelCoor = brickCoor * (BRICK_SIZE-1);
    int3 voxelOffset = int3(relVoxelCoor); // No floor needed, brickCoor ∈ [0,1]
    float3 voxelCoor = fract(relVoxelCoor);

    // Get pointer of allocated brick
    int3 brickIndex = id2index(brickId);
    int3 clipIndex = brickIdx2clipIdx(brickIndex, level);
    int clipPointer = clipMap[clipIndex] & NOT_DIRTY_BIT;

    // Update current voxel data and index
    fetchBrickValue(clipPointer,voxelOffset,level,sBottom,sTop);
    // Interpolate current sdf value of the voxel2
    float t = triLerp(voxelCoor,sBottom,sTop);

    // Cutoff condition: A distance of 1/10 voxels
    // TODO: Fine tune to better performance/quality
    if(t < voxel_size/10){

      // Debug visualizer of step count
      if(drawSteps) debug = scalarToColor(float(step)/float(MAX_ITERATIONS));

      // Level indicator
      if(drawLevel) debug = scalarToColor(float(level)/float(CLIPMAP_LEVELS-1));

      // Debug visualizer of hit brick index
      if(drawHitIndex) debug = float3(brickIndex)/float3(NUM_BRICKS_PER_AXIS);

      // Brick pointer visualizer
      if(drawBrickPointer) debug = scalarToColor(float(clipPointer)/float(NUM_BRICKS_IN_ATLAS));
      
      // Returns depth of hit
      return length(p - r.orig);
    }
    
    // Update p to move the estimated length to surface
    p += r.dir*t;
  
    // Check if ray has left the bbox
    if(!insideBbox(p,bbox)){
      // Outside grid
      if(drawSteps){
        debug = scalarToColor(float(step)/float(MAX_ITERATIONS));
        return 1;
      }else{
        // No hit
        return -1;
      }
    }
  }

  // Over iterated 
  if(drawSteps){
    // Debug visualizer of step count
    debug = scalarToColor(2.0);
    return 1;
  }else{
    // No hit
    return -1;
  }
}

// Pre: r.orig must be inside bbox
void sphereTraceShadowBbox(Ray r, Bbox bbox, float k, float depthOffset, float minShadow, inout float shadow){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS * int(CLIPMAP_LEVELS*0.5);

  float3 p = r.orig, dir = r.dir;
  float4 sBottom, sTop;
  int level;

  float depth = depthOffset;

  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    level = getLevelOfPoint(p);

    const float brick_size = BRICK_SIZES[level];
    const float voxel_size = VOXEL_SIZES[level];

    // Scale normalization
    float3 pBrickNorm = p/brick_size;
    // Truncate normalized coordinates to get the id
    int3 brickId = int3(floor(pBrickNorm));
    // Get coordinates inside brick
    float3 brickCoor = frac(pBrickNorm);

    // Get de voxel params
    float3 relVoxelCoor = brickCoor * (BRICK_SIZE-1);
    int3 voxelOffset = int3(relVoxelCoor); // No floor needed, brickCoor ∈ [0,1]
    float3 voxelCoor = fract(relVoxelCoor);

    // Get pointer of allocated brick
    int3 brickIndex = id2index(brickId);
    int3 clipIndex = brickIdx2clipIdx(brickIndex, level);
    int clipPointer = clipMap[clipIndex] & NOT_DIRTY_BIT;

    // Update current voxel data and index
    fetchBrickValue(clipPointer,voxelOffset,level,sBottom,sTop);
    // Interpolate current sdf value of the voxel2
    float t = triLerp(voxelCoor,sBottom,sTop);

    // Update shadow
    if(t < MAX_VOXEL_VALUES[0]){
      shadow = min(shadow,k*t/depth);
    }

    // Update p to move the estimated length to surface
    depth += t;
    p += r.dir*t;

    // Check if ray has left the bbox or shadow is too dark
    if(shadow <= minShadow || !insideBbox(p,bbox)){
      return;
    }
  }
}

float traceShadow(float3 p, float k){
  const int3 camIdN = sceneInfo.cameraId0.xyz >> (CLIPMAP_LEVELS-1);
  const float3 camIdNPos = camIdN * BRICK_SIZES[CLIPMAP_LEVELS-1];
  const float3 worldMin = camIdNPos - AXIS_SIZES[CLIPMAP_LEVELS-1]/2;
  const float3 worldMax = camIdNPos + AXIS_SIZES[CLIPMAP_LEVELS-1]/2;
  const float bboxPadding = MAX_VOXEL_VALUES[0];

  const float minShadow = 0.001;

  float shadow = 1.0f;

  Ray r;
  r.orig = p;
  r.dir = pushConst.lp.lightDir;

  for(int i = 0; i<pushConst.numObjects; i++){
    Bbox bbox = aabbs[i];

    if(any(bbox.bMin >= bbox.bMax))
      continue;

    bbox.bMin -= bboxPadding;
    bbox.bMax += bboxPadding;
    bbox.bMin = max(bbox.bMin,worldMin);
    bbox.bMax = min(bbox.bMax,worldMax);

    Ray bboxRay = r;
    float3 far, dir = r.dir;
    if(!intersectCube(r, bbox, bboxRay.orig, far)) continue;
    float orig2P = length(bboxRay.orig - r.orig);

    sphereTraceShadowBbox(bboxRay,bbox,k,orig2P,minShadow,shadow);

    if(shadow <= minShadow)
      break;
  }

  return shadow;
}

float traceScene(Ray r, out float3 normal, out Material mat, out float3 debug){
  const int3 camIdN = sceneInfo.cameraId0.xyz >> (CLIPMAP_LEVELS-1);
  const float3 camIdNPos = camIdN * BRICK_SIZES[CLIPMAP_LEVELS-1];
  const float3 worldMin = camIdNPos - AXIS_SIZES[CLIPMAP_LEVELS-1]/2;
  const float3 worldMax = camIdNPos + AXIS_SIZES[CLIPMAP_LEVELS-1]/2;

  debug = float3(0.0);
  float depth = -1.0f;
  normal = float3(0.0);
  float3 lastDebug = debug;

  for(int i = 0; i<pushConst.numObjects; i++){
    Bbox bbox = aabbs[i];

    if(any(bbox.bMin >= bbox.bMax))
      continue;

    bbox.bMin = max(bbox.bMin,worldMin);
    bbox.bMax = min(bbox.bMax,worldMax);

    // Peprare the ray that will trace the object
    Ray bboxRay = r;
    float3 far, dir = r.dir;
    if(!intersectCube(r, bbox, bboxRay.orig, far)) continue;

    // Cull if bbox is too far
    float orig2P = length(bboxRay.orig - r.orig);
    if(depth>0 && depth<orig2P)
      continue;

    // Trace bbox using the method defined by user
    float objDepth;
    objDepth = sphereTraceBbox(bboxRay,bbox,debug);
    //objDepth = evalTraceMap(bboxRay,length(far - bboxRay.orig));

    // Update min depth
    if(objDepth >= 0){
      objDepth += orig2P;
      if(depth < 0 || (objDepth>=0 && objDepth<depth)){
        depth = objDepth;
        lastDebug = debug;
      }
    }
  }

  // Use the nearest debug value that counted as the hit
  debug = lastDebug;

  // Calc normal using user defined method
  if(depth >= 0){
    float3 p = r.orig + r.dir*depth;
    
    mat = evalMat(p);
    //mat = sampleInterpolatedMat(p);


    normal = analyticNormal(p);
    //normal = evalNormal(p);
  }

  return depth;
}

// PRE: r.dir must be normalized
float rayMarchAmaWoo(Ray r, Bbox bbox, out float3 normal, out float3 debug){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS*3;

  // Debug visualizers
  const bool drawSilhouette = debugColor && true;   // Draw the silhouette of the grid 
  const bool drawSteps      = debugColor && true;   // Draw the numer of steps taken
  const bool drawHitIndex   = debugColor && false;  // Draw the grid voxel index at the hit point
  const bool drawFirstStep  = debugColor && false;  // Draw the axis of the step taken by the first iteration

  normal = float3(0.0);
  debug = float3(0.0);

  // Initial boundig box hit test
  float3 orig, far, dir = r.dir;
  bool hit = intersectCube(r, bbox, orig, far);
  if(!hit) return -1;

  // Silhouette highlighter
  if(drawSilhouette && length(far-orig)<0.02){
    debug = scalarToColor(-1);
    return 1;
  }

  const int numVoxelsPerAxis = NUM_BRICKS_PER_AXIS;       // How many voxels are per axis the grid
  const float3 gridSize = bbox.bMax - bbox.bMin;          // World space grid size
  const int3 step = sign(dir);                            // How many indices to jump per axis traversal
  const float3 voxelDensity = numVoxelsPerAxis/gridSize;  // How many voxels per unit of world space
  const float3 voxelSize = gridSize/numVoxelsPerAxis;     // Size of a voxel
  const float3 delta = voxelSize/abs(dir);                // How much to travel in each direction till we meet the next voxel of that axis
  
  float3 hitCoorPos = (orig-bbox.bMin)*voxelDensity;
  float3 hitCoorNeg = max(bbox.bMax-orig,float3(0))*voxelDensity;
  float3 hitCoor = lerp(hitCoorNeg,hitCoorPos,step*0.5+0.5);

  int3 origIndex = min(int3(numVoxelsPerAxis-1),max(int3(0),int3(
    trunc(hitCoorPos)
  )));
  int3 index = origIndex;
  float3 dOffset = 1.0-fract(hitCoor);
  float3 iniDelta = delta * dOffset;
  float3 maxD = iniDelta;
  float3 lastTraversal;

  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    // Check current voxel for hit
    int pointer = clipMap[index];
    if(pointer < NUM_BRICKS_IN_ATLAS){
      // Debug visualizer of step count
      //if(drawSteps) debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
      if(drawSteps){
        if((pointer & DIRTY_BIT) != 0){
          debug = float3(1,1,0);
        }else{
          debug = scalarToColor(float(pointer)/float(NUM_BRICKS_IN_ATLAS*0.2));
        }
      }

      // Calculate normal based on last step
      normal = -step*lastTraversal;

      // Distance = size of step * numer of steps
      float3 deltas = voxelSize*abs(index-origIndex);
      return length(deltas*step + orig - r.orig);
    }
    
    if (maxD.x < maxD.y && maxD.x < maxD.z) {
      // X-axis traversal.
      maxD.x += delta.x;
      index.x += step.x;
      lastTraversal = float3(1,0,0);
    } else if (maxD.y < maxD.z) {
      // Y-axis traversal.
      maxD.y += delta.y;
      index.y += step.y;
      lastTraversal = float3(0,1,0);
    } else {
      // Z-axis traversal.
      maxD.z += delta.z;
      index.z += step.z;
      lastTraversal = float3(0,0,1);
    }

    // Debug visualizer of first step
    if(drawFirstStep){
      debug = lastTraversal;
      return 1;
    }

    if (any(index < 0) || any(index >= NUM_VOXELS_PER_AXIS+1)){
      // Outside grid
      if(drawSteps){
        debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
        return 1;
      }else{
        // No hit
        return -1;
      }
    }
  };

  // Over iterated 
  if(drawSteps){
    // Debug visualizer of step count
    debug = scalarToColor(1.1);
    return 1;
  }else{
    // No hit
    return -1;
  }
}