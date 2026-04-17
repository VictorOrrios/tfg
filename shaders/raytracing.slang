#include "shaderio.h"                 // Shared definitions with CPU
#include "utils/common.slang"         // Shared definitions with shaders
#include "utils/bbox.slang"           // AABB helpers
#include "utils/palettes.slang"       // Debug color palette
#include "utils/descriptors.slang"    // External memory definitions
#include "utils/transforms.slang"     // World/View/Clip transforms
#include "utils/tracing.slang"        // Ray/Sphere tracing functions
#include "utils/evalFuncs.slang"      // Scene evaluation functions
#include "nvshaders/constants.h.slang"


struct HitPayload{
  float3 debug;
  float3 normal;
  float depth;
  Material mat;
  int level;
};

struct ShadowPayload{
  bool occluded;
};

struct IntersectionPayload{
  float3 debug;
  float depth;
  int level;
};


[shader("raygeneration")]
void rgenMain(){
  float depth, shadow = 1.0, shininess; float3 albedo, normal, debug, position;

  // Get the current pixel coordinates and image dimensions
  float2 launchID = (float2)DispatchRaysIndex().xy;        // Current pixel (x,y)
  float2 launchSize = (float2)DispatchRaysDimensions().xy; // Image size (width,height)
  float2 uv = launchID/launchSize;

  const uint rayFlags = 0;
  const uint shadowRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

  Ray r = get_ray(uv);

  // Create the primary ray
  RayDesc ray;
  // Transform camera origin (0,0,0) from view space to world space
  ray.Origin = r.orig;
  // Transform ray direction from view space to world space
  ray.Direction = r.dir;
  ray.TMin = VOXEL_SIZES[0]/100;  // Minimum distance to avoid self-intersection
  ray.TMax = INFINITE;            // Maximum distance (infinite for primary rays)

  // Initialize ray payload with default values
  HitPayload payload;
  payload.mat = Material(float4(0));
  payload.debug = float3(0);
  payload.normal = float3(1);
  payload.depth = -1;
  

  // Cast the ray into the scene using the acceleration structure
  // Parameters: AS, flags, instance mask, sbt offset, sbt stride, miss offset, ray, payload
  TraceRay(topLevelAS, rayFlags, 0xff, 0,0,0, ray, payload);
  
  albedo = payload.mat.albedo_shininess.xyz;
  shininess = payload.mat.albedo_shininess.w;
  debug = payload.debug;
  normal = payload.normal;
  depth = payload.depth;

  bool hit = depth >= 0.0;
  if(!hit){
    albedo = float3(0);
    debug = r.dir;
    position = float3(0.0,0.0,-1000000.0);
  }else{
    position = worldPoint2viewSpace(r.orig + r.dir*depth);
  }

  if(debugColor || bboxColor){
    albedo = debug;
  }

  // If  its a shadow pixel sample
  if(all(launchID % pushConst.lp.shadowTexelSize == pushConst.lp.shadowTexelSize/2)){
    // If there was a hit and its facing the light => Cast shadow ray
    if( hit && dot(normal,pushConst.lp.lightDir) > -0.3 ){
      ShadowPayload shadowPayload;
      ray.Origin = r.orig + r.dir*depth + normal * 0.05;
      float occlussion = 0.0;


      for(int i = 0; i<pushConst.lp.shadowSamples; i++){
        float3 randUnitVec = shadowKernels[i];
        ray.Direction = randUnitVec;
        shadowPayload.occluded = false;
        TraceRay(topLevelAS, shadowRayFlags, 0xff, 1, 0, 1, ray, shadowPayload);
        if(shadowPayload.occluded)
          occlussion++;
      }
 
      
/* 
      for(int i = 0; i<pushConst.lp.shadowRays; i++){
        float3 noise = noiseImage[(int2(launchID)+i)%NOISE_TEX_SIZE].xyz;
        float3 dir = normalize(pushConst.lp.lightDir*pushConst.lp.shadowSharpness + noise);
        ray.Direction = dir;
        shadowPayload.occluded = false;
        TraceRay(topLevelAS, shadowRayFlags, 0xff, 1, 0, 1, ray, shadowPayload);
        if(shadowPayload.occluded)
          occlussion++;
      }
       */

      shadow = 1.0 - occlussion/pushConst.lp.shadowSamples;
    }
    shadowImage[int2(launchID/pushConst.lp.shadowTexelSize)] = shadow;
  }

  normalImage[int2(launchID)] = float4(normal, 0.0F);
  albedoImage[int2(launchID)] = float4(albedo,shininess/MAX_SHININESS);
  depthImage[int2(launchID)] = depth;
  positionImage[int2(launchID)] = float4(position,0.0);
}

// Intersection shader - called when ray gets inside a brick
[shader("intersection")]
void rintMain(){
  const float eps = 1e-4;

  IntersectionPayload payload;
  Ray r; 
  r.orig = WorldRayOrigin();
  r.dir  = WorldRayDirection();

  uint instanceID = InstanceIndex();

  float3x4 objToWorld = ObjectToWorld();
  float3 pos = float3(objToWorld[0][3],objToWorld[1][3],objToWorld[2][3]);

  float size = objToWorld[0][0];
  int level = InstanceID();
  Bbox bbox = Bbox(pos,pos+size);

  float3 nearP, farP;
  if(intersectCube(r,bbox,nearP,farP)){
    nearP += r.dir * eps; // floating point error correction
    Ray brickRay = Ray(nearP,r.dir);
    float depthToBbox = length(r.orig-nearP);
    payload.depth = depthToBbox;

    if(sphereTraceBrick(brickRay,instanceID,pos,level,payload.depth)){
      payload.depth += depthToBbox; 
      //payload.debug = scalarToColor(float(instanceID)/float(NUM_BRICKS_IN_ATLAS));
      payload.debug = scalarToColor(float(level)/float(CLIPMAP_LEVELS-1));
      payload.level = level;
      ReportHit(payload.depth, 0, payload);
    }
  }
}

[shader("intersection")]
void rintShadow() {
  const float eps = VOXEL_SIZES[0]/2.0;

  ShadowPayload payload;
  Ray r; 
  r.orig = WorldRayOrigin();
  r.dir  = WorldRayDirection();

  uint instanceID = InstanceIndex();

  float3x4 objToWorld = ObjectToWorld();
  float3 pos = float3(objToWorld[0][3],objToWorld[1][3],objToWorld[2][3]);

  float size = objToWorld[0][0];
  int level = InstanceID();
  Bbox bbox = Bbox(pos,pos+size);

  float3 nearP, farP;
  if(intersectCube(r,bbox,nearP,farP)){
    nearP += r.dir * eps; // floating point error correction
    Ray brickRay = Ray(nearP,r.dir);

    if(sphereTraceBrickShadow(brickRay,instanceID,pos,level)){
      payload.occluded = true;
      ReportHit(1.0, 0, payload);
    }
  }
}

// Closest hit shader - called when ray hits geometry
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in IntersectionPayload attr){
  payload.debug = attr.debug;
  payload.depth = attr.depth;
  payload.level = attr.level;
  Ray r; 
  r.orig = WorldRayOrigin();
  r.dir  = WorldRayDirection();
  float3 p = r.orig + r.dir*payload.depth;
  payload.mat = evalMat(p);
  //payload.mat = sampleInterpolatedMat(p);

  payload.normal = analyticNormal(p);
  //payload.normal = evalNormal(p);
}

[shader("closesthit")]
void rchitShadow(inout ShadowPayload payload, in ShadowPayload attr){
  payload.occluded = attr.occluded;
}

// Miss shader - called when ray doesn't hit any geometry
[shader("miss")]
void rmissMain(inout HitPayload payload){
  //payload.depth = -1;
  // Do nothing
}

[shader("miss")]
void rmissShadow(inout ShadowPayload payload){   
  payload.occluded = false;
}