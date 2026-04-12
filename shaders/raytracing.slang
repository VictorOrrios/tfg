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
  float3 albedo;
  float3 debug;
  float3 normal;
  float depth;
  int level;
};

struct IntersectionPayload{
  float3 debug;
  float depth;
  int level;
};


[shader("raygeneration")]
void rgenMain(){
  float depth, shadow; float3 albedo, normal, debug, position;

  // Get the current pixel coordinates and image dimensions
  float2 launchID = (float2)DispatchRaysIndex().xy;        // Current pixel (x,y)
  float2 launchSize = (float2)DispatchRaysDimensions().xy; // Image size (width,height)
  float2 uv = launchID/launchSize;

  const uint rayFlags = 0;

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
  payload.albedo = float3(0);
  payload.debug = float3(0);
  payload.normal = float3(1);
  payload.depth = -1;
  

  // Cast the ray into the scene using the acceleration structure
  // Parameters: AS, flags, instance mask, sbt offset, sbt stride, miss offset, ray, payload
  TraceRay(topLevelAS, rayFlags, 0xff, 0,0,0, ray, payload);
  
  albedo = payload.albedo;
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

  // Shadow ray experiment. TODO: Make it faster!!
  if(dot(normal,pushConst.lp.lightDir) > -0.3){
    HitPayload shadowPayload;
    shadowPayload.albedo = float3(0);
    shadowPayload.debug = float3(0);
    shadowPayload.normal = float3(1);
    shadowPayload.depth = -1;
    const float maxSoftDepthShadow = L0_AXIS_WORLD_SIZE;
    ray.Origin = r.orig + r.dir*depth + normal * 0.1 + pushConst.lp.lightDir*0.1;
    ray.Direction = pushConst.lp.lightDir;
    TraceRay(topLevelAS, rayFlags, 0xff, 0,0,0, ray, shadowPayload);
    hit = shadowPayload.depth >= 0.0;
    float shadowDepth = shadowPayload.depth;
    float smoothShadow = 1.0 - min(shadowDepth,maxSoftDepthShadow)/maxSoftDepthShadow;
    shadow = hit? 0.0 : 1.0;
  }

  normalImage[int2(launchID)] = float4(normal, 0.0F);
  albedoImage[int2(launchID)] = float4(albedo,1.0F);
  depthImage[int2(launchID)] = depth;
  shadowImage[int2(launchID)] = shadow;
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
  Material mat = evalMat(p);
  payload.albedo = mat.albedo.xyz;
  payload.normal = analyticNormal(p);
  //payload.normal = evalNormal(p);
}

// Miss shader - called when ray doesn't hit any geometry
[shader("miss")]
void rmissMain(inout HitPayload payload){
  //payload.depth = -1;
  // Do nothing
}
