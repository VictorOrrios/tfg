#include "shaderio.h"
#include "nvshaders/constants.h.slang"
#include "utils/common.slang"     // Shared definitions with shaders
#include "utils/bbox.slang"       // AABB helpers
#include "utils/palettes.slang"   // Debug color palette


[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

[[vk::binding(BindingPoints::normalBuffer)]] RWTexture2D<float4> normalImage;
[[vk::binding(BindingPoints::albedoBuffer)]] RWTexture2D<float4> albedoImage;
[[vk::binding(BindingPoints::depthBuffer)]] RWTexture2D<float> depthImage;
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;
[[vk::binding(BindingPoints::aabbs)]] StructuredBuffer<Bbox> aabbs;
[[vk::binding(BindingPoints::tLas)]] RaytracingAccelerationStructure topLevelAS;

 // Numer of voxel2 per axis of globalGrids
[[vk::binding(BindingPoints::globalGrid)]] RWTexture3D<float> globalGrid;

#ifdef NDEBUGSHADER
// TODO: Change to false for final release and boost performance
const static bool debugColor = pushConst.debug.mode == 1;
const static bool bboxColor = pushConst.debug.mode == 5;
#else
const static bool debugColor = pushConst.debug.mode == 1;
const static bool bboxColor = pushConst.debug.mode == 5;
#endif

// Ray payload structure - carries data through the ray tracing pipeline
struct HitPayload
{
    float3 debug;
    float3 normal;
    float depth;
};

float3 scalarToColor(float x){
  return scalarToColor(x, pushConst.debug.palette);
}

void sampleVoxel2(int3 index, out float4 sBottom, out float4 sTop){
  sBottom = float4(
    globalGrid[index + int3(0,0,0)],  // s000
    globalGrid[index + int3(1,0,0)],  // s100
    globalGrid[index + int3(1,0,1)],  // s101
    globalGrid[index + int3(0,0,1)]   // s001
  )  * MAX_VOXEL_VALUE;

  sTop = float4(
    globalGrid[index + int3(0,1,0)],  // s010
    globalGrid[index + int3(1,1,0)],  // s110
    globalGrid[index + int3(1,1,1)],  // s111
    globalGrid[index + int3(0,1,1)]   // s011
  )  * MAX_VOXEL_VALUE;
}

float triLerp(float3 p, float4 sBottom, float4 sTop){
  //p = saturate(p); // Ensures it is in [0,1]³
  return lerp(
    lerp(
      lerp(sBottom.r,sBottom.a,p.z),
      lerp(sBottom.g,sBottom.b,p.z),
      p.x
    ),
    lerp(
      lerp(sTop.r,sTop.a,p.z),
      lerp(sTop.g,sTop.b,p.z),
      p.x
    ),
    p.y
  );
}

// Calculates analytically the normal of a voxel2 at point p
// Pre: p must be within voxel range in [0,1]³
float3 calcNormalAnalytic(float3 p, float4 sBottom, float4 sTop){
  //p = saturate(p);
  float dx = lerp(
    lerp(
      sBottom.g - sBottom.r,  // s100 - s000
      sTop.g - sTop.r,        // s110 - s010
      p.y),
    lerp(
      sBottom.b - sBottom.a,  // s101 - s001
      sTop.b - sTop.a,        // s111 - s011
      p.y),
    p.z
  );

  float dy = lerp(
    lerp(
      sTop.r - sBottom.r,     // s010 - s000
      sTop.g - sBottom.g,     // s110 - s100
      p.x),
    lerp(
      sTop.a - sBottom.a,     // s011 - s001
      sTop.b - sBottom.b,     // s111 - s101
      p.x),
    p.z
  );

  float dz = lerp(
    lerp(
      sBottom.a - sBottom.r,  // s001 - s000
      sBottom.b - sBottom.g,  // s101 - s100
      p.x),
    lerp(
      sTop.a - sTop.r,        // s011 - s010
      sTop.b - sTop.g,        // s111 - s110
      p.x),
    p.y
  );

  return normalize(float3(dx,dy,dz));
}

float sphereTraceGridMod(Ray r, Bbox bbox, out float3 normal, out float3 debug){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS;
  Bbox gBbox = unitBBox;
  debug = float3(0);

  // Debug visualizers
  const bool drawSilhouette = debugColor && false;  // Draw the silhouette of the grid
  const bool drawSteps      = debugColor && true;   // Draw the numer of steps taken
  const bool drawHitIndex   = debugColor && false;  // Draw the grid voxel index at the hit point

  normal = float3(0);

  // Initial boundig box hit test
  float3 p, far, dir = r.dir;
  bool hit = intersectCube(r, bbox, p, far);
  if(!hit) return -1;

  // Silhouette highlighter
  if(drawSilhouette && length(far-p)<0.02){
    debug = scalarToColor(-1);
    return 1;
  }

  const float3 gridSize = gBbox.bMax - gBbox.bMin;          // World space grid size
  const float3 voxelSize = gridSize/NUM_VOXELS_PER_AXIS;    // Size of a voxel
  const float3 voxelDensity = NUM_VOXELS_PER_AXIS/gridSize; // How many voxels per unit of world space
  
  float3 hitCoor, voxelCoor;
  int3 currentIndex, lastIndex;
  float4 coeff, sBottom, sTop;

  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    hitCoor = (p-gBbox.bMin)*voxelDensity; // P coordinates relative to grid, in [0,1]³

    // TODO: Check if it needs clamping
    currentIndex = int3(trunc(hitCoor));  // Grid index of p
    voxelCoor = frac(hitCoor);            // P Coordinates relative to voxel, in [0,1]³

    // If this voxel hasn't been visited yet
    if(steps == 1 || any(currentIndex != lastIndex)){
      // Update current voxel data and index
      sampleVoxel2(currentIndex,sBottom,sTop);
      //getkConstants(sBottom,sTop,k0123,k4567);
      lastIndex = currentIndex;
    }

    // Interpolate current sdf value of the voxel2
    //float t = triLerp(voxelCoor,sBottom,sTop);
    float t = triLerp(voxelCoor,sBottom, sTop);


    // Cutoff condition: A distance of 1/10 voxels
    // TODO: Fine tune to better performance/quality
    if(t < voxelSize.x/10){

      // Debug visualizer of step count
      if(drawSteps) debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
      // Debug visualizer of hit grid index
      if(drawHitIndex) debug = float3(currentIndex)/float3(NUM_VOXELS_PER_AXIS);
      
      normal = calcNormalAnalytic(voxelCoor,sBottom,sTop);
      // Returns depth of hit
      return length(p - r.orig);
    }
    
    // Update p to move the estimated length to surface
    p += r.dir*t;
  
    // Check if ray has left the bbox
    if(!insideBbox(p,bbox)){
      // Outside grid
      if(drawSteps){
        debug = scalarToColor(float(steps)/float(MAX_ITERATIONS));
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
    debug = scalarToColor(1.1);
    return 1;
  }else{
    // No hit
    return -1;
  }
}

Ray get_ray(float2 uv){
  // Calculate offsets
  float2 ndc = uv * 2.0 - 1.0;
  float4 rayClip = float4(ndc.x, ndc.y, -1.0, 1.0);

  float4 view = mul(rayClip,sceneInfo.projMatrix);
  float3 world = mul(float4(normalize(view.xyz),0.0),sceneInfo.viewMatrix).xyz;
  float3 dir = normalize(world);

  Ray ray;
  ray.orig = sceneInfo.cameraPosition;
  ray.dir = dir;

  return ray;
}

// Ray generation shader - entry point for each pixel
[shader("raygeneration")]
void rgenMain()
{
    float depth; float3 albedo, normal, debug;

    // Get the current pixel coordinates and image dimensions
    float2 launchID = (float2)DispatchRaysIndex().xy;     // Current pixel (x,y)
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
    ray.TMin = 0.001;    // Minimum distance to avoid self-intersection
    ray.TMax = INFINITE; // Maximum distance (infinite for primary rays)

    // Initialize ray payload with default values
    HitPayload payload;
    payload.debug = float3(0);
    payload.normal = float3(1);
    payload.depth = -1;
    

    // Cast the ray into the scene using the acceleration structure
    // Parameters: AS, flags, instance mask, sbt offset, sbt stride, miss offset, ray, payload
    TraceRay(topLevelAS, rayFlags, 0xff, 0,0,0, ray, payload);
    
    albedo = float3(1);
    debug = payload.debug;
    normal = payload.normal;
    depth = payload.depth;

    bool hit = depth >= 0.0;
    if(hit){
        albedo = float3(1.0);
    }else{
        albedo = float3(0);
        debug = r.dir;
    }

    if(debugColor || bboxColor){
        albedo = debug;
    }

    normalImage[int2(launchID)] = float4(normal, 0.0F);
    albedoImage[int2(launchID)] = float4(albedo,1.0F);
    depthImage[int2(launchID)] = depth;
}


[shader("intersection")]
void rintMain()
{
    HitPayload payload;
    Ray r; 
    r.orig = WorldRayOrigin();
    r.dir  = WorldRayDirection();

    uint instanceID = InstanceIndex();
    uint primitiveID = PrimitiveIndex();
    uint geometryID = GeometryIndex();
    Bbox bbox = aabbs[instanceID];

    if(bboxColor){
        payload.debug = scalarToColor(float(instanceID)/float(pushConst.numObjects));
        payload.depth = 1.0;
        payload.normal = float3(1);
        ReportHit(payload.depth, 0, payload);
        return;
    }

    payload.depth = sphereTraceGridMod(r,bbox,payload.normal,payload.debug);

    if(payload.depth >= 0.0){
        ReportHit(payload.depth, 0, payload);
    }
}

// Closest hit shader - called when ray hits geometry
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in HitPayload attr)
{
    // For now, just set a simple solid red color for all hits
    // In a full implementation, this would:
    // 1. Sample material properties (albedo, normal, etc.)
    // 2. Calculate lighting (direct + indirect)
    // 3. Potentially cast secondary rays (reflection, refraction)
    payload = attr;

}

// Miss shader - called when ray doesn't hit any geometry
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
    // Set background color when ray misses all geometry
    // In a full implementation, this could sample from:
    // - Skybox textures
    // - Procedural sky gradients
    // - Environment maps
    payload.debug = float3(0,0,1); // Gray background
/* 
    Ray r;
    r.orig = WorldRayOrigin();
    r.dir  = WorldRayDirection();
    Bbox bbox = aabbs[0];
    float3 nearP, farP;
    if(intersectCube(r, bbox, nearP, farP))
    {
        payload.color = float3(0, 1, 0);
    }
     */
}
