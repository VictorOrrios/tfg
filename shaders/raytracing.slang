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

[[vk::binding(BindingPoints::clipMap)]] RWTexture3D<uint32_t> clipMap;
[[vk::binding(BindingPoints::brickAtlas)]] RWTexture3D<float> brickAtlas;

#ifdef NDEBUGSHADER
// TODO: Change to false for final release and boost performance
const static bool debugColor = pushConst.debug.mode == 1;
const static bool bboxColor = pushConst.debug.mode == 5;
#else
const static bool debugColor = pushConst.debug.mode == 1;
const static bool bboxColor = pushConst.debug.mode == 5;
#endif

const static float EPSILON = 0.000001;

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

void sampleVoxel2(int3 index, int level, out float4 sBottom, out float4 sTop){
  const int NUM_VOXELS_PER_BRICK = BRICK_SIZE-1;
  const float max_voxel_value = MAX_VOXEL_VALUES[level];

  int3 brickIndex = index/NUM_VOXELS_PER_BRICK;
  brickIndex.z += NUM_BRICKS_PER_AXIS*level;
  uint pointer = clipMap[brickIndex] & NOT_DIRTY_BIT;

  if(pointer == UNIFORM_POSITIVE_BRICK_POINTER){
    sBottom = float4(max_voxel_value);
    sTop    = float4(max_voxel_value);
  }else if(pointer == UNIFORM_NEGATIVE_BRICK_POINTER){
    sBottom = float4(-max_voxel_value);
    sTop    = float4(-max_voxel_value);
  }else{

    int3 brickAtlasIndex = int3(pointer%BRICK_PER_ATLAS_AXIS,pointer/BRICK_PER_ATLAS_AXIS,0);
    int3 voxelId = int3(index.x%NUM_VOXELS_PER_BRICK,index.y%NUM_VOXELS_PER_BRICK,index.z%NUM_VOXELS_PER_BRICK);
    int3 voxelIndex = brickAtlasIndex*BRICK_SIZE + voxelId;

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
}

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

int getLevelOfPoint(float3 p) {
  int3 camId0 = sceneInfo.cameraId0.xyz;
  int3 pId0 = int3(floor(p / BRICK_SIZES[0]));
  
  int3 diff = pId0 - camId0;
  diff += max(0,sign(diff));
  diff = abs(diff);
  int maxDiff = max(diff.x,max(diff.y,diff.z));
  
  const int HALF = NUM_BRICKS_PER_AXIS / 2;
  float logVal = log2((float)(maxDiff) / (float)HALF);
  int level = clamp((int)ceil(logVal),0,CLIPMAP_LEVELS - 1);
  
  int3 camId = id0LevelTransform(camId0,level);
  int3 pId = id0LevelTransform(pId0,level);
  int3 rel_id = pId - camId;
  if (all(rel_id < HALF)) {
    return level;
  }else{
    return min(level+1,CLIPMAP_LEVELS - 1);
  }
}

float3 analyticNormal(float3 p){
  int level = getLevelOfPoint(p);
  float voxel_size = VOXEL_SIZES[level];
  // Scale normalization
  float3 pVoxelNorm = p/voxel_size;
  // Truncate normalized coordinates to get the id
  int3 voxelId = int3(floor(pVoxelNorm));
  // Get the grid indices with floor division modulus (WARNING: Normal modulus % will not give correct result)
  int3 voxelIndex = id2voxelIndex(voxelId);
  // Get grid index and coordinates inside the voxel
  int3 currentIndex = max(int3(0),min(int3(NUM_VOXELS_PER_AXIS-1),
    voxelIndex));
  float3 voxelCoor = frac(pVoxelNorm);
  // Sample voxel
  float4 sBottom,sTop;
  sampleVoxel2(currentIndex,level,sBottom,sTop);
  // Calc normal
  return calcNormalAnalytic(voxelCoor,sBottom,sTop);
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

    if(step == 0 || any(prev_voxel_offset != voxel_offset) || true){
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

float sphereTraceBbox(Ray r, Bbox bbox){
  const int MAX_ITERATIONS = NUM_VOXELS_PER_AXIS * int(CLIPMAP_LEVELS*0.5);

  float3 p = r.orig, dir = r.dir;
  int3 currentIndex, lastIndex;
  float4 sBottom, sTop;
  int level;

  for(int steps = 1; steps < MAX_ITERATIONS; steps++){
    level = getLevelOfPoint(p);

    const float voxel_size = VOXEL_SIZES[level];

    // Scale normalization
    float3 pVoxelNorm = p/voxel_size;
    // Truncate normalized coordinates to get the id
    int3 voxelId = int3(floor(pVoxelNorm));
    // Get the grid indices with floor division modulus (WARNING: Normal modulus % will not give correct result)
    int3 voxelIndex = id2voxelIndex(voxelId);
    // Get grid index and coordinates inside the voxel
    currentIndex = voxelIndex;
    float3 voxelCoor = frac(pVoxelNorm);

    // If this voxel hasn't been visited yet
    if(steps == 1 || any(currentIndex != lastIndex)){
    //if(true){
      // Update current voxel data and index
      sampleVoxel2(currentIndex,level,sBottom,sTop);
      //sampleVoxel2New(voxelOffset,currentIndex,level,sBottom,sTop);
      //getkConstants(sBottom,sTop,k0123,k4567);
      lastIndex = currentIndex;
    }

    // Interpolate current sdf value of the voxel2
    float t = triLerp(voxelCoor,sBottom,sTop);


    // Cutoff condition: A distance of 1/10 voxels
    // TODO: Fine tune to better performance/quality
    if(t < voxel_size/10){
      return length(p - r.orig);
    }
    
    // Update p to move the estimated length to surface
    p += r.dir*t;
  
    // Check if ray has left the bbox
    if(!insideBbox(p,bbox)){
        return -1;
    }
  }

  // Over iterated 
    return -1;
}

Ray get_ray(float2 uv){
  // Calculate offsets
  float2 ndc = uv * 2.0 - 1.0;
  float4 rayClip = float4(ndc.x, ndc.y, -1.0, 1.0);

  float4 view = mul(rayClip,sceneInfo.projMatrix);
  float3 world = mul(float4(normalize(view.xyz),0.0),sceneInfo.viewMatrix).xyz;
  float3 dir = normalize(world);

  Ray ray;
  ray.orig = sceneInfo.cameraPosition.xyz;
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
    //debug = r.dir;
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

  float3x4 objToWorld = ObjectToWorld();
  float3 pos = float3(objToWorld[0][3],objToWorld[1][3],objToWorld[2][3]);

  float size = objToWorld[0][0];
  int level = InstanceID();
  Bbox bbox = Bbox(pos,pos+size);

  float3 nearP, farP;
  if(intersectCube(r,bbox,nearP,farP)){
    nearP = max(bbox.bMin,min(bbox.bMax,nearP));
    Ray brickRay = Ray(nearP,r.dir);
    float depthToBbox = length(r.orig-nearP);
    payload.depth = depthToBbox;

    if(sphereTraceBrick(brickRay,instanceID,pos,level,payload.depth)){
      payload.depth += depthToBbox; 
      //payload.debug = scalarToColor(float(instanceID)/float(NUM_BRICKS_IN_ATLAS));
      payload.debug = scalarToColor(float(level)/float(CLIPMAP_LEVELS-1));
      ReportHit(payload.depth, 0, payload);
    }

/*  
    payload.depth = sphereTraceBbox(brickRay,bbox);
    if(payload.depth >= 0){
      payload.depth += depthToBbox; 
      //payload.debug = scalarToColor(float(instanceID)/float(NUM_BRICKS_IN_ATLAS));
      payload.debug = scalarToColor(float(level)/float(CLIPMAP_LEVELS-1));
      ReportHit(payload.depth, 0, payload);
    }
 */
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
  Ray r; 
  r.orig = WorldRayOrigin();
  r.dir  = WorldRayDirection();
  float3 p = r.orig + r.dir*payload.depth;
  payload.normal = analyticNormal(p);
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
