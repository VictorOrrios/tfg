#include "shaderio.h"
#include "nvshaders/constants.h.slang"
#include "utils/common.slang"     // Shared definitions with shaders
#include "utils/bbox.slang"       // AABB helpers


[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

[[vk::binding(BindingPoints::normalBuffer)]] RWTexture2D<float4> normalImage;
[[vk::binding(BindingPoints::albedoBuffer)]] RWTexture2D<float4> albedoImage;
[[vk::binding(BindingPoints::depthBuffer)]] RWTexture2D<float> depthImage;
[[vk::binding(BindingPoints::sceneInfo)]] ConstantBuffer<SceneInfo> sceneInfo;
[[vk::binding(BindingPoints::aabbs)]] StructuredBuffer<Bbox> aabbs;
[[vk::binding(BindingPoints::tLas)]] RaytracingAccelerationStructure topLevelAS;

// Ray payload structure - carries data through the ray tracing pipeline
struct HitPayload
{
    float3 color;   // Accumulated color along the ray path
    float  weight;  // Weight/importance of this ray (for importance sampling)
    int    depth;   // Current recursion depth (for limiting bounces)
};

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
    float depth; float3 albedo, normal;

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
    payload.color = float3(1,0,1);  // Start with black
    payload.weight = 1;               // Full weight for primary rays
    payload.depth = 0;                // Start at depth 0
    

    // Cast the ray into the scene using the acceleration structure
    // Parameters: AS, flags, instance mask, sbt offset, sbt stride, miss offset, ray, payload
    TraceRay(topLevelAS, rayFlags, 0xff, 0,0,0, ray, payload);
    
    // Get the final color from the ray tracing result
    float3 color = payload.color;
    
    albedo = color;
    //albedo = float3(uv,0);
    normal = float3(0.0);
    depth = 1.0;

    // Write the result to the output image
    normalImage[int2(launchID)] = float4(normal, 0.0F);
    albedoImage[int2(launchID)] = float4(albedo,1.0F);
    depthImage[int2(launchID)]  = depth;
}

[shader("intersection")]
void rintMain()
{
    BuiltInTriangleIntersectionAttributes attr;
    ReportHit(1.0f, 0, attr);
    return;

    Ray r;
    r.orig = WorldRayOrigin();
    r.dir  = WorldRayDirection();

    uint primitiveID = PrimitiveIndex();
    Bbox bbox = aabbs[primitiveID];

    float3 nearP, farP;
    if(intersectCube(r, bbox, nearP, farP)){
        float3 vecToNear = nearP - r.orig;
        float3 vecToFar = farP - r.orig;
        
        float tNear = length(nearP - r.orig);
        float tFar = length(farP - r.orig);
        
        if(tNear < 0 && tFar >= 0){
            tNear = tFar;
        }
        
        if(tNear >= RayTMin()){
            BuiltInTriangleIntersectionAttributes attr;
            attr.barycentrics = float2(0, 0);
            
            ReportHit(tNear, 0, attr);
        }
    }
}

// Closest hit shader - called when ray hits geometry
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // For now, just set a simple solid red color for all hits
    // In a full implementation, this would:
    // 1. Sample material properties (albedo, normal, etc.)
    // 2. Calculate lighting (direct + indirect)
    // 3. Potentially cast secondary rays (reflection, refraction)
    payload.color = float3(1, 0, 0); // Red color
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
    payload.color = float3(0,0,1); // Gray background
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
