#include "shaderio.h"             // Shared definitions with CPU
#include "utils/common.slang"     // Shared definitions with shaders
#include "utils/bbox.slang"       // AABB helpers
#include "utils/sdf.slang"        // SDF functions, operations and primitives

//----------------------------
// External memory definition
//----------------------------
[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;

[[vk::binding(BindingPoints::aabbs)]] StructuredBuffer<Bbox> aabbs;
[[vk::binding(BindingPoints::objects)]] StructuredBuffer<SceneObject> objects;
[[vk::binding(BindingPoints::globalGrid)]] RWTexture3D<float> globalGrid;




float evalPrimitive(int primType, float3 p){
  switch(primType)
  {
    default:
    case 0: return sdEmpty(p);
    case 1: return sdBox(p);
    case 2: return sdSphere(p);
    case 3: return sdTorus(p);
    case 4: return sdSnowMan(p);
  }
}

float evalCombOp(int opIndex, float d, float result, float smoothness){
  switch(opIndex)
  {
    default:
    case 0: return opUnion(d,result);
    case 1: return opSubtraction(d,result);
    case 2: return opIntersection(d,result);
    case 3: return opSmoothUnion(d,result,smoothness);
    case 4: return opSmoothSubtraction(d,result,smoothness);
    case 5: return opSmoothIntersection(d,result,smoothness);
  }
}

float3 applyRepOp(int opIndex, float3 p, float3 spacing, int3 limit){
  switch(opIndex)
  {
    default:
    case 0: return p;
    case 1: return opLimRepetition(p,spacing,limit);
    case 2: return opRepetition(p,spacing);
  }
}

float3 applyDefOp(int opIndex, float3 p, float3 defP){
  switch(opIndex)
  {
    default:
    case 0: return p;
    case 1: return opElongate(p,defP);
  }
}

float map(float3 point){
  const float iniD = 10000.0f;
  float result = iniD;

  for(int obIdx = 0; obIdx < pushConst.numObjects; obIdx++) {
    float3 p = float3(point);
    Bbox bbox = aabbs[obIdx];

    // If not inside bbox primitive continue with next 
    if(!insideBbox(p,bbox))
      continue;

    SceneObject params = objects[obIdx];

    p = mul(params.tInv, float4(p, 1.0)).xyz;

    p = applyRepOp(params.repOp, p, params.spacing.xyz, params.limit.xyz);

    p = applyDefOp(params.defOp, p, params.defP.xyz);

    p /= params.scale;

    float d = evalPrimitive(params.type, p) - params.roundness;

    d *= params.scale;

    result = evalCombOp(params.combOp, d, result, params.smoothness);
  }

  return result;
}

[shader("compute")]
[numthreads(MAX_BUILD_JOB_SIZE, MAX_BUILD_JOB_SIZE, MAX_BUILD_JOB_SIZE)]
void computeMain(uint3 tId : SV_DispatchThreadID){
  const float3 center = float3(0.5);
  const float3 p = tId*VOXEL_SIZE - center;
  float mapV = clamp(map(p)/MAX_VOXEL_VALUE,-1.0,1.0);
  globalGrid[tId] = mapV;
}
