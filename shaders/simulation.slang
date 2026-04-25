
#include "shaderio.h"               // Shared definitions with CPU
#include "utils/descriptors.slang"  // External memory definitions
#include "utils/quat.slang"         // Quaternion support
#include "utils/evalFuncs.slang"    // Scene evaluation functions

void integrate(inout DynamicObject dyn){
  // Linear motion
  dyn.prev_position = dyn.position;
  dyn.vel.xyz += pushConst.pyp.gravity * pushConst.pyp.dts;
  dyn.position.xyz += dyn.vel.xyz * pushConst.pyp.dts;

  // Angular motion
  dyn.prev_rotation = dyn.rotation;
  float4 d_rot = float4(dyn.omega.xyz, 0.0);
  d_rot = quatMul(d_rot, dyn.rotation);
  dyn.rotation += 0.5 * pushConst.pyp.dts * d_rot;
  dyn.rotation = quatNormalize(dyn.rotation);
  dyn.inv_rotation = quatInverse(dyn.rotation);
}

void updateVelocities(inout DynamicObject dyn){
  // Linear
  dyn.vel.xyz = (dyn.position.xyz - dyn.prev_position.xyz) / pushConst.pyp.dts;

  // Angular
  float4 pre_rot_inv = quatInverse(dyn.prev_rotation);
  float4 d_rot = quatMul(dyn.rotation, pre_rot_inv);

  dyn.omega.xyz = d_rot.xyz * (2.0 / pushConst.pyp.dts);

  if(d_rot.w < 0.0)
    dyn.omega.xyz *= -1.0;
}

float getInverseMass(const DynamicObject dyn, float3 normal, float3 pos){
  float3 rn = pos - dyn.position.xyz;
  rn = cross(rn, normal);

  rn = rotateByQuat(dyn.inv_rotation, rn);

  float w = dyn.inv_mass +
    rn.x * rn.x * dyn.inv_inertia.x +
    rn.y * rn.y * dyn.inv_inertia.y +
    rn.z * rn.z * dyn.inv_inertia.z;

  return w;
}

void applyCorrection(inout DynamicObject dyn, float3 corr, float3 point){
  // Linear
  dyn.position.xyz += corr * dyn.inv_mass;

  // Angular
  float3 dOmega = point - dyn.position.xyz;
  dOmega = cross(dOmega, corr);

  dOmega = rotateByQuat(dyn.inv_rotation, dOmega);
  dOmega *= dyn.inv_inertia.xyz;
  dOmega = rotateByQuat(dyn.rotation, dOmega);

  float4 d_rot = float4(dOmega, 0.0);
  d_rot = quatMul(d_rot, dyn.rotation);

  dyn.rotation += 0.5 * d_rot;
  dyn.rotation = quatNormalize(dyn.rotation);

  dyn.inv_rotation = quatInverse(dyn.rotation);
}

float applyCorrection(
  inout DynamicObject na,
  inout DynamicObject nb,
  bool hasB,
  float compliance,
  float3 corr,
  float3 pointA,
  float3 pointB)
{
  float C = length(corr);

  if(C == 0.0)
    return 0.0;

  float dt_m2 = 1.0 / (pushConst.pyp.dts * pushConst.pyp.dts);
  float3 normal = corr / C;

  float w = getInverseMass(na, normal, pointA);
  if(hasB) w += getInverseMass(nb, normal, pointB);

  // XPBD
  float alpha = compliance * dt_m2;
  float lambda = -C / (w + alpha);

  float3 impulse = normal * (-lambda);

  applyCorrection(na, impulse, pointA);
  if(hasB) applyCorrection(nb, -impulse, pointB);

  return lambda * dt_m2;
}

float3 local2World(DynamicObject dyn, float3 p){
  return rotateByQuat(dyn.rotation, p) + dyn.position.xyz;
}

float3 world2Local(DynamicObject dyn, float3 p){
  return rotateByQuat(dyn.inv_rotation, (p - dyn.position.xyz));
}

void solveCollisionConstraint(
  inout DynamicObject dyn,
  float compliance)
{
  float sdValue = mapStatic(dyn.position.xyz);
  float radius = dyn.scale * 0.5;

  float C = radius - sdValue;

  if(C <= 0.0)
    return;

  float3 normal = evalNormalStatic(dyn.position.xyz);
  float3 corr = normal * C;

  applyCorrection(
    dyn, dyn, false,
    compliance, corr,
    dyn.position.xyz,
    float3(0.0,0.0,0.0));
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_1D, 1, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  if(threadIdx.x >= pushConst.numDynamicObjects || pushConst.pyp.dts <= 0.0)
    return;

  int dObjIdx = threadIdx.x;
  DynamicObject dyn = dynamic_objects[dObjIdx];
  /* 
  printf("Processing dyn obj %i static obj %i pos %f,%f,%f\n",
    dObjIdx,dyn.index,
    dyn.position.x,dyn.position.y,dyn.position.z);
  */
  for(int sub_step = 0; sub_step < SIM_NUM_SUBSTEPS; sub_step++){
    integrate(dyn);
    solveCollisionConstraint(dyn,0.0);
    updateVelocities(dyn);
  }
  /* 
  printf("Current map %f\n",
    mapStatic(dyn.position.xyz)
  );
 */
/* 
  printf("End of processing dyn obj %i pos %f,%f,%f\n",
    dObjIdx,
    dyn.position.x,dyn.position.y,dyn.position.z);
   */

  dynamic_objects[dObjIdx] = dyn;
}