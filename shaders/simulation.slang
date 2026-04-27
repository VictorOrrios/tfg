
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

void applyCorrection(inout DynamicObject dyn, float3 impulse, float3 point){
  float3 center_old = dyn.position.xyz;
  
  float3 r = point - center_old;
  float3 torque = cross(r, impulse);
  
  // Linear
  dyn.position.xyz += impulse * dyn.inv_mass;
  
  float3 local_torque = rotateByQuat(dyn.inv_rotation, torque);
  float3 delta_omega = local_torque * dyn.inv_inertia.xyz;
  delta_omega = rotateByQuat(dyn.rotation, delta_omega);
  
  // Angular
  float angle = length(delta_omega);
  if (angle > 1e-6f) {
    float3 axis = normalize(delta_omega);
    float4 delta_q = float4(axis * sin(angle * 0.5f), cos(angle * 0.5f));
    dyn.rotation = quatMul(delta_q, dyn.rotation);
    dyn.rotation = quatNormalize(dyn.rotation);
    dyn.inv_rotation = quatInverse(dyn.rotation);
  }
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
  float3 center = dyn.position.xyz;
  float radius = dyn.scale * 0.5;
  float nearRange = 4.0*dyn.scale;

  float centerV = mapStatic(center,nearRange);
  const float h = 0.001;
  float3 grad_center = evalGradStatic(center,h,nearRange);
  float3 normal_center = normalize(grad_center);
  float C_center = radius-centerV;
  float vertical = clamp(dot(float3(0,1,0),normal_center),0.0,1.0);

  float3 sp = center - normal_center*radius;
  float surfaceV = mapStatic(sp,nearRange);
  float3 grad_s = evalGradStatic(center,h,nearRange);
  float3 normal_s = normalize(grad_s);
  float raw_C_surface = -surfaceV;
  float C_surface = min(radius/2.0, raw_C_surface);


  float depth = centerV;
  for(int i = 0; i<8; i++){
    float d = mapStatic(center-normal_center*depth,nearRange);
    depth += d;
    if(abs(d) < 1e-4)
      break;
  }
  
  float C_medium = radius-depth;

  float C = C_surface;

  if(C <= 0.0){
    return;
  }

  float maxCorr = radius * 0.5;
  C = clamp(C, 0.0, maxCorr);

  float3 corr = normal_s * C;
   
 
  float force = -applyCorrection(
    dyn, dyn, false,
    compliance, corr,
    sp,
    float3(0.0,0.0,0.0));

  printf("force %f | surfaceV: %f | raw_C: %f | C_surface: %f | frame %i\n",
    force,surfaceV, raw_C_surface, C_surface, pushConst.frameCount);
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