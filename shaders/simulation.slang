
#include "shaderio.h"               // Shared definitions with CPU
#include "utils/descriptors.slang"  // External memory definitions
#include "utils/quat.slang"         // Quaternion support
#include "utils/evalFuncs.slang"    // Scene evaluation functions

void integrate(inout DynamicObject dyn){
  /* 
  float3 pos_diff = dyn.position.xyz - dyn.prev_position.xyz;
  printf("Integrate: pos_diff %f,%f,%f len %f\n",
    pos_diff.x,pos_diff.y,pos_diff.z,length(pos_diff)
  );
 */
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
  float3 point)
{
  float C = length(corr);

  if(C == 0.0)
    return 0.0;

  float dt_m2 = 1.0 / (pushConst.pyp.dts * pushConst.pyp.dts);
  float3 normal = normalize(corr);

  float w = getInverseMass(na, normal, point);
  if(hasB) w += getInverseMass(nb, normal, point);

  // XPBD
  float alpha = compliance * dt_m2;
  float lambda = -C / (w + alpha);

  normal *= -lambda;

  applyCorrection(na, normal, point);
  if(hasB) applyCorrection(nb, -normal, point);

  printf("lambda %f | alpha %f | dt_m2 %f | dts %f | normal len %f | w %f \n",
    lambda, alpha, dt_m2, pushConst.pyp.dts, length(normal),w
  );
  return lambda * dt_m2;
}

float3 local2World(DynamicObject dyn, float3 p){
  return rotateByQuat(dyn.rotation, p) + dyn.position.xyz;
}

float3 world2Local(DynamicObject dyn, float3 p){
  return rotateByQuat(dyn.inv_rotation, (p - dyn.position.xyz));
}

static float3 prev_n = float3(0.0,0.0,0.0);
static int substep = 0;

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
  float gradLen = length(grad_center);
  if(gradLen < 1e-4) return;
  float3 normal_c = grad_center/gradLen;
  float C_center = radius-centerV;

  float n_diff = any(prev_n!=0.0)? dot(prev_n,normal_c): -2;
  prev_n = normal_c;

  float3 sp = center - normal_c*radius;
  float surfaceV = mapStatic(sp,nearRange);
  float3 grad_s = evalGradStatic(sp,h,nearRange);
  float3 normal_s = normalize(grad_s);
  float C_surface = -surfaceV;

  if(surfaceV >= 0)
    return;

  float depth = 0.0, step = radius/2.0, mediumV;
  float3 mp;
  for(int i = 2; i<10; i++){
    mp = center-normal_c*depth;
    mediumV = mapStatic(mp,nearRange);
    if(abs(mediumV) < radius/100.0) break;
    depth += step*sign(mediumV);
    step /= 2;
  }
  
  float C_medium = radius-depth;

  float C = C_medium*0.2;

  if(C <= 0.0){
    return;
  }

  float maxCorr = radius * 0.5;
  C = clamp(C, 0.0, maxCorr);

  float3 corr = normal_c * C;
 
  float applyRet = -applyCorrection(
    dyn, dyn, false,
    compliance, corr,
    mp);

  printf("applyRet %f | centerV: %f | surfaceV: %f | medium depth: %f | C %f | n_diff: %f | gradLen %f | frame %i | substep %i\n",
    applyRet, centerV, surfaceV, depth, C, n_diff, gradLen, pushConst.frameCount, substep);

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
    substep = sub_step;
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