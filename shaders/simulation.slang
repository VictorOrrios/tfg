
#include "shaderio.h"               // Shared definitions with CPU
#include "utils/descriptors.slang"  // External memory definitions
#include "utils/quat.slang"         // Quaternion support
#include "utils/evalFuncs.slang"    // Scene evaluation functions

const static int cube_p_size = 26;

const static float3 cube_p[cube_p_size] = {

  // Vertex
  float3(-0.5,-0.5,-0.5),
  float3( 0.5,-0.5,-0.5),
  float3(-0.5,-0.5, 0.5),
  float3( 0.5,-0.5, 0.5),
  float3(-0.5, 0.5,-0.5),
  float3( 0.5, 0.5,-0.5),
  float3(-0.5, 0.5, 0.5),
  float3( 0.5, 0.5, 0.5),

  // Edges
  float3( 0.0,-0.5,-0.5),
  float3( 0.0,-0.5, 0.5),
  float3( 0.0, 0.5,-0.5),
  float3( 0.0, 0.5, 0.5),

  float3(-0.5, 0.0,-0.5),
  float3(-0.5, 0.0, 0.5),
  float3( 0.5, 0.0,-0.5),
  float3( 0.5, 0.0, 0.5),

  float3(-0.5,-0.5, 0.0),
  float3( 0.5,-0.5, 0.0),
  float3(-0.5, 0.5, 0.0),
  float3( 0.5, 0.5, 0.0),

  // Faces
  float3( 0.0, 0.0,-0.5),
  float3( 0.0, 0.0, 0.5),
  float3(-0.5, 0.0, 0.0),
  float3( 0.5, 0.0, 0.0),
  float3( 0.0,-0.5, 0.0),
  float3( 0.0, 0.5, 0.0)
};

/* const static int sphere_p_size = 26;
const static float3 sphere_p[sphere_p_size] = {

  float3( 1, 0, 0),
  float3(-1, 0, 0),
  float3( 0, 1, 0),
  float3( 0,-1, 0),
  float3( 0, 0, 1),
  float3( 0, 0,-1),

  normalize(float3( 1, 1, 1)),
  normalize(float3(-1, 1, 1)),
  normalize(float3( 1,-1, 1)),
  normalize(float3(-1,-1, 1)),
  normalize(float3( 1, 1,-1)),
  normalize(float3(-1, 1,-1)),
  normalize(float3( 1,-1,-1)),
  normalize(float3(-1,-1,-1)),

  normalize(float3( 1, 1, 0)),
  normalize(float3(-1, 1, 0)),
  normalize(float3( 1,-1, 0)),
  normalize(float3(-1,-1, 0)),

  normalize(float3( 1, 0, 1)),
  normalize(float3(-1, 0, 1)),
  normalize(float3( 1, 0,-1)),
  normalize(float3(-1, 0,-1)),

  normalize(float3( 0, 1, 1)),
  normalize(float3( 0,-1, 1)),
  normalize(float3( 0, 1,-1)),
  normalize(float3( 0,-1,-1))
};
 */
 /* 
const static int sphere_p_size = 6;
const static float3 sphere_p[sphere_p_size] = {
  float3( 1, 0, 0),
  float3(-1, 0, 0),
  float3( 0, 1, 0),
  float3( 0,-1, 0),
  float3( 0, 0, 1),
  float3( 0, 0,-1)
};
 */

const static int sphere_p_size = 14;
const static float3 sphere_p[sphere_p_size] = {

  float3( 1, 0, 0),
  float3(-1, 0, 0),
  float3( 0, 1, 0),
  float3( 0,-1, 0),
  float3( 0, 0, 1),
  float3( 0, 0,-1),

  normalize(float3( 1, 1, 1)),
  normalize(float3(-1, 1, 1)),
  normalize(float3( 1,-1, 1)),
  normalize(float3(-1,-1, 1)),
  normalize(float3( 1, 1,-1)),
  normalize(float3(-1, 1,-1)),
  normalize(float3( 1,-1,-1)),
  normalize(float3(-1,-1,-1))
};

static float max_corr_length;
static const float FRICTION_COEFF_SPHERE = 0.5;
static const float FRICTION_COEFF_BOX = 0.5;

void integrate(inout DynamicObject dyn){
  // Linear motion
  dyn.poss_diff = dyn.position-dyn.prev_position;
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

void integrateLinear(inout DynamicObject dyn){
  // Linear motion
  dyn.poss_diff = dyn.position-dyn.prev_position;
  dyn.prev_position = dyn.position;
  dyn.vel.xyz += pushConst.pyp.gravity * pushConst.pyp.dts;
  dyn.position.xyz += dyn.vel.xyz * pushConst.pyp.dts;
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

void updateVelocitiesLinear(inout DynamicObject dyn){
  // Linear
  dyn.vel.xyz = (dyn.position.xyz - dyn.prev_position.xyz) / pushConst.pyp.dts;
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

float getInverseMassSphere(const DynamicObject dyn, float3 normal, float3 pos){
  float3 r = pos - dyn.position.xyz;
  float3 rn = cross(r, normal);

  float w = dyn.inv_mass + dot(rn, rn) * dyn.inv_inertia.x;

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

void applyCorrectionSphere(inout DynamicObject dyn, float3 impulse, float3 point){
  // Linear
  dyn.position.xyz += impulse * dyn.inv_mass;
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
/* 
  printf("lambda %f | alpha %f | dt_m2 %f | dts %f | normal len %f | w %f \n",
    lambda, alpha, dt_m2, pushConst.pyp.dts, length(normal),w
  );
 */
  return lambda * dt_m2;
}

float applyCorrectionSphere(
  inout DynamicObject dyn,
  float compliance,
  float3 corr,
  float3 point)
{
  float C = length(corr);

  if(C == 0.0)
    return 0.0;

  float dt_m2 = 1.0 / (pushConst.pyp.dts * pushConst.pyp.dts);
  float3 normal = normalize(corr);

  float w = getInverseMass(dyn, normal, point);

  // XPBD
  float alpha = compliance * dt_m2;
  float lambda = -C / (w + alpha);

  normal *= -lambda;

  applyCorrectionSphere(dyn, normal, point);

  return lambda * dt_m2;
}

float3 local2World(DynamicObject dyn, float3 p){
  return rotateByQuat(dyn.rotation, p) + dyn.position.xyz;
}

float3 world2Local(DynamicObject dyn, float3 p){
  return rotateByQuat(dyn.inv_rotation, (p - dyn.position.xyz));
}

float3 getVelocityAtPoint(DynamicObject dyn, float3 point)
{
  float3 r = point - dyn.position.xyz;
  float3 angularVel = cross(dyn.omega.xyz, r);
  return dyn.vel.xyz + angularVel;
}

void applyFriction(
  inout DynamicObject dyn, 
  float3 contactPoint, 
  float3 normal, 
  float mu)
{
  float3 v = dyn.vel.xyz;
  
  float vn = dot(v, normal);
  float3 vt = v - normal * vn;
  float vtLen = length(vt);
  
  if (vtLen > 1e-6){
    float reduction = mu * pushConst.pyp.dts;
    if (reduction > 1.0) reduction = 1.0;
    
    float3 dv = -vt * reduction;
    dyn.vel.xyz += dv;
    
    dyn.position.xyz += dv * pushConst.pyp.dts;
  }
}

void applyFrictionXPBD(
    inout DynamicObject dyn,
    float3 contactPoint,
    float3 normal,
    float mu,
    float compliance)
{
  float3 v_rel = getVelocityAtPoint(dyn, contactPoint);
  float vn = dot(v_rel, normal);
  float3 vt = v_rel - normal * vn;
  float vt_len = length(vt);
  
  if (vt_len < 1e-6) return;
  
  float3 t_dir = vt / vt_len;
  
  float3 delta_pos_tangential = -vt * pushConst.pyp.dts * mu;
  
  float invMass = dyn.inv_mass;
  if (invMass == 0.0) return;
  float3 impulse_t = delta_pos_tangential / invMass;
  
  applyCorrection(dyn, dyn, false, compliance, impulse_t, contactPoint);
}

void solveStaticCollisionCubeConstraint(
  inout DynamicObject dyn,
  float compliance)
{
  float radius = dyn.scale * 0.5;
  float nearRange = 4.0*dyn.scale;

  for(int i = 0; i<cube_p_size; i++){
    float3 local_p = cube_p[i]*dyn.scale;
    float3 global_p = local2World(dyn,local_p);
    float sdV = mapStatic(global_p,nearRange);
    if(sdV<0.0){
      float3 normal = evalNormalStatic(global_p,nearRange);
      float depth = -sdV;
      for(int s = 0; s<10; s++){
        float sdIV = mapStatic(global_p+normal*depth,nearRange);
        depth -= sdIV;
        if(sdIV>=0.0) break;
      }
      float C = clamp(depth,0.0,max_corr_length);
      float3 corr = normal*C;
      applyCorrection(
      dyn, dyn, false,
      compliance, corr,
      global_p);

      applyFrictionXPBD(
        dyn, global_p, normal, FRICTION_COEFF_BOX, 0.0
      );
    }
  }
}

void solveFastStaticCollisionSphereConstraint(
  inout DynamicObject dyn,
  float compliance)
{
  float3 center = dyn.position.xyz;
  float radius = dyn.scale * 0.5;
  float nearRange = 4.0*dyn.scale;

  float centerV = mapStatic(center,nearRange);
  if(centerV>=radius) return;

  float3 normal_c = evalNormalStatic(center,nearRange);
  
  float depth = radius-centerV;
  float C = clamp(depth,0.0,max_corr_length);
  float3 corr = normal_c*C;
  applyCorrection(
  dyn, dyn, false,
  compliance, corr,
  center);
}

void solveStaticCollisionSphereConstraint(
  inout DynamicObject dyn,
  float compliance)
{
  float3 center = dyn.position.xyz;
  float radius = dyn.scale * 0.5;
  float nearRange = 1.0*dyn.scale;

  float centerV = mapStatic(center,nearRange);
  if(centerV>=radius) return;
  float3 normal_c = evalNormalStatic(center,nearRange);

  bool touching = false;
  
  for(int i = 0; i<sphere_p_size; i++){
    float3 local_p = sphere_p[i]*dyn.scale/2.0;
    float3 global_p = local_p+center;
    float sdV = mapStatic(global_p,nearRange);
    if(sdV<0.0){
      touching = true;
      float3 normal = evalNormalStatic(global_p,nearRange);
      float depth = -sdV;
      float C = clamp(depth,0.0,max_corr_length);
      float3 corr = normal*C;
      applyCorrectionSphere(
        dyn,
        compliance, corr,
        global_p);
    }
  }


  if(touching) applyFriction(dyn, center, normal_c, FRICTION_COEFF_SPHERE);
  //if(touching) solveDragConstraint(dyn,4.0/pushConst.pyp.dts);

}

void solveBodyCollissionConstrain(
  inout DynamicObject dynA, 
  DynamicObject dynB,
  float compliance){
    
  float3 centerA = dynA.position.xyz;
  float3 centerB = dynB.position.xyz;
  float radiusA = dynA.scale*0.5;
  float radiusB = dynB.scale*0.5;
  float radiusSum = radiusA + radiusB;

  float3 center_diff = centerA-centerB;
  float d = length(center_diff);
  if(d>=radiusSum) return;

  float invWeightSum = dynA.inv_mass+dynB.inv_mass;
  float forceP = invWeightSum/dynA.inv_mass;

  float C = radiusSum-d;

  float dt_m2 = 1.0 / (pushConst.pyp.dts * pushConst.pyp.dts);
  float3 normal = normalize(center_diff);

  float w = getInverseMass(dynA, normal, centerA + normal*radiusA);
  w += getInverseMass(dynB, normal, centerB - normal*radiusB);

  // XPBD
  float alpha = compliance * dt_m2;
  float lambda = -C / (w + alpha);

  normal *= -lambda;

  dynA.corr_delta.xyz += normal * dynA.inv_mass;
  
}

void solveDragConstraint(
  inout DynamicObject dyn, 
  float compliance)
{
  float3 corr = dyn.prev_position.xyz - dyn.position.xyz;
  float applyRet = -applyCorrection(
    dyn, dyn, false,
    compliance, corr,
    dyn.position.xyz);
}

void solveDragConstraintSphere(
  inout DynamicObject dyn, 
  float compliance)
{
  float3 corr = dyn.prev_position.xyz - dyn.position.xyz;
  applyCorrectionSphere(
    dyn,
    compliance, corr,
    dyn.position.xyz);
}

void solveDynamicCollisionConstraint(
  inout DynamicObject dynA, int dynBIdx, float compliance 
){
  DynamicObject dynb = dynamic_objects[dynBIdx];


  dynamic_objects[dynBIdx] = dynb;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_1D, 1, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  if(threadIdx.x >= pushConst.numDynamicObjects || pushConst.pyp.dts <= 0.0)
    return;

  int dObjIdx = threadIdx.x;
  DynamicObject dyn = dynamic_objects[dObjIdx];

  const float bodyColissionCompliance = 0.0/pushConst.pyp.dts;
  const float colissionCompliance = 0.0/pushConst.pyp.dts;
  const float dragCompliance = 0.1/pushConst.pyp.dts;
  const float gravity_length_dts = length(pushConst.pyp.gravity)*pushConst.pyp.dts;

  int dynConstNum;
  if(dObjIdx<(pushConst.numDynamicObjects/2-1)){
    dynConstNum = pushConst.numDynamicObjects/2;
  }else{
    dynConstNum = pushConst.numDynamicObjects/2 - 1 + (pushConst.numDynamicObjects%2);
  }

  switch(dyn.type){
    default:
    case PrimType::Sphere:
    for(int sub_step = 0; sub_step < 1; sub_step++){
      max_corr_length = max(gravity_length_dts,length(dyn.poss_diff));
      integrateLinear(dyn);

      for(int i = 0; i<pushConst.numDynamicObjects; i++){
        if(i != dObjIdx)
          solveBodyCollissionConstrain(dyn,dynamic_objects[i],bodyColissionCompliance);
      }
      dyn.position.xyz += dyn.corr_delta.xyz;
      dyn.corr_delta = float4(0.0);
      solveStaticCollisionSphereConstraint(dyn,colissionCompliance);
      solveDragConstraintSphere(dyn,dragCompliance);
      
      updateVelocitiesLinear(dyn);
    }
    break;

    case PrimType::Box:
    for(int sub_step = 0; sub_step < pushConst.pyp.sub_steps; sub_step++){
      max_corr_length = length(dyn.poss_diff);
      integrate(dyn);

      solveStaticCollisionCubeConstraint(dyn,colissionCompliance);
      solveDragConstraint(dyn,dragCompliance);

      updateVelocities(dyn);
    }
    break;
  }
    
  dynamic_objects[dObjIdx] = dyn;
}