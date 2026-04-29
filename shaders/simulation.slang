
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

  // 🔹 ejes
  float3( 1, 0, 0),
  float3(-1, 0, 0),
  float3( 0, 1, 0),
  float3( 0,-1, 0),
  float3( 0, 0, 1),
  float3( 0, 0,-1),

  // 🔹 diagonales
  normalize(float3( 1, 1, 1)),
  normalize(float3(-1, 1, 1)),
  normalize(float3( 1,-1, 1)),
  normalize(float3(-1,-1, 1)),
  normalize(float3( 1, 1,-1)),
  normalize(float3(-1, 1,-1)),
  normalize(float3( 1,-1,-1)),
  normalize(float3(-1,-1,-1))
};

void integrate(inout DynamicObject dyn){
  /* 
  float3 pos_diff = dyn.position.xyz - dyn.prev_position.xyz;
  printf("Integrate: pos_diff %f,%f,%f len %f\n",
    pos_diff.x,pos_diff.y,pos_diff.z,length(pos_diff)
  );
 */
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
/* 
  printf("lambda %f | alpha %f | dt_m2 %f | dts %f | normal len %f | w %f \n",
    lambda, alpha, dt_m2, pushConst.pyp.dts, length(normal),w
  );
 */
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
static float prev_V = -1;
static float prev_pos_diff;

void solveFut(
  inout DynamicObject dyn,
  float compliance)
{
  float3 center = dyn.position.xyz;
  float radius = dyn.scale * 0.5;
  float nearRange = 4.0*dyn.scale;

  float3 vel = dyn.vel.xyz - pushConst.pyp.gravity;
  float3 fut_diff = dyn.poss_diff.xyz;
  float3 fut_pos = center + fut_diff;
  float3 normal_f = evalNormalStatic(fut_pos,nearRange);
  float fut_len = max(0,length(fut_diff));
  bool fut_threshold = fut_len > radius*0.00;
  if(fut_threshold){
    float3 fut_dir = normalize(vel);
    float3 fut_sp = center + fut_dir*radius;
    float futSV = mapStatic(fut_sp,nearRange);
    float C_fut = clamp(-futSV,0,length(fut_diff));
    if(C_fut > 0.0){
      float3 fur_corr = -fut_dir*C_fut;
      float applyRet = -applyCorrection(
        dyn, dyn, false,
        compliance, fur_corr,
        fut_sp);
      printf("FUT: applyRet %f | C_fut %f \n",
        applyRet,C_fut
      );
    }
  }
}

void solveFutMed(
  inout DynamicObject dyn,
  float compliance)
{
  float3 center = dyn.position.xyz;
  float radius = dyn.scale * 0.5;
  float nearRange = 4.0*dyn.scale;

  float3 vel = dyn.vel.xyz - pushConst.pyp.gravity;
  float3 fut_diff = dyn.poss_diff.xyz;
  float3 fut_pos = center + fut_diff;
  float3 normal_f = evalNormalStatic(fut_pos,nearRange);
  float fut_len = max(0,length(fut_diff));
  bool fut_threshold = fut_len > radius*0.00;
  if(fut_threshold){
    float3 fut_dir = normalize(vel);
    float3 ldir = lerp(normal_f,-fut_dir,0.5);
    float3 fut_sp = center - ldir*radius;
    float futSV = mapStatic(fut_sp,nearRange);
    float C_fut = clamp(-futSV,0,length(fut_diff));
    if(C_fut > 0.0){
      float3 fur_corr = ldir*C_fut;
      float applyRet = -applyCorrection(
        dyn, dyn, false,
        compliance, fur_corr,
        fut_sp);
      printf("FUTM: applyRet %f | C_fut %f \n",
        applyRet,C_fut
      );
    }
  }
}

void solveStaticCollisionConstraint(
  inout DynamicObject dyn,
  float compliance)
{
  solveFut(dyn,compliance);
  solveFutMed(dyn,compliance);

  float3 center = dyn.position.xyz;
  float radius = dyn.scale * 0.5;
  float nearRange = 4.0*dyn.scale;

  float fut_len = length(dyn.poss_diff);

  center = dyn.position.xyz;
  float centerV = mapStatic(center,nearRange);
  const float h = 0.001;
  float3 grad_c = evalGradStatic(center,h,nearRange);
  float gradLen = length(grad_c);
  float3 normal_c = evalNormalStatic(center,nearRange);
  float C_center = radius-centerV;

  float n_diff = any(prev_n!=0.0)? dot(prev_n,normal_c): -2;
  prev_n = normal_c;

  //if(n_diff<0.9) return;

  float3 sp = center - normal_c*radius;
  float surfaceV = mapStatic(sp,nearRange);
  float3 grad_s = evalGradStatic(sp,h,nearRange);
  float3 normal_s = normalize(grad_s);
  float C_surface = -surfaceV;
/* 
  if(surfaceV >= 0){
    prev_V = surfaceV; 
    return;
  }
 */
  float depth = 0.0, step = radius/2.0, mediumV;
  float3 mp;
  for(int i = 2; i<50; i++){
    mp = center-normal_c*depth;
    mediumV = mapStatic(mp,nearRange);
    if(abs(mediumV) < radius/100.0) break;
    depth += step*sign(mediumV);
    step /= 2;
  }

  depth = clamp(depth,0,radius);
  
  float C_medium = radius-depth;

  float C = C_surface;

  if(C <= 0.0){
    prev_V = surfaceV; 
    return;
  }

  float maxCorr = min(radius * 0.5, fut_len);
  float corrPercent = C/maxCorr;
  C = clamp(C, 0.0, maxCorr);

  float3 corr = normal_c * C;
 
  float applyRet;
  applyRet = -applyCorrection(
  dyn, dyn, false,
  compliance, corr,
  sp);

/* 
  printf("COL: applyRet %f | centerV: %f | surfaceV: %f | medium depth: %f | C %f -> %f/%f | n_diff: %f | gradLen %f | prev_pos_diff %f | frame %i | substep %i\n",
    applyRet, centerV, surfaceV, depth, prev_V, C, maxCorr, n_diff, gradLen, prev_pos_diff, pushConst.frameCount, substep);
   
   */
  if(corrPercent>1.0 && false){
    printf("===== WARN: Max correction reached: %f% %i =======\n",corrPercent*100.0,pushConst.frameCount);
  }

  prev_pos_diff = length(dyn.prev_position-dyn.position);
  prev_V = surfaceV; 
  
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
      float C = clamp(depth,0.0,length(dyn.poss_diff));
      float3 corr = normal*C;
      applyCorrection(
      dyn, dyn, false,
      compliance, corr,
      global_p);
    }
  }
}

void solveStaticCollisionSphereConstraint(
  inout DynamicObject dyn,
  float compliance)
{
  const bool print = pushConst.frameCount%1 == 0;
  float3 center = dyn.position.xyz;
  float radius = dyn.scale * 0.5;
  float nearRange = 4.0*dyn.scale;

  float centerV = mapStatic(center,nearRange);
  if(centerV>=radius){
    if(print) printf("Exit, center\n");
    return;
  }

  float3 normal_c = evalNormalStatic(center,nearRange);
  if(print)
  printf("normal_c %f,%f,%f\n",
    normal_c.x,normal_c.y,normal_c.z
  );

  float3 sp = center-normal_c*radius;
  float surfaceV = mapStatic(sp,nearRange);
  if(surfaceV>=0.0){
    if(print) printf("Exit, surface sp %f,%f,%f surfaceV %f\n",
      sp.x,sp.y,sp.z,surfaceV
    );
    return;
  }
  float3 normal_s = evalNormalStatic(sp,nearRange);
  if(print)
  printf("normal_s %f,%f,%f\n",
    normal_s.x,normal_s.y,normal_s.z
  );
  
  float depth = -surfaceV;
  /* 
  for(int s = 0; s < 10; s++){
    float3 p = sp + normal_s * depth;
    float sd = mapStatic(p, nearRange);
    if(sd >= 0.0 && false){
      if(print) printf("Break @ %i\n",s);
      break;
    }
    depth -= sd;
  }
 */
  float C = clamp(depth,0.0,max(9.8*pushConst.pyp.dts,length(dyn.poss_diff)));
  if(print) printf("C %f surfaceV %f depth %f\n",C,surfaceV,depth);
  float3 corr = normal_s*C;
  applyCorrection(
  dyn, dyn, false,
  compliance, corr,
  sp);
}

void solveStaticCollisionSpherePointsConstraint(
  inout DynamicObject dyn,
  float compliance)
{
  float3 center = dyn.position.xyz;
  float radius = dyn.scale * 0.5;
  float nearRange = 4.0*dyn.scale;

  float centerV = mapStatic(center,nearRange);
  if(centerV>=radius) return;

  for(int i = 0; i<sphere_p_size; i++){
    float3 local_p = sphere_p[i]*dyn.scale/2.0;
    float3 global_p = local2World(dyn,local_p);
    float sdV = mapStatic(global_p,nearRange);
    if(sdV<0.0){
      float3 normal = evalNormalStatic(global_p,nearRange);
      float depth = -sdV;
      /* 
      for(int s = 0; s<10; s++){
        float sdIV = mapStatic(global_p+normal*depth,nearRange);
        depth -= sdIV;
        if(sdIV>=0.0) break;
      }
 */
      float C = clamp(depth,0.0,max(9.8*pushConst.pyp.dts,length(dyn.poss_diff)));
      float3 corr = normal*C;
      applyCorrection(
      dyn, dyn, false,
      compliance, corr,
      global_p);
    }
  }
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

  int dynConstNum;
  if(dObjIdx<(pushConst.numDynamicObjects/2-1)){
    dynConstNum = pushConst.numDynamicObjects/2;
  }else{
    dynConstNum = pushConst.numDynamicObjects/2 - 1 + (pushConst.numDynamicObjects%2);
  }

  /* 
  printf("Processing dyn obj %i static obj %i pos %f,%f,%f\n",
    dObjIdx,dyn.index,
    dyn.position.x,dyn.position.y,dyn.position.z);
  */
  for(int sub_step = 0; sub_step < SIM_NUM_SUBSTEPS; sub_step++){
    substep = sub_step;
    integrate(dyn);

    
    //solveStaticCollisionConstraint(dyn,0.0);
    //solveStaticCollisionCubeConstraint(dyn,0.0);
    solveStaticCollisionSphereConstraint(dyn,0.0);
    //solveStaticCollisionSpherePointsConstraint(dyn,0.0);

    //solveDragConstraint(dyn,0.005);
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