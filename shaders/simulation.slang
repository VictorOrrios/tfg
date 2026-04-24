
#include "shaderio.h"               // Shared definitions with CPU
#include "utils/descriptors.slang"  // External memory definitions
#include "utils/quat.slang"         // Quaternion support

void integrate(inout DynamicObject dyn){
  /*
  pyp.prev_position = gp.position;
  pyp.vel += gravity * dt;
  gp.position += pyp.vel * dt;

  // Angular motion
  pyp.prev_rotation = gp.rotation;
  glm::quat d_rot = glm::quat(
    0,
    pyp.omega.x,
    pyp.omega.y,
    pyp.omega.z
  );
  d_rot *= gp.rotation;
  gp.rotation += 0.5f * dt * d_rot;
  gp.rotation = glm::normalize(gp.rotation);
  pyp.inv_rotation = glm::inverse(gp.rotation);
  */
  // Linear motion
  dyn.prev_position = dyn.position;
  dyn.vel.xyz += pushConst.pyp.gravity * pushConst.pyp.dts;
  dyn.position += dyn.vel * pushConst.pyp.dts;

  // Angular motion
  dyn.prev_rotation = dyn.rotation;


}

[shader("compute")]
[numthreads(WORKGROUP_SIZE_1D, 1, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  if(threadIdx.x >= pushConst.numDynamicObjects)
    return;

  int dObjIdx = threadIdx.x;
  DynamicObject dyn = dynamic_objects[dObjIdx];


  printf("Processing dyn obj %i\n",dObjIdx);
}