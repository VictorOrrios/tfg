#include "shaderio.h"              // Shared definitions with CPU
#include "utils/common.slang"      // Shared definitions with shaders
#include "utils/bbox.slang"        // AABB helpers
#include "utils/descriptors.slang" // External memory definitions

inline float3x4 createTransform(Bbox bbox) {
  float3 s = bbox.bMax - bbox.bMin;
  float3 pos = bbox.bMin;

  return float3x4(
      s.x, 0,   0,   pos.x,
      0,   s.y, 0,   pos.y,
      0,   0,   s.z, pos.z
  );
}

[shader("compute")]
[numthreads(8, 1, 1)]
void computeMain(uint3 tId: SV_DispatchThreadID) {
  const int id = tId.x;
  const int level = tId.z;

  if (id >= MAX_SCENE_OBJECTS) return;

  if (id >= pushConst.numObjects){
    instancesObj[level][id].mask = 0x00;
    return;
  }

  const float nearRange = level == CLIPMAP_LEVELS ? 0.0 : MAX_BRICK_VALUES[level];

  Bbox bbox = aabbs[id];
  bbox.bMin -= nearRange;
  bbox.bMax += nearRange;

  instancesObj[level][id].mask = 0xFF;
  instancesObj[level][id].transform = createTransform(bbox);
}
