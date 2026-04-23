
#include "shaderio.h"               // Shared definitions with CPU
#include "utils/descriptors.slang"  // External memory definitions

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  uint x = dynamic_objects[0];
  dynamic_objects[0] = uint(abs(noiseImage[int2(x)%NOISE_TEX_SIZE].x*100.0));
  printf("=====GPU: %u frame %i\n",x,pushConst.frameCount);

}