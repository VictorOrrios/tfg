
#include "shaderio.h"               // Shared definitions with CPU
#include "utils/bilateral.slang"    // Bilateral blur functions
#include "utils/descriptors.slang"  // External memory definitions

[shader("compute")]
[numthreads(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  int2 fragCoord = threadIdx.xy;
  int2 iResolution;
  outImage.GetDimensions(iResolution.x, iResolution.y);

  float result = applyBilateralBlurPass(fragCoord,iResolution,true);

  aoScratchImage[fragCoord] = result;
}