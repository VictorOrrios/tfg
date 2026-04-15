
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

  if(any(fragCoord >= iResolution))
    return;

  float ao_result = applyBilateralBlurPass(
    fragCoord,false,iResolution,
    pushConst.lp.aoTexelSize,aoSample,aoScratchSample);

  aoImage[fragCoord] = ao_result;

  float shadow_result = applyBilateralBlurPass(
    fragCoord,false,iResolution,
    pushConst.lp.shadowTexelSize,shadowSample,shadowScratchSample);

  shadowImage[fragCoord] = shadow_result;
}