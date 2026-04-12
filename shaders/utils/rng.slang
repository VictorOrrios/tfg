#pragma once

uint pcgHash(uint x) {
  x = x * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return x;
}

float3 hash3(float3 p){
  p = frac(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return frac((p.xxy + p.yzz) * p.zyx);
}