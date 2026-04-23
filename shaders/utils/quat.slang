#pragma once

float4 quatMul(float4 q1, float4 q2){
  return float4(
    q1.w*q2.xyz + q2.w*q1.xyz + cross(q1.xyz, q2.xyz),
    q1.w*q2.w - dot(q1.xyz, q2.xyz)
  );
}

float3 rotateByQuat(float4 q, float3 v){
  float3 t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}