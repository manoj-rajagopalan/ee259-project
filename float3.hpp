#pragma once

#include <cmath>

// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#compilation-phases
#ifndef __CUDACC__
#include "assimp/vector3.h"
typedef aiVector3t<float> float3;
#endif

inline
float3 cross(float3 const& a, float3 const& b)
{
    float3 c;
    c[0] = a[1]*b[2] - b[1]*a[2];
    c[1] = a[2]*b[0] - b[2]*a[0];
    c[2] = a[0]*b[1] - b[0]*a[1];
    return c;
}

inline
float inner(float3 const& a, float3 const& b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline
void normalize(float3& v)
{
    float const len = std::sqrt(inner(v,v));
    if(len != 0.0f) {
        v /= len;
    }
}
