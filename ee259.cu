#include <optix_device.h>
#include <vector_functions.hpp>

#include "LaunchParams.hpp"

extern "C" __constant__ manojr::OptixLaunchParams launchParams;

__device__ __forceinline__
float2 scale(float const& c, float2 const& v)
{
    return make_float2(c*v.x, c*v.y);
}

__device__ __forceinline__
float2 operator + (float2 const& a, float2 const& b)
{
    return make_float2(a.x+b.x, a.y+b.y);
}

__device__ __forceinline__
float2 operator - (float2 const& a, float2 const& b)
{
    return make_float2(a.x-b.x, a.y-b.y);
}

__device__ __forceinline__
float2 operator / (float2 const& a, float2 const& b)
{
    return make_float2(a.x/b.x, a.y/b.y);
}

__device__ __forceinline__
float3 scale(float const& c, float3 const& v)
{
    return make_float3(c*v.x, c*v.y, c*v.z);
}

__device__ __forceinline__
float3 operator + (float3 const& a, float3 const& b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ __forceinline__
float3 operator - (float3 const& a, float3 const& b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ __forceinline__
float3 applyBary(float3 ABC[3], float const& alpha, float const& beta, float const& gamma)
{
    float3 v = scale(alpha, ABC[0]) + scale(beta, ABC[1]) + scale(gamma, ABC[2]);

}

extern "C" __global__ void __closesthit__xpoint()
{
    float3 ABC[3]; // triangle vertices
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const unsigned int sbtGasIndex = optixGetSbtGASIndex();
    const float rayTime = optixGetRayTime();
    optixGetTriangleVertexData(launchParams.gasHandle,
                               primitiveIndex,
                               sbtGasIndex,
                               rayTime,
                               ABC);
    const float2 bary = optixGetTriangleBarycentrics();
    const float alpha = bary.x;
    const float beta = bary.y;
    const float gamma = 1.0f - alpha - beta;
    const float3 xpoint = applyBary(ABC, alpha, beta, gamma);
    const int pointIndex = atomicAdd(&launchParams.gpuAtomicNumHits, 1);
    float3 *const writeDst = (float3*)launchParams.pointCloud + pointIndex;
    writeDst->x = xpoint.x;
    writeDst->y = xpoint.y;
    writeDst->z = xpoint.z;
}

extern "C" __global__ void __miss__noop()
{
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 launchIdx = optixGetLaunchIndex();
    const uint3 launchDim = optixGetLaunchDimensions();
    float2 frac = (make_float2(launchIdx.x, launchIdx.y) + make_float2(0.5f,0.5f))
                / make_float2(launchDim.x, launchDim.y); // (0,1) range
    frac = scale(2.0f, frac) - make_float2(1.0f,1.0f); // (-1,1) range

    // Transmitter points in negative Z direction (OpenGL coordinate system)
    float3 rayDir = scale(frac.x * 0.5f*launchParams.transmitter.width, launchParams.transmitter.xUnitVector)
                  + scale(frac.y * 0.5f*launchParams.transmitter.height, launchParams.transmitter.yUnitVector)
                  - scale(launchParams.transmitter.focalLength, launchParams.transmitter.zUnitVector);
    optixTrace(launchParams.gasHandle,
               launchParams.transmitter.position,
               rayDir,
               0.0f, // tmin
               1.0e15f, // tmax
               0.0f, // rayTime
               0xFF, // visibilityMask
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // rayFlags - EXPLORE enabling culling
               0, // sbtOffset
               1, // sbtStride
               0); // missSBTIndex
               // no payload
    
    // EXPLORE - return intersection points via payload and use atomic ops here to write to result buffer
}