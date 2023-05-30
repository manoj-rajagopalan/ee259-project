#include <optix_device.h>

#include "LaunchParams.hpp"

extern "C" __constant__ LaunchParams launchParams;

extern "C" __global__ void __closesthit__xpoint()
{
    float3 ABC[3]; // triangle vertices
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const unsigned int sbtGasIndex = optixGetSbtGASIndex();
    const float rayTime = optixGetRayTime();
    optixGetTriangleVertexData(launchParams.gas_handle,
                               primitiveIndex,
                               sbtGasIndex,
                               rayTime,
                               ABC);
    const float2 bary = optixGetTriangleBarycentrics();
    const float alpha = bary[0];
    const float beta = bary[1];
    const float gamma = 1.0f - alpha - beta;
    const float3 xpoint = alpha*ABC[0] + beta*ABC[1] * gamma*ABC[0];
    const int pointIndex = atomicAdd(&launchParams.atomicNumPoints, 1);
    float *writeDst = launchParams.pointCloud + 3*pointIndex;
    writeDst[0] = xpoint[0];
    writeDst[1] = xpoint[1];
    writeDst[2] = xpoint[2];
}

extern "C" __global__ void __miss__noop()
{
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 launchIdx = optixGetLaunchIndex();
    const uint3 launchDim = optixGetLaunchDimensions();
    float2 frac = (launchIdx.xy + float2(0.5f,0.5f)) / launchDim.xy; // (0,1) range
    frac = 2*frac - float2(1.0f,1.0f); // (-1,1) range
    float3 raydir = launchParams.screenDimensions[2] * launchParams.rayEmitter.zAxis
                  + (frac.x * 0.5f*launchParams.screenDimensions[0]) * xAxis
                  + (frac.y * 0.5f*launchParams.screenDimensions[1]) * yAxis;
    optixTrace(launchParams.gas_handle,
               launchParams.rayEmitter.origin,
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