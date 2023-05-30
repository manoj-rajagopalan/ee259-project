#pragma once

#include <optix.h>

namespace manojr {

struct RayEmitter
{
    // All items in world-space
    float3 origin;
    float3 xAxis;
    float3 yAxis;
    float3 zAxis; // look direction
    float3 screenDimensions; // z-axis is distance to screen from origin
};


struct OptixLaunchParams
{
    float *pointCloud;
    int numThreads_x;
    int numThreads_y;
    int atomicNumPoints;
    OptixTraversableHandle gas_handle;
    RayEmitter rayEmitter;
};

}
