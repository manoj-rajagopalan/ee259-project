#pragma once

#include <optix.h>

namespace manojr {

struct Frame
{
    int32_t x_resolution; // pixels, even
    int32_t y_resolution; // pixels, even
    float width; // m
    float height; // m
};

struct Camera
{
    // All items in world-space
    float3 origin; // m
    float3 xAxis; // unit vec
    float3 yAxis; // unit vec
    float3 zAxis; // unit vec, negative of "look" direction
    float3 screenDimensions; // m, z-axis is distance to screen from origin
};


struct OptixLaunchParams
{
    float *pointCloud;
    int numThreads_x;
    int numThreads_y;
    int atomicNumPoints;
    OptixTraversableHandle gas_handle;
    Camera camera;
    Frame frame;
};

}
