#pragma once

#include <optix.h>

#include "transmitter.hpp"

namespace manojr {

struct OptixLaunchParams
{
    void *pointCloud; // array of float3
    OptixTraversableHandle gasHandle;
    Transmitter transmitter;
    // TODO: transmitter orientation

    // CPU must set to 0 before invoking ray-tracing.
    // GPU threads will atomically increment for each collision.
    // Upon return to CPU, we get number of rays that intersected scene.
    // Max possible value is (sensor.numRays_x * sensor.numRays_y).
    int gpuAtomicNumHits;
};

} // namespace manojr
