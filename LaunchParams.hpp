/// Data structures for CPU-GPU communication.

#pragma once

#include <optix.h>

#include "transmitter.hpp"


struct OptixLaunchParams
{
    OptixTraversableHandle gasHandle;
    manojr::Transmitter transmitter;
};

template<typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) ShaderBindingTableRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT)
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData { /* TODO */};
using RayGenSbtRecord = ShaderBindingTableRecord<RayGenData>;

struct MissData { /* TODO */ };
using MissSbtRecord = ShaderBindingTableRecord<MissData>;

struct HitGroupData {
    float3 *pointCloud = nullptr; // array of float3
    // CPU must set to 0 before invoking ray-tracing.
    // GPU threads will atomically increment for each collision.
    // Upon return to CPU, we get number of rays that intersected scene.
    // Max possible value is (sensor.numRays_x * sensor.numRays_y).
    int32_t *atomicNumHits = nullptr;
 };
using HitGroupSbtRecord = ShaderBindingTableRecord<HitGroupData>;

