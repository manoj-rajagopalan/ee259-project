#pragma once

#include <cstdint>

//- #include "float3.hpp"

namespace manojr
{

/// OpenGL Coordinate system:
/// * x-axis pointing right
/// * y-axis pointing up
/// * z-axis pointing out of scene (negative of "look" direction)
struct Transmitter
{
    // All items in world-space
    float3 position; // m

    float3 xUnitVector; // unit vec
    float3 yUnitVector; // unit vec
    float3 zUnitVector; // unit vec

    float width; // m, distributed symmetrically about x-axis
    float height; // m, distributed symmetrically about y-axis
    float focalLength; // m, distance of focal plane from local origin (along -z axis)

    // Rays distributed equally about 0, for x- and y-axes
    int32_t numRays_x;
    int32_t numRays_y;
};

} // namespace manojr
