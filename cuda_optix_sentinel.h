// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

// optix 7
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)						                    \
    do {									                                              \
      cudaError_t res = cuda##call;                                      \
      if (res != cudaSuccess) {                                          \
        std::stringstream txt;                                          \
        txt << "cuda" << #call << " error #" << res                     \
            << " (" << cudaGetErrorName(res) << ')'                      \
            << " at " << __FILE__ << ':' << __LINE__ << " : "           \
            << cudaGetErrorString(res);                                 \
        throw std::runtime_error(txt.str());                            \
      }                                                                 \
    } while(false)

#define CUDA_CHECK_NOEXCEPT(call)                                       \
    do {									                                              \
      cuda##call;                                                       \
    } while(false)

#define OPTIX_CHECK( call )                                             \
  do {                                                                  \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        std::stringstream txt;                                          \
        txt << #call << " error #" << res                               \
            << " (" << optixGetErrorName(res) << ')'                     \
            << " at " << __FILE__ << ':' << __LINE__ << " : "           \
            << optixGetErrorString(res);                                \
        throw std::runtime_error(s.str());                              \
      }                                                                 \
  } while(false)

#define CUDA_STREAM_SYNC_CHECK() \
  do { \
    cudaStreamSynchronize(); \
    cudaError_t const res = cudaGetLastError(); \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        std::stringstream txt;                                          \
        txt << " CUDA Error upon stream-sync #" << res                  \
            << " (" << optixGetErrorName(res) << ')'                     \
            << " at " << __FILE__ << ':' << __LINE__ << " : "           \
            << optixGetErrorString(res);                                \
        throw std::runtime_error(s.str());                              \
      }                                                                 \
  } while(false)