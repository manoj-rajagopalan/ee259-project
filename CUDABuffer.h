// Manoj Rajagopalan's (manojr) contributions:
// - Added per-CUDA-stream logic to this file.
//   This allows asynchronous memcpy-s into GPU.
//   Earlier, all operations worked on the default CUDA stream (#0).      

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

#include "cuda_optix_sentinel.h" // manojr

// common std stuff
#include <vector>
#include <cassert>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  /*! simple wrapper for creating, and managing a device-side CUDA
      buffer */
  struct CUDABuffer {

    ~CUDABuffer() {
      if(d_ptr) {
        free();
      }
    }

    inline CUdeviceptr d_pointer() const
    { return (CUdeviceptr)d_ptr; }

    inline CUdeviceptr& d_pointer_ref() {
      return *((CUdeviceptr*) d_ptr);
    }

    //! re-size buffer to given number of bytes
    void resize(size_t sizeInBytes, cudaStream_t cudaStream = 0)
    {
      if(sizeInBytes != this->sizeInBytes) {
        if (d_ptr) free();
        alloc(sizeInBytes, cudaStream);
      }
    }
    
    //! allocate to given number of bytes
    void alloc(size_t sizeInBytes, cudaStream_t cudaStream = 0)
    {
      assert(d_ptr == nullptr);
      this->sizeInBytes = sizeInBytes;
      CUDA_CHECK( Malloc( (void**)&d_ptr, sizeInBytes, cudaStream) );
    }

    //! free allocated memory
    void free()
    {
      CUDA_CHECK(Free(d_ptr));
      d_ptr = nullptr;
      sizeInBytes = 0;
      numElements = 0;
      sizeOfElement = 0;
    }

    // manojr
    void* detach() // ownership to another resource
    {
      void *result = d_ptr;
      d_ptr = nullptr;
      return result;
    }

    template<typename T>
    void allocAndUpload(const std::vector<T> &vt, cudaStream_t cudaStream = 0)
    {
      alloc(vt.size()*sizeof(T), cudaStream);
      upload((const T*)vt.data(), vt.size(), cudaStream);
    }
    
    template<typename T>
    void allocAndUpload(T const *t, size_t count, cudaStream_t cudaStream = 0)
    {
      alloc(count*sizeof(T), cudaStream);
      upload(t, count, cudaStream);
    }

    template<typename T>
    void upload(const T *t, size_t count, cudaStream_t cudaStream = 0)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      this->numElements = count;
      this->sizeOfElement = sizeof(T);
      CUDA_CHECK (Memcpy(d_ptr,
                         (void *)t,
                         count*sizeof(T),
                         cudaMemcpyHostToDevice,
                         cudaStream));
    }
    
    template<typename T>
    void download(T *t, size_t count, cudaStream_t cudaStream = 0)
    {
      assert(d_ptr != nullptr);
      assert(numElements >= count);
      assert(sizeOfElement == sizeof(T));
      assert(sizeInBytes >= count*sizeof(T));
      CUDA_CHECK( Memcpy((void *)t,
                         d_ptr,
                         count*sizeof(T),
                         cudaMemcpyDeviceToHost,
                         cudaStream));
    }
    
    template<typename T>
    void download(std::vector<T>& v, cudaStream_t cudaStream = 0)
    {
      v.resize(this->numElements);
      this->download(v.data(), v.size(), cudaStream);
    }

    size_t numElements{ 0 };
    size_t sizeOfElement{ 0 };
    size_t sizeInBytes { 0 };
    void  *d_ptr { nullptr };
  };

} // ::osc
