#pragma once

#include "cuda_optix_sentinel.h"

// common std stuff
#include <vector>
#include <cassert>

namespace manojr {

  struct AsyncCudaBuffer {

    AsyncCudaBuffer() = delete;

    explicit AsyncCudaBuffer(cudaStream_t cudaStream) {
      // this->cudaStream = cudaStream;
      this->cudaStream = 0; // default stream
    }

    AsyncCudaBuffer(const AsyncCudaBuffer&) = delete; // non-copyable
    AsyncCudaBuffer(AsyncCudaBuffer &&) = delete; // non-movable (ok in theory but enable when needed, to prevent accidents)

    ~AsyncCudaBuffer() {
      if(d_ptr) {
        asyncFree();
        //? CUDA_STREAM_SYNC_CHECK(cudaStream);
      }
    }

    void* d_pointer() const { return d_ptr; }

    //! re-size buffer to given number of bytes
    void asyncResize(size_t sizeInBytes)
    {
      if(sizeInBytes != this->sizeInBytes) {
        if (d_ptr) asyncFree();
        asyncAlloc(sizeInBytes);
      }
    }
    
    //! allocate to given number of bytes
    void asyncAlloc(size_t sizeInBytes)
    {
      assert(d_ptr == nullptr);
      this->sizeInBytes = sizeInBytes;
      // CUDA_CHECK( MallocAsync( (void**)&d_ptr, sizeInBytes, cudaStream) );
      CUDA_CHECK( Malloc( (void**)&d_ptr, sizeInBytes) );
    }

    //! asyncFree allocated memory
    void asyncFree()
    {
      // CUDA_CHECK(FreeAsync(d_ptr, cudaStream));
      CUDA_CHECK(Free(d_ptr));
      d_ptr = nullptr;
      sizeInBytes = 0;
      numElements = 0;
      sizeOfElement = 0;
    }

    // manojr
    void* detach() // transfer ownership to another resource
    {
      void *result = d_ptr;
      d_ptr = nullptr;
      sizeInBytes = 0;
      sizeOfElement = 0;
      numElements = 0;
      return result;
    }

    template<typename T>
    void asyncAllocAndUpload(const std::vector<T> &vt)
    {
      asyncResize(vt.size()*sizeof(T));
      asyncUpload((const T*)vt.data(), vt.size());
    }
    
    template<typename T>
    void asyncAllocAndUpload(T const *t, size_t count)
    {
      asyncResize(count*sizeof(T));
      asyncUpload(t, count);
    }

    template<typename T>
    void asyncUpload(const T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      this->numElements = count;
      this->sizeOfElement = sizeof(T);
      // CUDA_CHECK (MemcpyAsync(d_ptr,
      //                         (void *)t,
      //                         count*sizeof(T),
      //                         cudaMemcpyHostToDevice,
      //                         cudaStream));
      CUDA_CHECK (Memcpy(d_ptr,
                        (void *)t,
                        count*sizeof(T),
                        cudaMemcpyHostToDevice));
    }
    
    template<typename T>
    void asyncDownload(T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(numElements >= count);
      assert(sizeOfElement == sizeof(T));
      assert(sizeInBytes >= count*sizeof(T));
      // CUDA_CHECK( MemcpyAsync((void *)t,
      //                         d_ptr,
      //                         count*sizeof(T),
      //                         cudaMemcpyDeviceToHost,
      //                         cudaStream));
      CUDA_CHECK( Memcpy((void *)t,
                          d_ptr,
                          count*sizeof(T),
                          cudaMemcpyDeviceToHost));
    }
    
    template<typename T>
    void asyncDownload(std::vector<T>& v)
    {
      v.asyncResize(this->numElements);
      this->asyncDownload(v.data(), v.size());
    }

    void sync()
    {
      // CUDA_STREAM_SYNC_CHECK(cudaStream);
      CUDA_CHECK( DeviceSynchronize() );
    }

    cudaStream_t cudaStream;
    size_t numElements{ 0 };
    size_t sizeOfElement{ 0 };
    size_t sizeInBytes { 0 };
    void *d_ptr { nullptr };
  };

} // namespace manojr
