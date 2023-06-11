#pragma once

#include "cuda_optix_sentinel.h"

// common std stuff
#include <vector>
#include <cassert>

namespace manojr {

  struct AsyncCudaBuffer {

    AsyncCudaBuffer() : cudaStream{0} {}

    explicit AsyncCudaBuffer(cudaStream_t cudaStream) {
      this->cudaStream = cudaStream;
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
      CUDA_CHECK( MallocAsync( (void**)&d_ptr, sizeInBytes, cudaStream) );
    }

    //! asyncFree allocated memory
    void asyncFree()
    {
      CUDA_CHECK(FreeAsync(d_ptr, cudaStream));
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
      asyncAlloc(vt.size()*sizeof(T));
      asyncUpload((const T*)vt.data(), vt.size());
    }
    
    template<typename T>
    void asyncAllocAndUpload(T const *t, size_t count)
    {
      asyncAlloc(count*sizeof(T));
      asyncUpload(t, count);
    }

    template<typename T>
    void asyncUpload(const T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(sizeInBytes == count*sizeof(T));
      this->numElements = count;
      this->sizeOfElement = sizeof(T);
      CUDA_CHECK (MemcpyAsync(d_ptr,
                              (void *)t,
                              count*sizeof(T),
                              cudaMemcpyHostToDevice,
                              cudaStream));
    }
    
    template<typename T>
    void asyncDownload(T *t, size_t count)
    {
      assert(d_ptr != nullptr);
      assert(numElements >= count);
      assert(sizeOfElement == sizeof(T));
      assert(sizeInBytes >= count*sizeof(T));
      CUDA_CHECK( MemcpyAsync((void *)t,
                              d_ptr,
                              count*sizeof(T),
                              cudaMemcpyDeviceToHost,
                              cudaStream));
    }
    
    template<typename T>
    void asyncDownload(std::vector<T>& v)
    {
      v.asyncResize(this->numElements);
      this->asyncDownload(v.data(), v.size());
    }

    void sync()
    {
      CUDA_STREAM_SYNC_CHECK(cudaStream);
    }

    cudaStream_t cudaStream;
    size_t numElements{ 0 };
    size_t sizeOfElement{ 0 };
    size_t sizeInBytes { 0 };
    void *d_ptr { nullptr };
  };

} // namespace manojr
