#include <cuda_runtime.h>

namespace manojr
{

class CudaDeviceUniquePtr
{
  public:
    CudaDeviceUniquePtr()
    : cuDevicePtr_(nullptr),
      isValid_(false)
    {}

    // explicit CudaDeviceUniquePtr(CUdevicePtr p)
    // : cuDevicePtr_(p),
    //   isValid_(true)
    // {}

    explicit CudaDeviceUniquePtr(size_t numBytes)
    {
        if(numBytes != numBytes_) {
            CUDA_CHECK(
                cudaMalloc(&cuDevicePtr_, numBytes);
            );
            numBytes_ = numBytes;
        }
        isValid_ = true;
    }

    CudaDeviceUniquePtr(const CudaDeviceUniquePtr&) = delete;

    CudaDeviceUniquePtr(CudaDeviceUniquePtr&& p)
    {
        *this = std::move(p);
    }

    ~CudaDeviceUniquePtr()
    {
        this->release();
    }

    // CudaDeviceUniquePtr& operator= (CUdevicePtr p)
    // {
    //     cuDevicePtr_ = p;
    // }

    CudaDeviceUniquePtr& operator= (const CudaDeviceUniquePtr&) = delete;

    CudaDeviceUniquePtr& operator= (CudaDeviceUniquePtr&& p)
    {
        this->release();
        cuDevicePtr_ = p;
        isValid_ = p.isValid_;
        p.isValid_ = false;
        return *this;
    }

    CUdevicePtr get() const {
        return cuDevicePtr_;
    }

    void release() {
        if(isValid_) {
            cudaFree(cuDevicePtr_);
            isValid_ = false;
        }
    }

    operator bool () const {
        return isValid_;
    }

    size_t memSize() const {
        return numBytes_;
    }

  private:
    CUdevicePtr cuDevicePtr_;
    size_t numBytes_ = 0;
    bool isValid_;
};

}
