#include <vector>

#include <cuda_runtime.h>

#include "my_cuda_device_unique_ptr.hpp"

namespace manojr
{

class XPU_Base
{
  public:
    bool isOnCuDevice() const {
        return bool(ptr_);
    }

    CUdevicePtr devicePtr() {
        return ptr_.get();
    }

  protected:
    XPU_Base() = default;

    CudaDeviceUniquePtr ptr_;
}

// fwd decl
template<typename T>
class XPU; 

template<typename T>
class XPU : public T, public XPU_Base
{
  public:

    using T::T; // inherit all constructors

    void copyToCuDevice()
    {
        size_t numBytes = sizeof(T);
        if(!ptr_) {
            ptr_ = CudaDeviceUniquePtr(numBytes);
        }
        CUDA_CHECK( cudaMemcpy(ptr_.get(), (void*) (T*) this, numBytes, cudaMemcpyHostToDevice) );
        isOnDevice_ = true;
    }

    bool isOnDevice() const {
        return isOnDevice_;
    }

  private:
    T data_;
    bool isOnDevice_ = false;
};

template<typename T>
class XPU<std::vector<T>>: public std::vector<T>, public XPU_Base
{
  public:
    using std::vector<T>::vector; // inherit all CTORs

    void copyToCuDevice()
    {
        size_t numBytes = this->size() * sizeof(T);
        if(!ptr_ || this->size() != prevSize_) {
            ptr_ = CudaDeviceUniquePtr(numBytes);
            prevSize_ = this->size();
        }
        assert(ptr_);
        CUDA_CHECK(
            cudaMemcpy(ptr_.get(), (void*) this->data, numBytes, cudaMemcpyHostToDevice);
        );
        isOnDevice_ = true;
    }

    bool isOnDevice() const {
        return isOnDevice_;
    }

  private:
    size_t prevSize_ = 0;
    bool isOnDevice_ = false;
};

} // namespace manojr
