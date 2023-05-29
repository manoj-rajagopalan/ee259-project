#include <vector>

#include <optix.h>
#include <cuda_runtime.h>

#include "my_cuda_device_unique_ptr.hpp"

namespace manojr {

template<typename T>
class OptixSbtRecord
{
  public:

    void packHeader(OptixProgramGroup optixProgramGroup)
    {
        OPTIX_CHECK( optixSbtRecordPackHeader(optixProgramGroup, (void*) this) );
    }

    operator T& () {
        return payload_;
    }

private:
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header_[OPTIX_SBT_RECORD_HEADER_SIZE];
    T payload_;

};


} // namespace manojr
