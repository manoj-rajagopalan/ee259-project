#include <array>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "CUDABuffer.h"
#include "transmitter.hpp"

namespace manojr {

class AssimpOptixRayTracer
{
  public:
    AssimpOptixRayTracer(std::function<void(std::string const&)> logInfo,
                         std::function<void(std::string const&)> logError);
    ~AssimpOptixRayTracer();

    void setScene(aiScene const& scene);
    void setSceneTransform(aiMatrix4D& modelToWorldTransform);
    
    void setTransmitter(manojr::Transmitter const& transmitter) {
      transmitter_ = transmitter;
    }
    void setTransmitterTransform(aiMatrix4D const& transmitterToWorldTransform);

    aiScene runRayTracing();

  private:

      void initializeCuda_(ExecutionCursor whereInProgram);
      void finalizeCuda_(ExecutionCursor whereInProgram);
 
      void initializeOptix_(ExecutionCursor whereInProgram);
      void finalizeOptix_(ExecutionCursor whereInProgram);
 
      void createOptixDeviceContext_(ExecutionCursor whereInProgram);

      /// Set gasBuild_ and gasHandle_
      void buildOptixAccelerationStructures_(ExecutionCursor whereInProgram);
      
      std::string loadOptixModuleSourceCode_(ExecutionCursor whereInProgram);
      void makeOptixModule_(OptixPipelineCompileOptions optixPipelineCompileOptions,
                            char *optixLogBuffer,
                            size_t optixLogBufferSize,
                            ExecutionCursor whereInProgram);
      OptixPipelineCompileOptions makeOptixPipelineCompileOptions_();
      OptixPipelineLinkOptions makeOptixPipelineLinkOptions_();
      void makeRaygenProgramGroup_(char *optixLogBuffer,
                                   size_t optixLogBufferSize,
                                   ExecutionCursor whereInProgram);
      void makeHitGroupProgramGroup_(char *optixLogBuffer,
                                     size_t optixLogBufferSize,
                                     ExecutionCursor whereInProgram);
      void makeMissProgramGroup_(char *optixLogBuffer,
                                 size_t optixLogBufferSize,
                                 ExecutionCursor whereInProgram);
      void buildOptixPipeline_(ExecutionCursor whereInProgram);
      
      CUdeviceptr makeRaygenSbtRecord_(ExecutionCursor whereInProgram);
      std::pair<CUdeviceptr, unsigned int> makeMissSbtRecord_(ExecutionCursor whereInProgram);
      std::pair<CUdeviceptr, unsigned int> makeHitGroupSbtRecord_(ExecutionCursor whereInProgram);
      void makeOptixShaderBindingTable_(ExecutionCursor whereInProgram);

      // Callback functors which abstract the output streams for logging.
      std::function<void(std::string const&)> logInfo_;
      std::function<void(std::string const&)> logError_;
      
      int executionFrame_;

      // Scene data
      osc::CUDABuffer modelVertexBufferOnGpu_;
      osc::CUDABuffer worldVertexBufferOnGpu_;
      osc::CUDABuffer indexBufferOnGpu_;
      osc::CUDABuffer rayHitVerticesOnGpu_; // world coords

      // Transmitter data
      manojr::Transmitter transmitter_;
      aiMatrix4D transmitterToWorldTransform_;

      // CUDA data
      CUcontext cuCtx_ = 0;
      cudaStream_t cudaStream_;

      // OptiX stuff
      OptixDeviceContext optixDeviceContext_ = nullptr;

      // OptiX geometry acceleration structures (mesh + BVH etc.)
      OptixTraversableHandle gasHandle_;
      osc::CUDABuffer gasBuild_;

      // OptiX ray-tracing code structures
      OptixModule optixModule_;
      OptixPipeline optixPipeline_;
      OptixProgramGroup raygenProgramGroup_;
      OptixProgramGroup hitGroupProgramGroup_;
      OptixProgramGroup missProgramGroup_;

      // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_shader_binding_table.html
      OptixShaderBindingTable optixSbt_;

}; // class AssimpOptixRayTracer
