#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

#include <cuda_runtime.h>

#include "assimp/scene.h"
#include "assimp/matrix4x4.h"

#include "AsyncCudaBuffer.hpp"
#include "ExecutionCursor.hpp"
#include "transmitter.hpp"

namespace manojr {

class AssimpOptixRayTracer
{
  public:
    AssimpOptixRayTracer(std::mutex& rayTracingMutex,
	                       std::condition_variable& rayTracingConditionVariable,
							           int& rayTracingCommand);
    ~AssimpOptixRayTracer();

    /// @brief Call before doing anything else.
    void registerLoggingFunctions(std::function<void(std::string const&)> logInfo,
                                  std::function<void(std::string const&)> logError);

    void initialize(); ///< CUDA and OptiX. Call after setting logging functions.

    void setScene(aiScene const& scene);
    void setSceneTransform(aiMatrix4x4 const& modelToWorldTransform);
    
    void setTransmitter(Transmitter const& transmitter);
    void setTransmitterTransform(aiMatrix4x4 const& transmitterToWorldTransform /* 3x3 rotation matrix only! */);

    /// Caller takes ownership of result.
    aiScene* runRayTracing();

    void eventLoop();

    void logOptixMessage(unsigned int level, const char *tag, const char *msg);

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
      
      void* makeRaygenSbtRecord_(ExecutionCursor whereInProgram);
      std::pair<void*, unsigned int> makeMissSbtRecord_(ExecutionCursor whereInProgram);
      std::pair<void*, unsigned int> makeHitGroupSbtRecord_(ExecutionCursor whereInProgram);
      void makeOptixShaderBindingTable_(ExecutionCursor whereInProgram);

      // Multi-threaded communication with parent
      std::mutex& rayTracingMutex_;
      std::condition_variable& rayTracingConditionVariable_;
      int& rayTracingCommand_;

      // Callback functors which abstract the output streams for logging.
      std::function<void(std::string const&)> logInfo_;
      std::function<void(std::string const&)> logError_;
    
      int executionFrame_;

      // Scene data
      AsyncCudaBuffer modelVertexBufferOnGpu_;
      AsyncCudaBuffer worldVertexBufferOnGpu_;
      AsyncCudaBuffer indexBufferOnGpu_;
      AsyncCudaBuffer rayHitVerticesOnGpu_; // world coords

      // Transmitter data
      Transmitter transmitter_;
      aiMatrix4x4 transmitterToWorldTransform_;

      // CUDA data
      CUcontext cuCtx_ = 0;
      cudaStream_t cudaStream_;

      // OptiX stuff
      OptixDeviceContext optixDeviceContext_ = nullptr;

      // OptiX geometry acceleration structures (mesh + BVH etc.)
      OptixTraversableHandle gasHandle_;
      AsyncCudaBuffer gasBuild_;

      // OptiX ray-tracing code structures
      OptixModule optixModule_;
      OptixPipeline optixPipeline_;
      OptixProgramGroup raygenProgramGroup_;
      OptixProgramGroup hitGroupProgramGroup_;
      OptixProgramGroup missProgramGroup_;

      // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_shader_binding_table.html
      OptixShaderBindingTable optixSbt_;

}; // class AssimpOptixRayTracer

} // namespace manojr
