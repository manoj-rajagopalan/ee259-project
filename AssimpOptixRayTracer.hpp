#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <QObject>

#include <cuda_runtime.h>

#include "assimp/scene.h"
#include "assimp/matrix4x4.h"

#include "AsyncCudaBuffer.hpp"
#include "ExecutionCursor.hpp"
#include "transmitter.hpp"

namespace manojr {

class AssimpOptixRayTracer : public QObject
{
    Q_OBJECT;

  public:
    AssimpOptixRayTracer(std::mutex& rayTracingCommandMutex,
	                       std::condition_variable& rayTracingCommandConditionVariable,
                         std::mutex& rayTracingResultMutex);
    ~AssimpOptixRayTracer();

    /// @brief Call before doing anything else.
    void registerLoggingFunctions(std::function<void(std::string const&)> logInfo,
                                  std::function<void(std::string const&)> logError);

    void setScene(aiScene const * scene) {
      scene_ = scene;
      command_ |= COMMAND_SET_SCENE;
    }
    void setSceneTransform(aiMatrix4x4 const *const modelToWorldTransform) {
      modelToWorldTransform_ = modelToWorldTransform;
      command_ |= COMMAND_SET_SCENE_TRANSFORM;
    }
    void setTransmitter(Transmitter const *const transmitter) {
      transmitter_ = transmitter;
      command_ |= COMMAND_SET_TRANSMITTER;
    }
    void setTransmitterTransform() {
      command_ |= COMMAND_SET_TRANSMITTER_TRANSFORM;
    }

    void quit() { command_ |= COMMAND_QUIT; }

    void eventLoop(); // Run in a separate thread.
    
    // The `rayTracingComplete()` signal indicates this is ready.
    // Caller owns result-pointer and must eventually `delete` it.
    aiScene const* rayTracingResult();

    void logOptixMessage(unsigned int level, const char *tag, const char *msg);

  signals:
    void rayTracingComplete();

  private:

    static unsigned int constexpr COMMAND_NONE = 0x0u;
    static unsigned int constexpr COMMAND_SET_SCENE = 0x1u;
    static unsigned int constexpr COMMAND_SET_SCENE_TRANSFORM = 0x1u << 1;
    static unsigned int constexpr COMMAND_SET_TRANSMITTER = 0x1u << 2;
    static unsigned int constexpr COMMAND_SET_TRANSMITTER_TRANSFORM = 0x1u << 3;
    static unsigned int constexpr COMMAND_QUIT = 0x1u << 7;

    void initialize_(ExecutionCursor whereInProgram); ///< CUDA and OptiX. Call after setting logging functions.
    void initializeCuda_(ExecutionCursor whereInProgram);
    void initializeOptix_(ExecutionCursor whereInProgram);
    
    void finalize_(ExecutionCursor whereInProgram);
    void finalizeCuda_(ExecutionCursor whereInProgram);
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

    void processScene_(ExecutionCursor whereInProgram);
    void processSceneTransform_(ExecutionCursor whereInProgram);
    void processTransmitter_(ExecutionCursor whereInProgram);
    void runRayTracing_(ExecutionCursor whereInProgram);

    // Multi-threaded communication with parent
    std::mutex& commandMutex_;
    std::condition_variable& commandConditionVariable_;
    unsigned int command_;

    aiScene const *scene_;
    aiMatrix4x4 const *modelToWorldTransform_;
    Transmitter const *transmitter_;

    std::mutex& resultMutex_;
    std::unique_ptr<aiScene> resultPointCloud_;

    // Callback functors which abstract the output streams for logging.
    std::function<void(std::string const&)> logInfo_;
    std::function<void(std::string const&)> logError_;
  
    int executionFrame_;

    // Scene data. AsyncCudaBuffer-s need to be initialized on the same (ray-tracing)
    // thread as all the other logic because they all use the same CUDA stream, and
    // each CUDA stream must be exclusively accessed by one CPU thread.
    // So these objects can only be instantiated after construction, once `initialize_()`
    // is called by the ray-tracing thread.
    std::unique_ptr<AsyncCudaBuffer> modelVertexBufferOnGpu_;
    std::unique_ptr<AsyncCudaBuffer> worldVertexBufferOnGpu_;
    std::unique_ptr<AsyncCudaBuffer> indexBufferOnGpu_;
    std::unique_ptr<AsyncCudaBuffer> rayHitVerticesOnGpu_; // world coords

    // CUDA data
    CUcontext cuCtx_ = 0;
    cudaStream_t cudaStream_;

    // OptiX stuff
    OptixDeviceContext optixDeviceContext_ = nullptr;

    // OptiX geometry acceleration structures (mesh + BVH etc.)
    OptixTraversableHandle gasHandle_;
    std::unique_ptr<AsyncCudaBuffer> gasBuild_;

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
