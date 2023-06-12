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
  public:

    Q_OBJECT;

    AssimpOptixRayTracer(std::mutex& rayTracingCommandMutex,
	                       std::condition_variable& rayTracingCommandConditionVariable,
                         std::mutex& rayTracingResultMutex);
    ~AssimpOptixRayTracer();

    /// @brief Call before doing anything else.
    void registerLoggingFunctions(std::function<void(std::string const&)> logInfo,
                                  std::function<void(std::string const&)> logError);

    void setScene(aiScene const * scene) {
      scene_ = scene;
      command_ |= command_t::SET_SCENE;
    }
    void setSceneTransform(aiMatrix4x4 const *const modelToWorldTransform) {
      modelToWorldTransform_ = modelToWorldTransform;
      command_ |= command_t::SET_SCENE_TRANSFORM;
    }
    void setTransmitter(Transmitter const *const transmitter) {
      transmitter_ = transmitter;
      command_ |= command_t::SET_TRANSMITTER;
    }
    void setTransmitterTransform() {
      command_ |= command_t::SET_TRANSMITTER_TRANSFORM;
    }

    void quit() { command_ |= command_t::QUIT; }

    void eventLoop(); // Run in a separate thread.
    
    // The `rayTracingComplete()` signal indicates this is ready.
    // Caller owns result-pointer and must eventually `delete` it.
    aiScene const* rayTracingResult();

    void logOptixMessage(unsigned int level, const char *tag, const char *msg);

  public signals:
    void rayTracingComplete();

  private:

    enum class command_t : uint8_t {
      NONE = 0x0u,
      SET_SCENE = 0x1u,
      SET_SCENE_TRANSFORM = 0x1u << 1,
      SET_TRANSMITTER = 0x1u << 2,
      SET_TRANSMITTER_TRANSFORM = 0x1u << 3,
      QUIT = 0x1u << 7
    };

    void initialize_(ExecutionCursor whereInProgram); ///< CUDA and OptiX. Call after setting logging functions.
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

    void processScene_(ExecutionCursor whereInProgram);
    void processSceneTransform_(ExecutionCursor whereInProgram);
    void processTransmitter_(ExecutionCursor whereInProgram);
    void runRayTracing_(ExecutionCursor whereInProgram);

    // Multi-threaded communication with parent
    std::mutex& rayTracingCommandMutex_;
    std::condition_variable& rayTracingCommandConditionVariable_;
    std::mutex& rayTracingResultMutex_;

    aiScene const *scene_;
    aiMatrix4x4 const *modelToWorldTransform_;
    Transmitter const *transmitter_;

    std::unique_ptr<aiScene> rayTracedPointCloud_;

    command_t command_;

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
