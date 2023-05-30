#include <array>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "CUDABuffer.h"

#include "mainwindow.hpp"


class MainWindow_Impl : public MainWindow
{
  public:
    explicit MainWindow_Impl( QWidget* pParent = 0 );

    void logOptixMessage(unsigned int level, const char *tag, const char *msg);

  private:

      void initializeCuda_();
      void finalizeCuda_();
 
      void initializeOptix_();
      void finalizeOptix_();
 
      void createOptixDeviceContext_(const std::string &indent);
      OptixTraversableHandle buildOptixAccelerationStructures_(const std::string &indent);
      void extractGeometry_(const std::string &indent);
      std::string loadOptixModuleSourceCode_(const std::string &indent);
      void makeOptixModule_(OptixPipelineCompileOptions optixPipelineCompileOptions,
                            char *optixLogBuffer,
                            size_t optixLogBufferSize,
                            const std::string &indent);
      OptixPipelineCompileOptions makeOptixPipelineCompileOptions_();
      OptixPipelineLinkOptions makeOptixPipelineLinkOptions_();
      OptixProgramGroupDesc makeRaygenProgramGroupDescription_();
      void makeRaygenProgramGroup_(char *optixLogBuffer,
                                   size_t optixLogBufferSize,
                                   const std::string &indent);
      OptixProgramGroupDesc makeHitGroupProgramGroupDescription_();
      void makeHitGroupProgramGroup_(char *optixLogBuffer,
                                     size_t optixLogBufferSize,
                                     const std::string &indent);
      OptixProgramGroupDesc makeMissProgramGroupDescription_();
      void makeMissProgramGroup_(char *optixLogBuffer,
                                 size_t optixLogBufferSize,
                                 const std::string &indent);
      void buildOptixPipeline_(const std::string &indent);
      
      CUdeviceptr makeRaygenSbtRecord_(const std::string &indent);
      std::pair<CUdeviceptr, unsigned int> makeMissSbtRecord_(const std::string &indent);
      std::pair<CUdeviceptr, unsigned int> makeHitGroupSbtRecord_(const std::string &indent);
      OptixShaderBindingTable makeOptixShaderBindingTable_(const std::string &indent);

      void launchOptix_(OptixTraversableHandle& gas_handle,
                        const OptixShaderBindingTable &sbt,
                        const std::string &indent);

      osc::CUDABuffer vertexBufferOnGpu_;
      osc::CUDABuffer indexBufferOnGpu_;

      CUcontext cuCtx_ = 0;
      cudaStream_t cudaStream_;

      OptixDeviceContext optixDeviceContext_ = nullptr;
      OptixModule optixModule_;
      OptixPipeline optixPipeline_;
      osc::CUDABuffer gasBuild_;
      OptixProgramGroup raygenProgramGroup_;
      OptixProgramGroup hitGroupProgramGroup_;
      OptixProgramGroup missProgramGroup_;


};
