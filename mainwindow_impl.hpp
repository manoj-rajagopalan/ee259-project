#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "mainwindow.hpp"

#include <cuda_runtime.h>

class MainWindow_Impl : public MainWindow
{
  public:
    explicit MainWindow_Impl( QWidget* pParent = 0 );

	void logOptixMessage(unsigned int level, const char *tag, const char *msg);

  private:

      void initializeOptix_();
      void createOptixDeviceContext_(const std::string &indent);
      OptixTraversableHandle buildOptixAccelerationStructures_(const std::string &indent);
      void extractGeometry(std::vector<std::array<float, 3>> &vertexBuffer,
                       std::vector<std::array<uint16_t, 3>> &indexBuffer,
                       const std::string &indent);
      std::string loadOptixModuleSourceCode_(const std::string &indent);
      void makeOptixModule_(OptixPipelineCompileOptions optixPipelineCompileOptions,
                            char *optixLogBuffer,
                            size_t optixLogBufferSize,
                            const std::string &indent);
      OptixPipelineCompileOptions makeOptixPipelineCompileOptions_();
      OptixPipelineLinkOptions makeOptixPipelineLinkOptions_();
      OptixProgramGroupDesc makeRayGenProgramGroupDescription_();
      void makeRayGenProgramGroup_(char *optixLogBuffer,
                                   size_t optixLogBufferSize,
                                   const std::string &indent);
      OptixProgramGroupDesc makeHitGroupProgramGroupDescription_();
      void makeHitGroupProgramGroup_(char *optixLogBuffer,
                                     size_t optixLogBufferSize,
                                     const std::string &indent);
      void makeMissProgramGroup_(char *optixLogBuffer,
                                 size_t optixLogBufferSize,
                                 const std::string &indent);
      void buildOptixPipeline_(const std::string &indent);
      OptixShaderBindingTable makeOptixShaderBindingTable_(const std::string &indent);

      void releaseOptix_();

      void extractGeometry_(std::vector<std::array<float, 3>> &vertexBuffer,
                            std::vector<std::array<uint16_t, 3>> &indexBuffer)

      CUcontext cuCtx_ = 0;
      OptixDeviceContext optixDeviceContext_ = nullptr;
      OptixModule optixModule_;
      OptixPipeline optixPipeline_;
      CUdevicePtr gasBuildDevicePtr_;
      OptixProgramGroup raygenProgramGroup_;
      OptixProgramGroup hitGroupProgramGroup_;
      OptixProgramGroup missProgramGroup_;
};
