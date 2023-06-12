#include "AssimpOptixRayTracer.hpp"
#include "cuda_optix_sentinel.h"

// https://forums.developer.nvidia.com/t/linking-optix7-with-gcc/107036
#include <optix_function_table_definition.h>

#include <array>
#include <cassert>
#include <fstream>
#include <iterator>
#include <thread>

#include "LaunchParams.hpp"
#include "RaiiScopeLimitsLogger.hpp"

namespace {

// Help OptiX runtime route error messages to
manojr::AssimpOptixRayTracer *g_theAssimpOptixRayTracer = nullptr;

void optixLogCallback(unsigned int level, const char *tag, const char *msg, void* /* cbData */)
{
	g_theAssimpOptixRayTracer->logOptixMessage(level, tag, msg);
}

} // namespace

namespace manojr
{

AssimpOptixRayTracer::AssimpOptixRayTracer(
    std::mutex& rayTracingCommandMutex,
    std::condition_variable& rayTracingCommandConditionVariable,
    std::mutex& rayTracingResultMutex)
: commandMutex_(rayTracingCommandMutex),
  commandConditionVariable_(rayTracingCommandConditionVariable),
  resultMutex_(rayTracingResultMutex),
  executionFrame_{ 0 }
{
    // Capture all OptiX messages via optixLogCallback() above.
    g_theAssimpOptixRayTracer = this;
}

AssimpOptixRayTracer::~AssimpOptixRayTracer()
{}

void AssimpOptixRayTracer::eventLoop()
{
    ExecutionCursor whereInProgram;
    whereInProgram.frame = std::to_string(executionFrame_++);
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);

    initialize_(whereInProgram.advance());

    while(true) {
        std::unique_lock<std::mutex> lock(commandMutex_);
        commandConditionVariable_.wait(lock, [&](){ return command_ != COMMAND_NONE; });
        if(command_ == COMMAND_NONE) { continue; } // Guard against spurious wakes.

        whereInProgram.frame = std::to_string(executionFrame_++);
        whereInProgram.subFrameLevel = 0;

        if(command_ == COMMAND_QUIT) {
            command_ = COMMAND_NONE;
            break;
        }
        if(command_ | COMMAND_SET_SCENE) {
            command_ ^= COMMAND_SET_SCENE;
            processScene_(whereInProgram.advance());
        }
        if(command_ | COMMAND_SET_SCENE_TRANSFORM) {
            command_ ^= COMMAND_SET_SCENE_TRANSFORM;
            processSceneTransform_(whereInProgram.advance());
        }
        if(command_ | COMMAND_SET_TRANSMITTER) {
            command_ ^= COMMAND_SET_TRANSMITTER;
            processTransmitter_(whereInProgram.advance());
        }
        if(command_ | COMMAND_SET_TRANSMITTER_TRANSFORM) {
            command_ ^= COMMAND_SET_TRANSMITTER_TRANSFORM;
            // Update to transmitter pose is done in main thread itself, so nothing to do here
            // except to ray-trace later.
        }

        lock.unlock();
        runRayTracing_(whereInProgram.advance()); // emits rayTracingComplete()
    }

    whereInProgram.frame = std::to_string(executionFrame_++);
    whereInProgram.subFrameLevel = 0;
    finalize_(whereInProgram.advance());
}

aiScene const* AssimpOptixRayTracer::rayTracingResult() {
    std::unique_lock<std::mutex> lock(resultMutex_);
    return resultPointCloud_.release();
}

void AssimpOptixRayTracer::initialize_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLog(logInfo_, whereInProgram, __func__);
    
    initializeCuda_(whereInProgram.advance());
    modelVertexBufferOnGpu_.reset(new AsyncCudaBuffer(cudaStream_));
    worldVertexBufferOnGpu_.reset(new AsyncCudaBuffer(cudaStream_));
    indexBufferOnGpu_.reset(new AsyncCudaBuffer(cudaStream_));
    rayHitVerticesOnGpu_.reset(new AsyncCudaBuffer(cudaStream_));
    gasBuild_.reset(new AsyncCudaBuffer(cudaStream_));

    initializeOptix_(whereInProgram.advance());

    resultPointCloud_.reset();
}

void AssimpOptixRayTracer::finalize_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    finalizeOptix_(whereInProgram.advance());
    finalizeCuda_(whereInProgram.advance());
}

void AssimpOptixRayTracer::registerLoggingFunctions(
    std::function<void(std::string const&)> logInfo,
    std::function<void(std::string const&)> logError
)
{
    logInfo_ = logInfo;
    logError_ = logError;
}

void AssimpOptixRayTracer::initializeCuda_(ExecutionCursor whereInProgram)
{
    cudaFree(0);
	cuCtx_ = 0;

    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        int cudaDeviceCount = 0;
        CUDA_CHECK( GetDeviceCount(&cudaDeviceCount) );
        logInfo_('[' + whereInProgram.toString() + "] "
                 + "Found " + std::to_string(cudaDeviceCount) + " CUDA device");
        assert(cudaDeviceCount == 1);

        int cudaDevice = -1;
        CUDA_CHECK( GetDevice(&cudaDevice) );
        cudaDeviceProp cudaDeviceProps;
        CUDA_CHECK( GetDeviceProperties(&cudaDeviceProps, cudaDevice) );
        logInfo_('[' + whereInProgram.toString() + "] "
                 + std::string("CUDA device name is ") + cudaDeviceProps.name);

        CUDA_CHECK( SetDevice(cudaDevice) );
        CUDA_CHECK( StreamCreate(&cudaStream_) );
    }
    catch(std::exception& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::finalizeCuda_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        CUDA_CHECK( StreamDestroy(cudaStream_) );
    }
    catch(std::exception& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}


void AssimpOptixRayTracer::logOptixMessage(unsigned int level, const char *tag, const char *msg)
{
    std::ostringstream s;
	s << "-OptiX @" << level << '[' << tag << "] " << msg;
	logInfo_(s.str());
}


/// @brief Push geometry into GPU and reserve memory for results.
void AssimpOptixRayTracer::processScene_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);

    if(!scene_->HasMeshes()) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** Scene has no meshes!");
        return;
    }
    if(scene_->mNumMeshes != 1) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** Require scene to have 1 mesh. "
                "Has " + std::to_string(scene_->mNumMeshes));
        return;
    }
    aiMesh const& mesh = *(scene_->mMeshes[0]);
    if(mesh.mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** Require mesh to have only triangles."
                " Not the case");
        return;
    }

    // - Vertex and index buffers for geometry - 

    try {
        modelVertexBufferOnGpu_->asyncAllocAndUpload(mesh.mVertices, mesh.mNumVertices);
        worldVertexBufferOnGpu_->asyncAlloc(mesh.mNumVertices * sizeof(aiVector3D));
        worldVertexBufferOnGpu_->numElements = mesh.mNumVertices;
        worldVertexBufferOnGpu_->sizeOfElement = sizeof(aiVector3D);

        std::vector<int16_t> indexBuffer(mesh.mNumFaces * 3);
        int indexCounter = 0;
        for(uint32_t i = 0; i < mesh.mNumFaces; ++i) {
            const aiFace& face = mesh.mFaces[i];
            assert(face.mNumIndices == 3);
            assert(face.mIndices[0] < (1u << 16));
            indexBuffer[indexCounter++] = (uint16_t) face.mIndices[0];
            assert(face.mIndices[1] < (1u << 16));
            indexBuffer[indexCounter++] = (uint16_t) face.mIndices[1];
            assert(face.mIndices[2] < (1u << 16));
            indexBuffer[indexCounter++] = (uint16_t) face.mIndices[2];
        }
        indexBufferOnGpu_->asyncAllocAndUpload(indexBuffer);
    }
    catch(std::exception& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

__device__ __forceinline__
float inner_prod(float4 const& m, float3 const& v )
{
    return m.x*v.x + m.y*v.y + m.z*v.z + m.w /* *1 */;
}

__global__
void transformVertexBufferOnGpu(void *worldVertices,
                                void const *modelVertices,
                                int32_t const numVertices,
                                void const *modelToWorldTransform /* 3x4, row-major*/)
{
    int const linearIdx = blockIdx.x * blockDim.x + threadIdx.x;
    float3& w = *(((float3*) worldVertices) + linearIdx);
    float3 const& v = *(((float3 const*) modelVertices) + linearIdx);
    float4 const *const M = (float4 const *) modelToWorldTransform;
    w.x = inner_prod(M[0], v);
    w.y = inner_prod(M[1], v);
    w.z = inner_prod(M[2], v);
}

void AssimpOptixRayTracer::processSceneTransform_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);
    try {
        AsyncCudaBuffer modelToWorldTransformOnGpu{cudaStream_};
        modelToWorldTransformOnGpu.asyncAllocAndUpload((float const*) modelToWorldTransform_, 12);
        // 8 block with 32 threads each
        int const kNumThreadsPerBlock = 32;
        // TODO: large meshes can exceed CUDA limits
        int const kNumBlocks = (modelVertexBufferOnGpu_->numElements + kNumThreadsPerBlock-1) / kNumThreadsPerBlock;
        transformVertexBufferOnGpu<<<kNumBlocks, kNumThreadsPerBlock, 0, cudaStream_>>>(
            worldVertexBufferOnGpu_->d_pointer(),
            modelVertexBufferOnGpu_->d_pointer(),
            modelVertexBufferOnGpu_->numElements,
            modelToWorldTransformOnGpu.d_pointer()
        );
        CUDA_CHECK( GetLastError() );

        // Rebuild BVH etc.
        buildOptixAccelerationStructures_(whereInProgram.advance()); // set gasBuild_, gasHandle_

    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

/// Allocate on-GPU memory for ray-tracing results (collision points in world space).
void AssimpOptixRayTracer::processTransmitter_(ExecutionCursor whereInProgram) {
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        // In worst-case all rays will hit geometry.
        int32_t const numRays = transmitter_->numRays_x * transmitter_->numRays_y;
        rayHitVerticesOnGpu_->asyncFree();
        rayHitVerticesOnGpu_->asyncAlloc(numRays * sizeof(float3));
        // The following metadata needs to be ready before we 'download' results from GPU to CPU.
        rayHitVerticesOnGpu_->sizeOfElement = sizeof(float3);
        // Haven't ray-traced yet.
        // Will set this to the value of atomic counter (incremented by ray-collisions), post-ray-tracing.
        rayHitVerticesOnGpu_->numElements = 0;

        // The pose part of transmitter is handled by setTransmitterTransform() and processTransmitterTransform_().
        // Other data within Transmitter are sent as part of (optix)Launch(Params);
    }
    catch(std::exception const& e) {
        logInfo_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::runRayTracing_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);
    try {
        OptixLaunchParams launchParams{};
        {
            launchParams.pointCloud = (void*) rayHitVerticesOnGpu_->d_pointer();
            launchParams.gasHandle = gasHandle_;
            launchParams.gpuAtomicNumHits = 0;
            std::unique_lock<std::mutex> lock(commandMutex_);
            launchParams.transmitter = *transmitter_;
        }
        AsyncCudaBuffer launchParamsOnGpu{cudaStream_};
        launchParamsOnGpu.asyncAllocAndUpload(&launchParams, 1);
        OPTIX_CHECK(
            optixLaunch(optixPipeline_,
                        cudaStream_,
                        (CUdeviceptr) launchParamsOnGpu.d_pointer(),
                        launchParamsOnGpu.sizeInBytes,
                        &optixSbt_,
                        launchParams.transmitter.numRays_x,
                        launchParams.transmitter.numRays_y,
                        1)
        );
        launchParamsOnGpu.asyncDownload(&launchParams, 1);
        launchParamsOnGpu.sync(); // Ensure download completes before accessing results.
        rayHitVerticesOnGpu_->numElements = launchParams.gpuAtomicNumHits;
        
        aiMesh *const pointCloudMesh = new aiMesh;
        pointCloudMesh->mNumVertices = launchParams.gpuAtomicNumHits;
        pointCloudMesh->mVertices = new aiVector3D[launchParams.gpuAtomicNumHits];
        rayHitVerticesOnGpu_->asyncDownload(pointCloudMesh->mVertices, pointCloudMesh->mNumVertices);
        rayHitVerticesOnGpu_->sync(); // Ensure download completes before accessing results.

        // Scenes must have a root node which indexes into the master mesh-array within the parent scene.
        aiNode *const pointCloudNode = new aiNode;
        pointCloudNode->mName = "point-cloud";
        pointCloudNode->mNumMeshes = 1;
        pointCloudNode->mMeshes = new unsigned int [pointCloudNode->mNumMeshes];
        pointCloudNode->mMeshes[0] = 0; // First entry in the array of meshes in the scene.
        pointCloudNode->mNumChildren = 0;
        pointCloudNode->mChildren = nullptr;

        resultPointCloud_.reset(new aiScene);
        using aiMeshPtr_t = aiMesh*;
        resultPointCloud_->mMeshes = new aiMeshPtr_t[1];
        resultPointCloud_->mMeshes[0] = pointCloudMesh;
        resultPointCloud_->mNumMeshes = 1;
        resultPointCloud_->mRootNode = pointCloudNode;

        emit rayTracingComplete();
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::createOptixDeviceContext_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        OptixDeviceContextOptions optixDeviceContextOptions;
        optixDeviceContextOptions.logCallbackFunction = &optixLogCallback;
        optixDeviceContextOptions.logCallbackLevel = 4;
        optixDeviceContextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        OPTIX_CHECK(
            optixDeviceContextCreate(cuCtx_, &optixDeviceContextOptions, &optixDeviceContext_)
        );
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::buildOptixAccelerationStructures_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger execLog(logInfo_, whereInProgram, __func__);

	// https://raytracing-docs.nvidia.com/optix7/api/struct_optix_accel_build_options.html
	OptixAccelBuildOptions optixAccelBuildOptions = {};
	{
		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ggaff328b8278fbd1900558593599698bbaafa820662dca4a85935ab74c704665d93
		optixAccelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga411c1c6d9f4d8e039ae19e9dea65958a
		optixAccelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		// Ignore optixAccessBuildOptions.motionOptions
	}

	// https://raytracing-docs.nvidia.com/optix7/api/struct_optix_build_input.html
	OptixBuildInput optixBuildInput = {};
	{
		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga7932d1d9cdf33506a75a5da5d8a62d94
		optixBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// https://raytracing-docs.nvidia.com/optix7/api/struct_optix_build_input_triangle_array.html
        OptixBuildInputTriangleArray& triangleArray = optixBuildInput.triangleArray;
		triangleArray.vertexBuffers = (CUdeviceptr*) &(worldVertexBufferOnGpu_->d_ptr);
		triangleArray.numVertices = (unsigned int) worldVertexBufferOnGpu_->numElements;
		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga55c6d96161ef202d48023a8a1d126102
		triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleArray.vertexStrideInBytes = 0; // tightly-packed

		triangleArray.indexBuffer = (CUdeviceptr) indexBufferOnGpu_->d_pointer();
		triangleArray.numIndexTriplets = (unsigned int) indexBufferOnGpu_->numElements / 3; // triplets, remember?
		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa98b8fb6bf2d2455c310125f3fab74e6
		triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
		triangleArray.indexStrideInBytes = 0; // tightly-packed

        const unsigned int triangle_flags[]{OPTIX_GEOMETRY_FLAG_NONE};
        triangleArray.flags = triangle_flags;
        triangleArray.numSbtRecords = 1;
	}

    // Compute buffer sizes for building acceleration structures
	OptixAccelBufferSizes optixAccelBufferSizes{};
	OPTIX_CHECK(
		optixAccelComputeMemoryUsage(optixDeviceContext_,
		                             &optixAccelBuildOptions,
									 &optixBuildInput, 1,
									 &optixAccelBufferSizes)
	);
    assert(optixAccelBufferSizes.tempUpdateSizeInBytes == 0);

    // Allocate buffers for the build
    gasBuild_->asyncResize(optixAccelBufferSizes.outputSizeInBytes);

    AsyncCudaBuffer gasBuildTemp{cudaStream_};
    gasBuildTemp.asyncResize(optixAccelBufferSizes.tempSizeInBytes);

    OPTIX_CHECK(
        optixAccelBuild(optixDeviceContext_,
                        cudaStream_,
                        &optixAccelBuildOptions,
                        &optixBuildInput, 1,
                        (CUdeviceptr) gasBuildTemp.d_pointer(), gasBuildTemp.sizeInBytes,
                        (CUdeviceptr) gasBuild_->d_pointer(), gasBuild_->sizeInBytes,
                        &gasHandle_,
                        nullptr, 0 // emitted properties
                        );
    );
}

std::string AssimpOptixRayTracer::loadOptixModuleSourceCode_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    std::string sourceCode;
    try {
        std::ifstream sourceCodeFile("ee259.ptx");
        // https://stackoverflow.com/a/2912614
        std::istreambuf_iterator<char> sourceCodeFileIter{sourceCodeFile};
        sourceCode.assign(sourceCodeFileIter,
                          std::istreambuf_iterator<char>{});
        logInfo_('[' + whereInProgram.toString() + "] Source code has "
                 + std::to_string(sourceCode.size()) + " chars");
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
    return sourceCode;
}

void AssimpOptixRayTracer::makeOptixModule_(OptixPipelineCompileOptions optixPipelineCompileOptions,
                                            char* logBuffer,
                                            size_t logBufferSize,
                                            ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        ExecutionCursor whereInSubProgram = whereInProgram.nextLevel();

        // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_module_compile_options.html
        OptixModuleCompileOptions optixModuleCompileOptions{};
        {
            optixModuleCompileOptions.maxRegisterCount = 0; // unlimited
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaea8ecab8ad903804364ea246eefc79b2
            optixModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // none, for debug
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga2a921efc5016b2b567fa81ddb429e81a
            optixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            optixModuleCompileOptions.numBoundValues = 0;
            optixModuleCompileOptions.boundValues = 0; // null pointer
            optixModuleCompileOptions.numPayloadTypes = 0;
            optixModuleCompileOptions.payloadTypes = 0;
        }

        std::string optixModuleSourceCode = loadOptixModuleSourceCode_(whereInSubProgram.advance());
        OPTIX_CHECK(
            optixModuleCreate(optixDeviceContext_,
                              &optixModuleCompileOptions,
                              &optixPipelineCompileOptions,
                              optixModuleSourceCode.c_str(),
                              optixModuleSourceCode.size(),
                              logBuffer, &logBufferSize,
                              &optixModule_)
        );
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

OptixPipelineCompileOptions AssimpOptixRayTracer::makeOptixPipelineCompileOptions_()
{
    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_pipeline_compile_options.html
    OptixPipelineCompileOptions optixPipelineCompileOptions;
    optixPipelineCompileOptions.usesMotionBlur = 0;
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gabd8bb7368518a44361e045fe5ad1fd17
    optixPipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    optixPipelineCompileOptions.numPayloadValues = 0; // TODO
    optixPipelineCompileOptions.numAttributeValues = 2; // min
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga95e8175699d1a23c5c1d5333c4468190
    optixPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                                                 OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                                 OPTIX_EXCEPTION_FLAG_USER |
                                                 OPTIX_EXCEPTION_FLAG_DEBUG;
    optixPipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga1171b332da08991dd9e6ef54b52b3ba4
    optixPipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    optixPipelineCompileOptions.allowOpacityMicromaps = 0u;
    return optixPipelineCompileOptions;
}

OptixPipelineLinkOptions AssimpOptixRayTracer::makeOptixPipelineLinkOptions_()
{
    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_pipeline_link_options.html
    OptixPipelineLinkOptions optixPipelineLinkOptions;
    optixPipelineLinkOptions.maxTraceDepth = 2;
    return optixPipelineLinkOptions;
}

void AssimpOptixRayTracer::makeRaygenProgramGroup_(char *logBuffer,
                                                   size_t logBufferSize,
                                                   ExecutionCursor whereInProgram)
{
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga0b44fd0708cced8b77665cfac5453573

    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_desc.html
        OptixProgramGroupDesc raygenProgramGroupDescription{};
        {
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gabca35b1218b4df575a5c42926da0d978
            raygenProgramGroupDescription.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa65e764f5bba97fda45d4453c0464596
            raygenProgramGroupDescription.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

            // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_single_module.html
            raygenProgramGroupDescription.raygen.module = optixModule_;
            raygenProgramGroupDescription.raygen.entryFunctionName = "__raygen__rg";
        }

        // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_options.html
        OptixProgramGroupOptions raygenProgramGroupOptions{}; // no payload

        // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
        OPTIX_CHECK(
            optixProgramGroupCreate(optixDeviceContext_,
                                    &raygenProgramGroupDescription, 1, // array
                                    &raygenProgramGroupOptions,
                                    logBuffer, &logBufferSize, // modified
                                    &raygenProgramGroup_)
        );
        logBuffer[logBufferSize] = '\0';
        logInfo_('[' + whereInProgram.toString() + "] " + (char*) logBuffer);
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::makeHitGroupProgramGroup_(char *logBuffer,
                                                     size_t logBufferSize,
                                                     ExecutionCursor whereInProgram)
{
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga0b44fd0708cced8b77665cfac5453573

    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_desc.html
        OptixProgramGroupDesc hitGroupProgramGroupDescription;
        {
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gabca35b1218b4df575a5c42926da0d978
            hitGroupProgramGroupDescription.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa65e764f5bba97fda45d4453c0464596
            hitGroupProgramGroupDescription.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

            // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_hitgroup.html
            hitGroupProgramGroupDescription.hitgroup.moduleCH = optixModule_;
            hitGroupProgramGroupDescription.hitgroup.entryFunctionNameCH = "__closesthit__xpoint";
        }

        // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_options.html
        OptixProgramGroupOptions hitGroupProgramGroupOptions{}; // no payload

        // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
        OPTIX_CHECK(
            optixProgramGroupCreate(optixDeviceContext_,
                                    &hitGroupProgramGroupDescription, 1, // array
                                    &hitGroupProgramGroupOptions,
                                    logBuffer, &logBufferSize,
                                    &hitGroupProgramGroup_)
        );
        logBuffer[logBufferSize] = '\0';
        logInfo_('[' + whereInProgram.toString() + "] " + (char*) logBuffer);
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::makeMissProgramGroup_(char *logBuffer,
                                                 size_t logBufferSize,
                                                 ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga0b44fd0708cced8b77665cfac5453573
        // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_desc.html
        OptixProgramGroupDesc missProgramGroupDescription{};
        {
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gabca35b1218b4df575a5c42926da0d978
            missProgramGroupDescription.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa65e764f5bba97fda45d4453c0464596
            missProgramGroupDescription.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

            // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_single_module.html
            missProgramGroupDescription.miss.module = optixModule_;
            missProgramGroupDescription.miss.entryFunctionName = "__miss__noop";
        }
        // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_options.html
        OptixProgramGroupOptions missProgramGroupOptions{}; // no payload

        // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
        OPTIX_CHECK(
            optixProgramGroupCreate(optixDeviceContext_,
                                    &missProgramGroupDescription, 1, // array
                                    &missProgramGroupOptions,
                                    logBuffer, &logBufferSize,
                                    &missProgramGroup_)
        );
        logBuffer[logBufferSize] = '\0';
        logInfo_('[' + whereInProgram.toString() + "] " + (char*) logBuffer);
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::buildOptixPipeline_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        ExecutionCursor whereInSubProgram = whereInProgram.nextLevel();

        // Char buffers for optix logging.
        size_t const kOptixLogBufferSize = 1024;
        char optixLogBuffer[kOptixLogBufferSize];
        optixLogBuffer[kOptixLogBufferSize-1] = '\0'; // ensure NULL-TERMINATED
        // ... Provide (kOptixLogBufferSize-1) as buffer-size in all uses to maintain this property.

        OptixPipelineCompileOptions optixPipelineCompileOptions =
            makeOptixPipelineCompileOptions_();
        makeOptixModule_(optixPipelineCompileOptions,
                        optixLogBuffer, kOptixLogBufferSize-1,
                        whereInSubProgram.advance());
        OptixPipelineLinkOptions optixPipelineLinkOptions =
            makeOptixPipelineLinkOptions_();
        makeRaygenProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, whereInSubProgram.advance());
        makeHitGroupProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, whereInSubProgram.advance());
        makeMissProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, whereInSubProgram.advance());

        constexpr unsigned int kNumProgramGroups = 3;
        std::array<OptixProgramGroup, kNumProgramGroups> optixProgramGroups{
            raygenProgramGroup_, hitGroupProgramGroup_, missProgramGroup_
        };
        
        size_t optixLogBufferSize = kOptixLogBufferSize-1;
        OPTIX_CHECK(
            optixPipelineCreate(optixDeviceContext_,
                                &optixPipelineCompileOptions,
                                &optixPipelineLinkOptions,
                                optixProgramGroups.data(), kNumProgramGroups,
                                optixLogBuffer, &optixLogBufferSize,
                                &optixPipeline_);
        );
        optixLogBuffer[optixLogBufferSize] = '\0';
        logInfo_('[' + whereInProgram.toString() + "] " + (char*) optixLogBuffer);
    }
    catch(std::exception const& e) {
        logInfo_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

template<typename T>
struct ShaderBindingTableRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT)
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData { /* TODO */};
using RayGenSbtRecord = ShaderBindingTableRecord<RayGenData>;

struct MissData { /* TODO */ };
using MissSbtRecord = ShaderBindingTableRecord<MissData>;

struct HitGroupData { /* TODO */ };
using HitGroupSbtRecord = ShaderBindingTableRecord<HitGroupData>;

void*
AssimpOptixRayTracer::makeRaygenSbtRecord_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    void* result { nullptr };
    try {
        RayGenSbtRecord raygenSbtRecord;
        OPTIX_CHECK(
            optixSbtRecordPackHeader(raygenProgramGroup_, (void*) &raygenSbtRecord)
        );
        // Populate raygenSbtRecord.data here if needed in future
        AsyncCudaBuffer raygenSbtRecordOnGpu{cudaStream_};
        raygenSbtRecordOnGpu.asyncAllocAndUpload(&raygenSbtRecord, 1);
        result = raygenSbtRecordOnGpu.detach();
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
    return result;
}

std::pair<void*, unsigned int>
AssimpOptixRayTracer::makeMissSbtRecord_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    std::pair<void*, unsigned int> result { 0, 0 };
    try {
        MissSbtRecord missSbtRecord;
        OPTIX_CHECK(
            optixSbtRecordPackHeader(missProgramGroup_, (void*) &missSbtRecord)
        );
        AsyncCudaBuffer missSbtRecordOnGpu{cudaStream_};
        missSbtRecordOnGpu.asyncAllocAndUpload(&missSbtRecord, 1);
        result.second = missSbtRecordOnGpu.sizeInBytes;
        // Destroys internal metadata, so ordered after extracting sizeInBytes.
        result.first = missSbtRecordOnGpu.detach();
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
    return result;
}

std::pair<void*, unsigned int>
AssimpOptixRayTracer::makeHitGroupSbtRecord_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    std::pair<void*, unsigned int> result{ 0, 0 };
    try {
        HitGroupSbtRecord hitGroupSbtRecord;
        OPTIX_CHECK(
            optixSbtRecordPackHeader(hitGroupProgramGroup_, (void*) &hitGroupSbtRecord)
        );
        AsyncCudaBuffer hitGroupSbtRecordOnGpu{cudaStream_};
        hitGroupSbtRecordOnGpu.asyncAllocAndUpload(&hitGroupSbtRecord, 1);
        result.second = hitGroupSbtRecordOnGpu.sizeInBytes;
        // Destroys internal metadata, so ordered after extracting sizeInBytes.
        result.first = hitGroupSbtRecordOnGpu.detach();
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
    return result;
}

/// @brief Populates optixSbt_
/// @param indent Indentation-prefix
void AssimpOptixRayTracer::makeOptixShaderBindingTable_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);
    try {
        ExecutionCursor whereInSubProgram = whereInProgram.nextLevel();
        optixSbt_.raygenRecord = (CUdeviceptr) makeRaygenSbtRecord_(whereInSubProgram.advance());

        optixSbt_.missRecordCount = 1;
        std::pair<void*, unsigned int> missSbtRecordInfo = makeMissSbtRecord_(whereInSubProgram.advance());
        optixSbt_.missRecordBase = (CUdeviceptr) missSbtRecordInfo.first;
        optixSbt_.missRecordStrideInBytes = missSbtRecordInfo.second;

        optixSbt_.hitgroupRecordCount = 1;
        std::pair<void*, unsigned int> hitGroupSbtRecordInfo = makeHitGroupSbtRecord_(whereInSubProgram.advance());
        optixSbt_.hitgroupRecordBase = (CUdeviceptr) hitGroupSbtRecordInfo.first;
        optixSbt_.hitgroupRecordStrideInBytes = hitGroupSbtRecordInfo.second;

        optixSbt_.callablesRecordBase = 0;
        optixSbt_.callablesRecordCount = 0;
        optixSbt_.callablesRecordStrideInBytes = 0;

        optixSbt_.exceptionRecord = 0;
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::initializeOptix_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);
    ExecutionCursor whereInSubProgram = whereInProgram.nextLevel();
	try {
		OPTIX_CHECK(optixInit());
		createOptixDeviceContext_(whereInSubProgram.advance()); // set optixDeviceContext_
        buildOptixPipeline_(whereInSubProgram.advance()); // set optixModule_, optixPipeline_
        makeOptixShaderBindingTable_(whereInSubProgram.advance()); // set optixSbt_
	}
	catch(std::exception& e) {
		logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
	}
}

void AssimpOptixRayTracer::finalizeOptix_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);
    try {
        OPTIX_CHECK( optixProgramGroupDestroy(raygenProgramGroup_) );
        OPTIX_CHECK( optixProgramGroupDestroy(hitGroupProgramGroup_) );
        OPTIX_CHECK( optixProgramGroupDestroy(missProgramGroup_) );
        OPTIX_CHECK( optixPipelineDestroy(optixPipeline_) );
        OPTIX_CHECK( optixModuleDestroy(optixModule_) );
        OPTIX_CHECK( optixDeviceContextDestroy(optixDeviceContext_) );
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

} // namespace manojr
