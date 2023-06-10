#include "assimp_optix_ray_tracer.hpp"
#include "cuda_optix_sentinel.h"

// https://forums.developer.nvidia.com/t/linking-optix7-with-gcc/107036
#include <optix_function_table_definition.h>

#include <array>
#include <cassert>
#include <fstream>
#include <iterator>

#include "CUDABuffer.h"
#include "LaunchParams.hpp"
#include "ExecutionCursor.hpp"
#include "RaiiScopeLimitsLogger.hpp"

#include "mainwindow.hpp"

std::unique_ptr<MainWindow> makeMainWindow()
{
	return std::make_unique<MainWindow_Impl>();
}

namespace {

// Help OptiX runtime route error messages to
AssimpOptixRayTracer *g_theAssimpOptixRayTracer = nullptr;

void optixLogCallback(unsigned int level, const char *tag, const char *msg, void* /* cbData */)
{
	g_theAssimpOptixRayTracer->logOptixMessage(level, tag, msg);
}

} // namespace

namespace manojr
{

AssimpOptixRayTracer::AssimpOptixRayTracer(
    std::function<void(std::string const&)> logInfo,
    std::function<void(std::string const&)> logError
)
: logInfo_{ logInfo },
  logError_{ logError },
  executionFrame_{ 0 }
{
    ExecutionCursor whereInProgram;
    whereInProgram.frame = std::to_string(executionFrame_++);
    RaiiScopeLimitsLogger scopeLog(logInfo_, whereInProgram, "AssimpOptixRayTracer CTOR");

    g_theAssimpOptixRayTracer = this; // Route all OptiX messages
    ExecutionCursor whereInSubProgram = whereInProgram.nexLevel();
    initializeCuda_(whereInSubProgram.advance());
    initializeOptix_(whereInSubProgram.advance());
}

AssimpOptixRayTracer::~AssimpOptixRayTracer()
{
    ExecutionCursor whereInProgram;
    whereInProgram.frame = std::to_string(executionFrame_++);
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, "AssimpOptixRayTracer DTOR");
    ExecutionCursor whereInSubProgram = whereInProgram.nextLevel();
    finalizeOptix_(whereInSubProgram.advance());
    finalizeCuda_(whereInSubProgram.advance());
}

void AssimpOptixRayTracer::initializeCuda_(ExecutionCursor whereInProgram)
{
    try {
        int cudaDeviceCount = 0;
        CUDA_CHECK( GetDeviceCount(&cudaDeviceCount) );
        logInfo('[' + whereInProgram.toString() + "] "
                + "Found " + std::to_string(cudaDeviceCount) + " CUDA device");
        assert(cudaDeviceCount == 1);

        int cudaDevice = -1;
        CUDA_CHECK( GetDevice(&cudaDevice) );
        cudaDeviceProp cudaDeviceProps;
        CUDA_CHECK( GetDeviceProperties(&cudaDeviceProps, cudaDevice) );
        logInfo('[' + whereInProgram.toString() + "] "
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
void AssimpOptixRayTracer::setScene(aiScene const& scene)
{
    ExecutionCursor whereInProgram;
    whereInProgram.frame = std::to_string(executionFrame++);
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);

    { // Check preconditions.
        if(!mScene->HasMeshes()) {
            logError_('[' + whereInProgram.toString() + "] ***ERROR*** Scene has no meshes!");
            return;
        }
        if(scene.mNumMeshes != 1) {
            logError_('[' + whereInProgram.toString() + "] ***ERROR*** Require scene to have 1 mesh. "
                    "Has " + std::to_string(scene.mNumMeshes));
            return;
        }
        aiMesh const& mesh = *scene.mMeshes[0];
        if(mesh.mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
            logError_('[' + whereInProgram.toString() + "] ***ERROR*** Require mesh to have only triangles."
                    " Not the case");
            return;
        }
    } // preconditions

    // - Vertex and index buffers for geometry - 

    try {
        modelVertexBufferOnGpu_.allocAndUpload(mesh.mVertices, mesh.mNumVertices, cudaStream_);

        std::vector<int16_t> indexBuffer(mesh->mNumFaces * 3);
        int indexCounter = 0;
        for(uint32_t i = 0; i < mesh->mNumFaces; ++i) {
            const aiFace& face = mesh->mFaces[i];
            assert(face.mNumIndices == 3);
            assert(face.mIndices[0] < (1u << 16));
            indexBuffer[indexCounter++] = (uint16_t) face.mIndices[0];
            assert(face.mIndices[1] < (1u << 16));
            indexBuffer[indexCounter++] = (uint16_t) face.mIndices[1];
            assert(face.mIndices[2] < (1u << 16));
            indexBuffer[indexCounter++] = (uint16_t) face.mIndices[2];
        }
        indexBufferOnGpu_.allocAndUpload(indexBuffer);

        // Identity transform: copies model vertex buffer into world vertex buffer
        // and also, importantly, builds all the OptiX geometry acceleration structures.
        aiMatrix4D const identityTransform{}
        setSceneTransform(identityTransform);

        // - Buffer for results on GPU -

        // In worst-case all rays will hit geometry.
        rayHitVerticesOnGpu_.alloc(mesh.mVertices * sizeof(aiVertex3D), cudaStream_);
        rayHitVerticesOnGpu_.sizeOfElement = sizeof(aiVertex3D);
        rayHitVerticesOnGpu_.numElements = 0; // Haven't ray-traced yet.
    }
    catch(std::exception& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

__device__ __forceinline__
float inner_prod(float4 const& m, float3 const& v )
{
    return m[0]*v[0] + m[1]*v[1] + m[2]*v[2] + m[3] /* *1 */;
}

__global__
void transformVertexBufferOnGpu(void *worldVertices,
                                void const *modelVertices,
                                int32_t const numVertices,
                                void const *modelToWorldTransform /* 3x4, row-major*/)
{
    float3 *const w = ((float3*) worldVertices) + threadIdx.x;
    float3 const *const v = ((float3 const*) modelVertices) + threadIdx.x;
    float4 const *const M = (float4 const *) modelToWorldTransform;
    w[0] = inner_prod(M[0], v);
    w[1] = inner_prod(M[1], v);
    w[2] = inner_prod(M[2], v);
}

void AssimpOptixRayTracer::setSceneTransform(aiMatrix4D& modelToWorldTransform)
{
    ExecutionCursor whereInProgram;
    whereInProgram.frame = std::to_string(execFrame_++);
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);

    try {
        osc::CUDABuffer modelToWorldTransformOnGpu;
        modelToWorldTransformOnGpu.allocAndUpload((float*) &modelToWorldTransform, 12, cudaStream_);
        // 8 block with 32 threads each
        int const kNumThreadsPerBlock = 32;
        // TODO: large meshes can exceed CUDA limits
        int const kNumBlocks = (modelVertexBufferOnGpu_.numElements + kNumThreadsPerBlock-1) / kNumThreadsPerBlock;
        transformVertexBufferOnGpu<<<kNumBlocks, kNumThreadsPerBlock, 0, cudaStream_>>>(
            modelVertexBufferOnGpu_.d_pointer(),
            modelVertexBufferOnGpu_.numElements,
            modelToWorldTransformOnGpu.d_pointer()
        );
        CUDA_CHECK( GetLastError() );

        ExecutionCursor whereInSubProgram = whereInProgram.nextLevel();
        // Rebuild BVH etc.
        buildOptixAccelerationStructures_(whereInSubProgram.advance()); // set gasBuild_, gasHandle_

    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
}

void AssimpOptixRayTracer::setTransmitterTransform(aiMatrix4D const& transmitterToWorldTransform)
{
	transmitter_.xUnitVector[0] = transmitterToWorldTransform[0][0];
	transmitter_.xUnitVector[1] = transmitterToWorldTransform[1][0];
	transmitter_.xUnitVector[2] = transmitterToWorldTransform[2][0];
	
	transmitter_.yUnitVector[0] = transmitterToWorldTransform[0][1];
	transmitter_.yUnitVector[1] = transmitterToWorldTransform[1][1];
	transmitter_.yUnitVector[2] = transmitterToWorldTransform[2][1];
	
	transmitter_.zUnitVector[0] = transmitterToWorldTransform[0][2];
	transmitter_.zUnitVector[1] = transmitterToWorldTransform[1][2];
	transmitter_.zUnitVector[2] = transmitterToWorldTransform[2][2];
}


aiScene* AssimpOptixRayTracer::runRayTracing()
{
    ExecutionCursor whereInProgram;
    whereInProgram.frame = std::to_string(executionFrame_++);
    RaiiScopeLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);

    aiScene *const pointCloudScene = new aiScene;
    try {
        OptixLaunchParams launchParams{};
        {
            OptixLaunchParams launchParams;
            launchParams.pointCloud = (void*) rayHitVerticesOnGpu_.d_pointer();
            launchParams.gasHandle = gasHandle_;
            launchParams.transmitter = transmitter_;
            launchParams.gpuAtomicNumHits = 0;
        }
        osc::CUDABuffer launchParamsOnGpu;
        launchParamsOnGpu.allocAndUpload(&launchParams, 1, cudaStream_);
        OPTIX_CHECK(
            optixLaunch(optixPipeline_,
                        cudaStream_,
                        launchParamsOnGpu.d_pointer(),
                        launchParamsOnGpu.sizeInBytes,
                        &optixSbt_,
                        launchParams.transmitter.numRays_x, // width
                        launchParams.transmitter.numRays_y, // height
                        1) // depth
        );
        CUDA_CHECK( StreamSynchronize(cudaStream_) );
        launchParamsOnGpu.download(&launchParams, 1, cudaStream_);
        rayHitVerticesOnGpu_.numElements = launchParams.gpuAtomicNumHits;
        
        aiMesh *const pointCloudMesh = new aiMesh;
        pointCloudMesh->mNumVertices = launchParams.gpuAtomicNumHits;
        pointCloudMesh->mVertices = new aiVertex3D[launchParams.gpuAtomicNumHits];
        rayHitVerticesOnGpu_.download(pointCloudMesh->mVertices, pointCloudMesh->mNumVertices, cudaStream_);

        pointCloudScene->mMeshes = new (aiMesh*)[1];
        pointCloudScene-> mMeshes[0] = pointCloudMesh;
        pointCloudScene->mNumMeshes = 1;
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }

    return pointCloudScene;
}

void AssimpOptixRayTracer::createOptixDeviceContext_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    try {
        OptixDeviceContextOptions optixDeviceContextOptions;
        optixDeviceContextOptions.logCallbackFunction = &optixLogCallback;
        optixDeviceContextOptions.logCallbackLevel = 4;
        optixDeviceContextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        OptixDeviceContext optixDeviceContext;
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
		triangleArray.vertexBuffers = &(worldVertexBufferOnGpu_.d_pointer_ref());
		triangleArray.numVertices = (unsigned int) worldVertexBufferOnGpu_.numElements;
		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga55c6d96161ef202d48023a8a1d126102
		triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

		triangleArray.indexBuffer = indexBufferOnGpu_.d_pointer();
		triangleArray.numIndexTriplets = (unsigned int) indexBufferOnGpu_.numElements;
		https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa98b8fb6bf2d2455c310125f3fab74e6
		triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
		triangleArray.indexStrideInBytes = 0; // tightly packed

        const unsigned int triangle_flags[]{OPTIX_GEOMETRY_FLAG_NONE};
        triangleArray.flags = triangle_flags;
        triangleArray.numSbtRecords = 1;
	}

    // Compute buffer sizes for building acceleration structures
	OptixAccelBufferSizes optixAccelBufferSizes = {};
	OPTIX_CHECK(
		optixAccelComputeMemoryUsage(optixDeviceContext_,
		                             &optixAccelBuildOptions,
									 &optixBuildInput, 1,
									 &optixAccelBufferSizes)
	);
    assert(optixAccelBufferSizes.tempUpdateSizeInBytes == 0);

    // Allocate buffers for the build
    gasBuild_.resize(optixAccelBufferSizes.outputSizeInBytes);

    osc::CUDABuffer gasBuildTemp;
    gasBuildTemp.resize(optixAccelBufferSizes.tempSizeInBytes);

    OPTIX_CHECK(
        optixAccelBuild(optixDeviceContext_,
                        cudaStream_,
                        &optixAccelBuildOptions,
                        &optixBuildInput, 1,
                        gasBuildTemp.d_pointer(), gasBuildTemp.sizeInBytes,
                        gasBuild_.d_pointer(), gasBuild_.sizeInBytes,
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
        std::ifstream sourceCodeFile("ee259.cu");
        // https://stackoverflow.com/a/2912614
        std::istreambuf_iterator<char> sourceCodeFileIter{sourceCodeFile};
        sourceCode.assign(sourceCodeFileIter,
                          std::istreambuf_iterator<char>{});
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
    RaiiScopeLimitsLogger execLog(logInfo_, whereInProgram, __func__);
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
    optixPipelineCompileOptions.pipelineLaunchParamsVariableName = 0; // TODO
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga1171b332da08991dd9e6ef54b52b3ba4
    optixPipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    optixPipelineCompileOptions.allowOpacityMicromaps = 0;
    return optixPipelineCompileOptions;
}

OptixPipelineLinkOptions AssimpOptixRayTracer::makeOptixPipelineLinkOptions_()
{
    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_pipeline_link_options.html
    OptixPipelineLinkOptions optixPipelineLinkOptions;
    optixPipelineLinkOptions.maxTraceDepth = 5;
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

        OptixProgramGroupOptions *raygenProgramGroupOptions = nullptr; // no payload

        // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
        OPTIX_CHECK(
            optixProgramGroupCreate(optixDeviceContext_,
                                    &raygenProgramGroupDescription, 1, // array
                                    raygenProgramGroupOptions,
                                    logBuffer, &logBufferSize, // modified
                                    &raygenProgramGroup_)
        );
        optixLogBuffer[optixLogBufferSize] = '\0';
        logInfo_('[' + whereInProgram.toString() + "] " + (char*) optixLogBuffer);
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

        OptixProgramGroupOptions *hitGroupProgramGroupOptions = nullptr; // no payload

        // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
        OPTIX_CHECK(
            optixProgramGroupCreate(optixDeviceContext_,
                                    &hitGroupProgramGroupDesc, 1, // array
                                    hitGroupProgramGroupOptions,
                                    logBuffer, &logBufferSize,
                                    &hitGroupProgramGroup_)
        );
        optixLogBuffer[optixLogBufferSize] = '\0';
        logInfo_('[' + whereInProgram.toString() + "] " + (char*) optixLogBuffer);
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
        OptixProgramGroupDesc missProgramGroupDescription{};
        {
            // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_desc.html
            OptixProgramGroupDesc missProgramGroupDescription;
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gabca35b1218b4df575a5c42926da0d978
            missProgramGroupDescription.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa65e764f5bba97fda45d4453c0464596
            missProgramGroupDescription.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

            // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_single_module.html
            missProgramGroupDescription.miss.module = optixModule_;
            missProgramGroupDescription.miss.entryFunctionName = "__miss__noop";
        }
        OptixProgramGroupOptions *missProgramGroupOptions = nullptr; // no payload

        // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
        OPTIX_CHECK(
            optixProgramGroupCreate(optixDeviceContext_,
                                    &missProgramGroupDesc, 1, // array
                                    missProgramGroupOptions,
                                    logBuffer, &logBufferSize,
                                    &missProgramGroup_)
        );
        optixLogBuffer[optixLogBufferSize] = '\0';
        logInfo_('[' + whereInProgram.toString() + "] " + (char*) optixLogBuffer);
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
        makeRaygenProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, whereInSubProgram.advance();
        makeHitGroupProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, whereInSubProgram.advance();
        makeMissProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, whereInSubProgram.advance();

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

CUdeviceptr
AssimpOptixRayTracer::makeRaygenSbtRecord_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    CUdeviceptr result { nullptr };
    try {
        OPTIX_CHECK(
            optixSbtRecordPackHeader(raygenProgramGroup_, (void*) &raygenSbtRecord)
        );
        RayGenSbtRecord raygenSbtRecord;
        // Populate raygenSbtRecord.data here if needed in future
        osc::CUDABuffer raygenSbtRecordOnGpu;
        raygenSbtRecordOnGpu.allocAndUpload(&raygenSbtRecord, 1);
        result = (CUdeviceptr) raygenSbtRecordOnGpu.detach();
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
    return result;
}

std::pair<CUdeviceptr, unsigned int>
AssimpOptixRayTracer::makeMissSbtRecord_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    std::pair<CUdeviceptr, unsigned int> result { nullptr, 0 };
    try {
        OPTIX_CHECK(
            optixSbtRecordPackHeader(missProgramGroup_, (void*) &missSbtRecord)
        );
        osc::CUDABuffer missSbtRecordOnGpu;
        MissSbtRecord missSbtRecord;
        missSbtRecordOnGpu.allocAndUpload(&missSbtRecord, 1);
        result.first = (CUdeviceptr) missSbtRecordOnGpu.detach();
        result.second = missSbtRecordOnGpu.sizeInBytes;
    }
    catch(std::exception const& e) {
        logError_('[' + whereInProgram.toString() + "] ***ERROR*** " + e.what());
    }
    return result;
}

std::pair<CUdeviceptr, unsigned int>
AssimpOptixRayTracer::makeHitGroupSbtRecord_(ExecutionCursor whereInProgram)
{
    RaiiScopeLimitsLogger scopeLogger(logInfo_, whereInProgram, __func__);
    std::pair<CUdeviceptr, unsigned int> result;
    try {
        MissSbtRecord hitGroupSbtRecord;
        OPTIX_CHECK(
            optixSbtRecordPackHeader(hitGroupProgramGroup_, (void*) &hitGroupSbtRecord)
        );
        osc::CUDABuffer hitGroupSbtRecordOnGpu;
        hitGroupSbtRecordOnGpu.allocAndUpload(&hitGroupSbtRecord, 1);
        result.first = (CUdeviceptr) hitGroupSbtRecordOnGpu.detach();
        result.second = hitGroupSbtRecordOnGpu.sizeInBytes;
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
        sbt.raygenRecord = makeRaygenSbtRecord_(whereInSubProgram.advance());

        optixSbt_.missRecordCount = 1;
        std::tie(optixSbt_.missRecordBase, optixSbt_.missRecordStrideInBytes)
            = makeMissSbtRecord_(whereInSubProgram.advance());

        optixSbt_.hitgroupRecordCount = 1;
        std::tie(optixSbt_.hitgroupRecordBase, optixSbt_.hitgroupRecordStrideInBytes)
            = makeHitGroupSbtRecord_(whereInSubProgram.advance());

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

	cudaFree(0);
	cuCtx_ = 0;
    ExecutionCursor whereInSubProgram = whereInProgram.nextLevel();
	try {
		OPTIX_CHECK(optixInit() , whereInProgram);
		createOptixDeviceContext_(whereInSubProgram.advance()); // set optixDeviceContext_
        buildOptixPipeline_(whereInSubProgram.advance()); // set optixModule_, optixPipeline_
        makeOptixShaderBindingTable_(whereInSubProgram.advance()); // set optixSbt_
	}
	catch(std::exception& e) {
		logError(e.what());
	}
}

void AssimpOptixRayTracer::finalizeOptix_(ExecutionCursor whereInProgram)
{
    RaiiScopedLimitsLogger raiiScopeLogger(logInfo_, whereInProgram, __func__);
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
