#include "mainwindow_impl.hpp"
#include "cuda_optix_sentinel.h"

#include <cassert>
#include <fstream>
#include <iterator>

#include "mainwindow.hpp"

namespace {

std::unique_ptr<MainWindow> makeMainWindow()
{
	return std::make_unique<MainWindow_Impl>();
}

MainWindow_Impl *g_theMainWindow = nullptr; // to route OptiX error messages to

struct ExecutionLogger
{
    ExecutionLogger() = delete;
    explicit ExecutionLogger(std::string&& theScope)
    : scope(std::move(theScope))
    {
        g_theMainWindow->LogInfo(theScope + ": begin");
    }

    ~ExecutionLogger()
    {
        g_theMainWindow->LogInfo(theScope + ": end");
    }

    std::string scope;
};

} // namespace

MainWindow_Impl::MainWindow_Impl(QWidget *parent)
: MainWindow(parent)
{
	g_theMainWindow = this; // route all OptiX messages
}

/********************************************************************/
/****************************** OptiX *******************************/
/********************************************************************/

void optixLogCallback(unsigned int level, const char *tag, const char *msg, void* /* cbData */)
{
	g_theMainWindow->logOptixMessage(level, tag, msg);
}

void MainWindow_Impl::logOptixMessage(unsigned int level, const char *tag, const char *msg)
{
	QString q_msg;
	QTextStream q_msg_stream(&q_msg);
	q_msg_stream << "-OptiX " << level <<'-' << '[' << tag << ']' << ' ' << msg;
	q_msg_stream.flush();
	this->LogInfo(q_msg);
}


void MainWindow_Impl::createOptixDeviceContext_(const std::string& indent)
{
    ExecutionLogger execLog(indent + __func__);
	OptixDeviceContextOptions optixDeviceContextOptions;
	optixDeviceContextOptions.logCallbackFunction = &optixLogCallback;
	optixDeviceContextOptions.logCallbackLevel = 4;
	optixDeviceContextOptions.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
	OptixDeviceContext optixDeviceContext;
	OPTIX_CHECK(
		optixDeviceContextCreate(cuCtx_, &optixDeviceContextOptions, &optixDeviceContext_)
	);
}


void MainWindow_Impl::extractGeometry(std::vector<std::array<float,3>>& vertexBuffer,
                                  std::vector<std::array<uint16_t,3>>& indexBuffer,
                                  const std::string& indent)
{
     ExecutionLogger execLog(indent + __func__);

	if(!mScene->HasMeshes()) {
		this->LogError(tr("Scene has no meshes!"));
        return;
	}

    assert(mScene->mNumMeshes > 0);
    aiMesh const* mesh = mScene->mMeshes[0];
    vertexBuffer.clear();
    vertexBuffer.resize(mesh->mNumVertices);
    for(uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        const aiVector3D& vertex = mesh->mVertices[i];
        vertexBuffer[i][0] = vertex[0];
        vertexBuffer[i][1] = vertex[1];
        vertexBuffer[i][2] = vertex[2];
    }
    indexBuffer.clear();
    indexBuffer.resize(mesh->mNumFaces);
    for(uint32_t i = 0; i < mesh->mNumFaces; ++i) {
        const aiFace& face = mesh->mFaces[i];
        assert(face.mNumIndices == 3);
        assert(face.mIndices[0] < (1u << 16));
        indexBuffer[i][0] = (uint16_t) face.mIndices[0];
        assert(face.mIndices[1] < (1u << 16));
        indexBuffer[i][1] = (uint16_t) face.mIndices[1];
        assert(face.mIndices[2] < (1u << 16));
        indexBuffer[i][2] = (uint16_t) face.mIndices[2];
    }
}

CUdevicePtr copyToDevice(const void *buffer, size_t size)
{
    CUdevicePtr bufferDevicePtr;
	CUDA_CHECK( Malloc(&bufferDevicePtr, size) );
	CUDA_CHECK( Memcpy(bufferDevicePtr, buffer, size, cudaMemcpyHostToDevice) );
	return bufferDevicePtr;
}

template<typename T>
inline
CUdevicePtr copyToDevice(const std::vector<T>& v)
{
	size_t const memSize = v.size() * sizeof(T);
    return copyToDevice((const void*) v.data(), memSize);
}

OptixTraversableHandle MainWindow_Impl::buildOptixAccelerationStructures_(const std::string& indent)
{
    OptixTraversableHandle gasHandle; // return value

    ExecutionLogger execLog(indent + __func__);

	// https://raytracing-docs.nvidia.com/optix7/api/struct_optix_accel_build_options.html
	OptixAccessBuildOptions optixAccessBuildOptions = {};
	{
		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ggaff328b8278fbd1900558593599698bbaafa820662dca4a85935ab74c704665d93
		optixAccessBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga411c1c6d9f4d8e039ae19e9dea65958a
		optixAccessBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		// Ignore optixAccessBuildOptions.motionOptions
	}

	// https://raytracing-docs.nvidia.com/optix7/api/struct_optix_build_input.html
	OptixAccelBuildInput optixAccessBuildInput = {};
	{
		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga7932d1d9cdf33506a75a5da5d8a62d94
		optixAccessBuildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// https://raytracing-docs.nvidia.com/optix7/api/struct_optix_build_input_triangle_array.html
		std::vector<std::array<float,3>> vertexBuffer;
		std::vector<std::array<uint16_t,3>> indexBuffer;
		extractGeometry_(vertexBuffer, indexBuffer, "..");

        OptixAccelBuildInputTriangleArray& triangleArray = optixAccelBuildInput.triangleArray;
		triangleArray.vertexBuffers = copyToCuDevice(vertexBuffer);
		triangleArray.numVertices = (unsigned int) vertexBuffer.size();
		// https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga55c6d96161ef202d48023a8a1d126102
		triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

		triangleArray.indexBuffer = copyToCuDevice(indexBuffer);
		triangleArray.numIndexTriplets = (unsigned int) indexBuffer.size();
		https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa98b8fb6bf2d2455c310125f3fab74e6
		triangleArray.inputFormat = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
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
									 &optixAccessBufferSizes)
	);
    assert(optixAccelBufferSizes.tempUpdateSizeInBytes == 0);

    // Allocate buffers for the build
    gasBuildDevicePtr_ = nullptr;
    CUDA_CHECK( Malloc(&gasBuildDevicePtr_, optixAccessBufferSizes.outputSizeInBytes) );

    CUdevicePtr d_buildTemp = nullptr;
    CUDA_CHECK( Malloc(&d_buildTemp, optixAccessBufferSizes.tempSizeInBytes) );

    OPTIX_CHECK(
        optixAccelBuild(optixDeviceContext_,
                        0, // cuStream (current)
                        &optixAccelBuildOptions,
                        &optixAccelBuildInput, 1,
                        d_buildTemp,
                        optixAccelBufferSizes.tempSizeInBytes,
                        gasBuildDevicePtr_,
                        optixAccelBufferSizes.outputSizeInBytes,
                        &gasHandle,
                        nullptr, 0 // emitted properties
                        );
    );

    CUDA_CHECK( Free((void*) d_buildTemp) );

    return gasHandle;
}

std::string MainWindow_Impl::loadOptixModuleSourceCode_(const std::string& indent)
{
    ExecutionLogger execLog(indent + __func__);
    std::ifstream sourceCodeFile("ee259.cu");
    // https://stackoverflow.com/a/2912614
    std::string sourceCode(std::istreambuf_iterator<char>(sourceCodeFile),
                           std::istreambuf_iterator<char>());
    return sourceCode;
}

void MainWindow_Impl::makeOptixModule_(OptixPipelineCompileOptions optixPipelineCompileOptions,
                                       char* optixLogBuffer,
                                       size_t optixLogBufferSize,
                                       const std::string& indent)
{
    ExecutionLogger execLog(indent + __func__);

    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_module_compile_options.html
    OptixModuleCompileOptions optixModuleCompileΟptions{};
    optixModuleCompileOptions.maxxRegisterCount = 0; // unlimited
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaea8ecab8ad903804364ea246eefc79b2
    optixModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // none, for debug
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga2a921efc5016b2b567fa81ddb429e81a
    optixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    optixModuleCompileOptions.numBoundValues = 0;
    optixModuleCompileOptions.boundValues = 0; // null pointer
    optixModuleCompileOptions.numPayloadTypes = 0;
    optixModuleCompileOptions.payloadTypes = 0;

    std::string optixModuleSourceCode = loadOptixModuleSourceCode_(indent + "..");
    OPTIX_CHECK(
        optixModuleCreate(optixDeviceContext_,
                          &optixModuleCompileOptions,
                          &optixPipelineCompileOptions,
                          optixModuleSourceCode.c_str(),
                          optixModuleSourceCode.size(),
                          optixLogBuffer, &optixLogBufferSize,
                          &optixModule_)
    );
}

OptixPipelineCompileOptions MainWindow_Impl::makeOptixPipelineCompileOptions_()
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
    optixPipelineCompileFlags.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    optixPipelineCompileFlags.allowOpacityMicromaps = 0;
    return optixPipelineCompileOptions;
}

OptixPipelineLinkOptions MainWindow_Impl::makeOptixPipelineLinkOptions_()
{
    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_pipeline_link_options.html
    OptixPipelineLinkOptions optixPipelineLinkOptions;
    optixPipelineLinkOptions.maxTraceDepth = 5;
    return optixPipelineLinkOptions;
}

OptixProgramGroupDesc MainWindow_Impl::makeRayGenProgramGroupDescription_()
{
    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_desc.html
    OptixProgramGroupDesc raygenProgramGroupDescription;
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gabca35b1218b4df575a5c42926da0d978
    raygenProgramGroupDescription.kind = OPTIX_PROGRAM_GROUP_RAYGEN;
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa65e764f5bba97fda45d4453c0464596
    raygenProgramGroupDescription.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_single_module.html
    raygenProgramGroupDescription.module = optixModule_;
    raygenProgramGroupDescription.entryFunctionName = "__raygen__rg";

    return raygenProgramGroupDescription;
}

void MainWindow_Impl::makeRayGenProgramGroup_(char* optixLogBuffer,
                                              size_t optixLogBufferSize,
                                              const std::string& indent)
{
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga0b44fd0708cced8b77665cfac5453573

    ExecutionLogger execLog(indent + __func__);

    OptixProgramGroupDesc raygenProgramGroupDesc = makeRayGenProgramGroupDescription_();
    OptixProgramGroupOptions *raygenProgramGroupOptions = nullptr; // no payload

    // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
    OPTIX_CHECK(
        optixProgramGroupCreate(optixDeviceContext_,
                                &raygenProgramGroupDesc, 1, // array
                                raygenProgramGroupOptions,
                                optixLogBuffer, &optixLogBufferSize, // modified
                                &raygenProgramGroup_)
    );
    optixLogBuffer[optixLogBufferSize] = '\0';
    this->LogInfo(QString(optixLogBuffer))
}

OptixProgramGroupDesc MainWindow_Impl::makeHitGroupProgramGroupDescription_()
{
    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_desc.html
    OptixProgramGroupDesc hitGroupProgramGroupDescription;
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gabca35b1218b4df575a5c42926da0d978
    hitGroupProgramGroupDescription.kind = OPTIX_PROGRAM_GROUP_HITGROUP;
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#gaa65e764f5bba97fda45d4453c0464596
    hitGroupProgramGroupDescription.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_program_group_single_module.html
    hitGroupProgramGroupDescription.module = optixModule_;
    hitGroupProgramGroupDescription.entryFunctionName = "__closest_hit__ch";

    return hitGroupProgramGroupDescription;
}

void MainWindow_Impl::makeHitGroupProgramGroup_(char *optixLogBuffer,
                                                size_t optixLogBufferSize,
                                                const std::string& indent)
{
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga0b44fd0708cced8b77665cfac5453573

    ExecutionLogger execLog(indent + __func__);

    OptixProgramGroupDesc hitGroupProgramGroupDesc = makeHitGroupProgramGroupDescription_();
    OptixProgramGroupOptions *hitGroupProgramGroupOptions = nullptr; // no payload

    // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
    OPTIX_CHECK(
        optixProgramGroupCreate(optixDeviceContext_,
                                &hitGroupProgramGroupDesc, 1, // array
                                hitGroupProgramGroupOptions,
                                optixLogBuffer, &optixLogBufferSize,
                                &hitGroupProgramGroup_)
    );
    optixLogBuffer[optixLogBufferSize] = '\0';
    this->LogInfo(QString(optixLogBuffer));
}

void MainWindow_Impl::makeMissProgramGroup_(char *optixLogBuffer,
                                            size_t optixLogBufferSize,
                                            const std::string& indent)
{
    // https://raytracing-docs.nvidia.com/optix7/api/group__optix__types.html#ga0b44fd0708cced8b77665cfac5453573

    ExecutionLogger execLog(indent + __func__);

    OptixProgramGroupDesc missProgramGroupDesc = makeMissProgramGroupDescription_();
    OptixProgramGroupOptions *missProgramGroupOptions = nullptr; // no payload

    // https://raytracing-docs.nvidia.com/optix7/api/optix__host_8h.html#aa3515445a876a8a381ced002e4020d42
    OPTIX_CHECK(
        optixProgramGroupCreate(optixDeviceContext_,
                                &missProgramGroupDesc, 1, // array
                                missProgramGroupOptions,
                                optixLogBuffer, &optixLogBufferSize,
                                &missProgramGroup_)
    );
    optixLogBuffer[optixLogBufferSize] = '\0';
    this->LogInfo(QString(optixLogBuffer));
}

void MainWindow_Impl::buildOptixPipeline_(const std::string& indent)
{
    ExecutionLogger execLog(indent + __func__);

    size_t const kOptixLogBufferSize = 1024;
    char optixLogBuffer[kOptixLogBufferSize];
    optixLogBuffer[kOptixLogBufferSize-1] = '\0'; // ensure NULL-TERMINATED
    // ... Provide (kOptixLogBufferSize-1) as buffer-size in all uses to maintain this property.

    OptixPipelineCompileOptions optixPipelineCompileOptions =
        makeOptixPipelineCompileOptions_();
    makeOptixModule_(optixPipelineCompileOptions,
                     optixLogBuffer, kOptixLogBufferSize-1,
                     indent+"..");
    OptixPipelineLinkOptions optixPipelineLinkOptions =
        makeOptixPipelineLinkOptions_();
    makeRayGenProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, indent+"..");
    makeHitGroupProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, indent+"..");
    makeMissGroupProgramGroup_(optixLogBuffer, kOptixLogBufferSize-1, indent+"..");

    const unsigned int kNumProgramGroups = 3
    OptixProgramGroup optixProgramGroups[kNumProgramGroups] {
        raygenProgramGroup_, hitGroupProgramGroup_, missProgramGroup_
    };
    
    size_t optixLogBufferSize = kOptixLogBufferSize-1;
    OPTIX_CHECK(
        optixPipelineCreate(optixDeviceContext_,
                            &optixPipelineCompileOptions,
                            &optixPipelineLinkOptions,
                            optixProgramGroups, kNumProgramGroups,
                            optixLogBuffer, &optixLogBufferSize,
                            &optixPipeline_);
    );
    optixLogBuffer[optixLogBufferSize] = '\0';
    this->LogInfo(QString(optixLogBuffer));
}

template<typename T>
struct ShaderBindingTableRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT)
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData { /* TODO */};
using RayGenSbtRecord = ShaderBindingTableRecord<RayGenData>

struct MissData { /* TODO */ };
using MissSbtRecord = ShaderBindingTableRecord<MissData>;

struct HitGroupData { /* TODO */ };
using HitGroupSbtRecord = ShaderBindingTableRecord<HitGroupData>;

CUdevicePtr MainWindow_Impl::makeRayGenSbtRecord_(const std::string& indent)
{
    CUdevicePtr rayGenSbtRecordDevicePtr; // return value

    ExecutionLogger execLog(indent + __func__);
    RayGenSbtRecord rayGenSbtRecord;    
    OPTIX_CHECK(
        optixSbtRecordPackHeader(rayGenProgramGroup_, (void*) &rayGenSbtRecord)
    );

    // Populate rayGenSbtRecord.data here if needed in future

    // Transfer to device
    rayGenSbtRecordDevicePtr = copyToDevice((const void*) &rayGenSbtRecord, sizeof(rayGenSbtRecord));
    return rayGenSbtRecordDevicePtr;
}

std::pair<CUdevicePtr, unsigned int>
MainWindow_Impl::makeMissSbtRecord_(const std::string& indent)
{
    // Return values
    CUdevicePtr missSbtRecordDevicePtr = 0;
    unsigned int missSbtRecordSizeInBytes = 0;

    ExecutionLogger execLog(indent + __func__);

    MissSbtRecord missSbtRecord;
    OPTIX_CHECK(
        optixSbtRecordPackHeader(missProgramGroup_, (void*) missSbtRecord)
    );
    missSbtRecordSizeInBytes = sizeof(missSbtRecord);
    missSbtRecordDevicePtr = copyToDevice((void*), &missSbtRecord, missSbtRecordSizeInBytes);

    return std::make_pair(missSbtRecordDevicePtr, missSbtRecordSizeInBytes);
}

std::pair<CUdevicePtr, unsigned int>
MainWindow_Impl::makeHitGroupSbtRecord_(const std::string& indent)
{
    // Return values
    CUdevicePtr hitGroupSbtRecordDevicePtr = 0;
    unsigned int hitGroupSbtRecordSizeInBytes = 0;

    ExecutionLogger execLog(indent + __func__);

    MissSbtRecord hitGroupSbtRecord;
    OPTIX_CHECK(
        optixSbtRecordPackHeader(hitGroupProgramGroup_, (void*) hitGroupSbtRecord)
    );
    hitGroupSbtRecordSizeInBytes = sizeof(hitGroupSbtRecord);
    hitGroupSbtRecordDevicePtr = copyToDevice((void*), &hitGroupSbtRecord, hitGroupSbtRecordSizeInBytes);

    return std::make_pair(hitGroupSbtRecordDevicePtr, hitGroupSbtRecordSizeInBytes);
}

OptixShaderBindingTable MainWindow_Impl::makeOptixShaderBindingTable_(const std::string& indent)
{
    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_shader_binding_table.html
    OptixShaderBindingTable sbt; // return value

    ExecutionLogger execLog(indent + __func__);

    sbt.raygenRecord = makeRayGenSbtRecord_(indent + "..");

    sbt.missRecordCount = 1;
    std::tie(sbt.missRecordBase, sbt.missRecordStrideInBytes)
        = makeMissSbtRecord_(indent + "..");

    sbt.hitGroupRecordCount = 1;
    std::tie(sbt.hitGroupRecordBase, sbt.hitGroupRecordStrideInBytes)
        = makeHitGroupSbtRecord_(indent + "..");

    sbt.callablesRecordBase = 0;
    sbt.callablesRecordCount = 0;
    sbt.callablesRecordStrideInBytes = 0;

    sbt.exceptionRecord = 0;

    return sbt;
}

void MainWindow_Impl::launchOptix_(const OptixShaderBindingTable& sbt,
                                   const std::string& indent)
{
    OPTIX_CHECK(
        optixLaunch(optixPipeline_,
                    cuStream_, // TODO
                    optixPipelineParams, // TODO
                    optixPipelineParamsSize, // TODO
                    &sbt,
                    width, // TODO
                    height, // TODO
                    depth) // TODO
    );
}

void MainWindow_Impl::initializeOptix_()
{
    ExecutionLogger execLog(__func__);

	cudaFree(0);
	cuCtx_ = 0;
    const std::string indent = "..";

	try {
		OPTIX_CHECK(optixInit());
		createOptixDeviceContext_("..");
		OptixTraversableHandle gas_handle = buildOptixAccelerationStructures_(indent);
        buildOptixPipeline_(indent);
        buildOptixShaderBindingTable_(indent);
        OptixShaderBindingTable sbt = makeOptixShaderBindingTable_(indent);
        launchOptix_(sbt, indent);
		// Shader binding table

		// Launch

		CUstream cuStream; // TODO
		CUdevicePtr pipelineParamsOnDevice; // TODO
		size_t const pipelineParamSize; // TODO
		OptixShaderBindingTable optixShaderBindingTable = {}; // TODO
		{

		}
		unsigned int raygenWidth; // TODO
		unsigned int raygenHeight; // TODO
		unsigned int raygenDepth; // TODO
		OPTIX_CHECK(optixLaunch(optixPipeline,
		                        cuStream,
								pipelineParamsOnDevice,
								pipelineParamsSize,
								&optixShaderBindingTable,
								raygenWidth,
								raygenHeight,
								raygenDepth));


	}
	catch(std::exception& e) {
		this->LogError(QString(e.what()));
	}
}

// TODO - call this with exception handling
void MainWindow_Impl::releaseOptix_()
{
    OPTIX_CHECK( optixProgramGroupDestroy(raygenProgramGroup_) );
    OPTIX_CHECK( optixProgramGroupDestroy(hitGroupProgramGroup_) );
    OPTIX_CHECK( optixProgramGroupDestroy(missProgramGroup_) );
    CUDA_CHECK( Free((void*) gasBuildDevicePtr_) );
	optixDestroyContext(optixDeviceContext_);

}