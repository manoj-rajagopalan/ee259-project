#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

// Header files, Assimp.
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/matrix4x4.h>

#include "transmitter.hpp"
#include "AssimpOptixRayTracer.hpp"

int main(void)
{
    std::mutex commandMutex;
    std::condition_variable commandCV;
    std::mutex resultMutex;

    Assimp::Importer assimpImporter;
    aiScene const *scene = assimpImporter.ReadFile("Cube.ply",
                                                   aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_ValidateDataStructure | \
												   aiProcess_GenUVCoords | aiProcess_TransformUVCoords | aiProcess_FlipUVs);
    
    manojr::AssimpOptixRayTracer assimpOptixRayTracer(commandMutex, commandCV, resultMutex);

	std::function<void(std::string const&)> infoToCout = [](std::string const& s) { std::cout << s << std::endl; };
	std::function<void(std::string const&)> errToCerr = [](std::string const& s) { std::cerr << s << std::endl; };
	assimpOptixRayTracer.registerLoggingFunctions(infoToCout, errToCerr);

    std::thread rayTracingThread(
        [&](){ assimpOptixRayTracer.eventLoop(); }
    );

    aiMatrix4x4 identityTransform{};
    manojr::Transmitter transmitter;
    transmitter.position = make_float3(0,0,0);
    transmitter.xUnitVector = make_float3(1,0,0);
    transmitter.yUnitVector = make_float3(0,1,0);
    transmitter.zUnitVector = make_float3(0,0,1);
    transmitter.width = 1.0f;
    transmitter.height = 1.0f;
    transmitter.focalLength = 1.0f;
    transmitter.numRays_x = 2;
    transmitter.numRays_y = 2;

    {
        std::unique_lock<std::mutex> commandLock(commandMutex);
        assimpOptixRayTracer.setScene(scene);
        assimpOptixRayTracer.setSceneTransform(&identityTransform);
        assimpOptixRayTracer.setTransmitter(&transmitter);
        assimpOptixRayTracer.setTransmitterTransform();
        commandCV.notify_one();
    }
    std::string s;
    std::cout << "Waiting for input: " << std::endl;
    std::cin >> s;
    {
        std::unique_lock<std::mutex> resultLock(resultMutex);
        aiScene const *temp_result = assimpOptixRayTracer.rayTracingResult(); // pointer-ownership transferred
        std::unique_ptr<aiScene const> result{temp_result};

        std::cout << "Waiting for input: " << std::endl;
        std::cin >> s;
    }
    {
        std::unique_lock commandLock(commandMutex);
        assimpOptixRayTracer.quit();
    }
    rayTracingThread.join();
    return 0;
}
