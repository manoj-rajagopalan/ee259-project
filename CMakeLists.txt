project(ee259 LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.1)

SET(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CUDA_STANDARD_REQUIRED ON)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --compiler-bindir /usr/bin/g++-10") # compile error without this!

FIND_PACKAGE(OpenGL QUIET)
FIND_PACKAGE(Qt5 COMPONENTS Gui Widgets OpenGL QUIET)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# https://github.com/NVlabs/instant-ngp/issues/119
# TL;DR using g++ v10 (instead of 11) prevents a nvcc error with std::function<> when g++==11.x
SET(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
SET(CMAKE_C_COMPILER "/usr/bin/gcc-10")

SET(ee259_SRCS
    main.cpp
    mainwindow.hpp
    mainwindow.cpp
    AssimpOptixRayTracer.hpp
    AssimpOptixRayTracer.cu
    transmitter.hpp
    glview.hpp
    glview.cpp
    raytracingglview.hpp
    loggerview.hpp
    loggerview.cpp
    ExecutionCursor.hpp
    RaiiScopeLimitsLogger.hpp
    LaunchParams.hpp
    AsyncCudaBuffer.hpp
    cuda_optix_sentinel.h)

INCLUDE_DIRECTORIES(/home/rmanoj/Software/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty)

add_executable(ee259 ${ee259_SRCS} ${ee259_UI_SRCS} ${ee259_MOC_SRCS})

# https://cmake.org/cmake/help/latest/manual/cmake-qt.7.html#automoc
set_target_properties(ee259
                      PROPERTIES AUTOMOC ON)

# https://cmake.org/cmake/help/latest/command/qt_wrap_ui.html?highlight=qt5_wrap_ui
# https://cmake.org/cmake/help/latest/prop_tgt/AUTOUIC.html#prop_tgt:AUTOUIC
set_target_properties(ee259
                      PROPERTIES AUTOUIC ON)

set_target_properties(ee259
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# https://cmake.org/cmake/help/v3.22/prop_tgt/CUDA_ARCHITECTURES.html
set_target_properties(ee259
                      PROPERTIES CUDA_ARCHITECTURES "75")

target_link_libraries(ee259
                      Qt5::Gui
                      Qt5::Widgets
                      Qt5::OpenGL
                      ${OPENGL_LIBRARIES}
                      assimp
                      cudart
                      cuda)

# Refs:
# https://developer.nvidia.com/blog/building-cuda-applications-cmake/


add_executable(test_AssimpOptixRayTracer
               test_AssimpOptixRayTracer.cpp
               AssimpOptixRayTracer.hpp
               AssimpOptixRayTracer.cu
               transmitter.hpp
               ExecutionCursor.hpp
               LaunchParams.hpp
               AsyncCudaBuffer.hpp)
target_link_libraries(test_AssimpOptixRayTracer
                      Qt5::Core
                      assimp
                      cudart
                      cuda)
set_target_properties(test_AssimpOptixRayTracer
                      PROPERTIES AUTOMOC ON)
set_target_properties(test_AssimpOptixRayTracer
                      PROPERTIES CUDA_ARCHITECTURES "75")
