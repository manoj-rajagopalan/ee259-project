project(ee259 LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.1)

FIND_PACKAGE(OpenGL QUIET)
FIND_PACKAGE(Qt5 COMPONENTS Gui Widgets OpenGL QUIET)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

SET(ee259_SRCS
    main.cpp
    mainwindow.hpp
    mainwindow.cpp
    AssimpOptixRayTracer.hpp
    AssimpOptixRayTracer.cpp
    transmitter.hpp
    glview.hpp
    glview.cpp
    loggerview.hpp
    loggerview.cpp
    ExecutionCursor.hpp
    RaiiScopeLimitsLogger.hpp
    LaunchParams.hpp
    cuda_optix_sentinel.h
    float3.hpp)

INCLUDE_DIRECTORIES(/home/rmanoj/Software/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty)

add_executable(ee259 ${ee259_SRCS} ${ee259_UI_SRCS} ${ee259_MOC_SRCS})

# https://cmake.org/cmake/help/latest/manual/cmake-qt.7.html#automoc
set_target_properties(ee259 PROPERTIES AUTOMOC ON)

# https://cmake.org/cmake/help/latest/command/qt_wrap_ui.html?highlight=qt5_wrap_ui
# https://cmake.org/cmake/help/latest/prop_tgt/AUTOUIC.html#prop_tgt:AUTOUIC
set_target_properties(ee259 PROPERTIES AUTOUIC ON)

set_target_properties(ee259 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ee259
                      Qt5::Gui
                      Qt5::Widgets
                      Qt5::OpenGL
                      ${OPENGL_LIBRARIES}
                      assimp
                      cudart)

# Refs:
# https://developer.nvidia.com/blog/building-cuda-applications-cmake/
