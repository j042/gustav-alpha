cmake_minimum_required(VERSION 3.11.0)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(splinelibpy)
set(CMAKE_CXX_STANDARD 20)

option(COMPILE_SPLINELIB "Compile splinelib together." ON)

set(SOURCE_DIR "src")
include_directories(${SOURCE_DIR})

add_subdirectory(third_party/pybind11)
pybind11_add_module(splinelibpy ${SOURCES} "${SOURCE_DIR}/splinelib.cpp")

if(COMPILE_SPLINELIB)
    set(GOOGLETEST OFF)
    add_subdirectory(third_party/SplineLib)
    target_link_libraries(splinelibpy PRIVATE splines fittings input_output)
else()
    find_package(SplineLib REQUIRED)
    target_link_libraries(splinelibpy PRIVATE SplineLib::splines SplineLib::fittings SplineLib::input_output)
endif()

target_compile_definitions(splinelibpy PRIVATE __PYTHON__)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(splinelibpy PRIVATE -ffast-math -mtune=native -O3)
endif()
