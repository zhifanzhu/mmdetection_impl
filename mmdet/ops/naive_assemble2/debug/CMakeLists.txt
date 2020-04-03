cmake_minimum_required(VERSION 3.13.2 FATAL_ERROR)
project(attgen)

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-g -G -O0 -std=c++11; -DVERBOSE")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

file( GLOB  cu  *.cu)

CUDA_ADD_EXECUTABLE(atten ${cu})
