cmake_minimum_required(VERSION 3.20)

project(cuda_csg_voxel LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_subdirectory("${CMAKE_SOURCE_DIR}/libs/cxxopts")
add_subdirectory(vplib)
add_subdirectory(apps/cli)


