cmake_minimum_required(VERSION 3.20)

project(cli_app LANGUAGES CXX CUDA VERSION 1.0)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE CU_FILES "*.cu")
set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CUDA)

add_executable(cli ${CPP_FILES} ${CU_FILES})

target_link_libraries(cli PRIVATE vplib cxxopts)
