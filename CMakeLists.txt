cmake_minimum_required(VERSION 3.20)

project(cuda_voxelization LANGUAGES CXX CUDA)

set(LIBS_DIR "${PROJECT_SOURCE_DIR}/libs")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")

set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug, Release, etc.)" FORCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCE_FILES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.cu")
file(GLOB_RECURSE CPP_FILES "${SRC_DIR}/*.cpp")

foreach(cpp_file ${CPP_FILES})
    set_source_files_properties(${cpp_file} PROPERTIES LANGUAGE CUDA)
endforeach()

add_executable(main ${SOURCE_FILES})
target_link_libraries(main) 
target_include_directories(main PUBLIC "${LIBS_DIR}")
