cmake_minimum_required(VERSION 3.20)

project(cuda_voxelization LANGUAGES CXX CUDA)

set(LIBS_DIR "${PROJECT_SOURCE_DIR}/libs")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE CPP_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_executable(main ${CPP_FILES} ${CU_FILES})
set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CUDA)

target_include_directories(main PUBLIC "${LIBS_DIR}" "${SRC_DIR}" cublas)

target_compile_features(main PRIVATE cxx_std_23)
target_compile_features(main PRIVATE cuda_std_20)

option(LOGGING "Turn on Log message" ON)
option(PROFILING "Turn on Profiling message" ON)

target_compile_definitions(main PRIVATE LOGGING=$<BOOL:${LOGGING}>)
target_compile_definitions(main PRIVATE PROFILING=$<BOOL:${PROFILING}>)

target_compile_options(main PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0 --expt-relaxed-constexpr>
)
