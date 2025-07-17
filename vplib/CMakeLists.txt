cmake_minimum_required(VERSION 3.20)

project(vplib LANGUAGES CXX CUDA VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


option(VPLIB_LOGGING "Turn on Log message" ON)
option(VPLIB_PROFILING "Turn on Profiling message" ON)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE CPP_FILES "${SRC_DIR}/*.cpp")
file(GLOB_RECURSE CU_FILES "${SRC_DIR}/*.cu")
set_source_files_properties(${CPP_FILES} PROPERTIES LANGUAGE CUDA)

add_library(vplib STATIC ${CPP_FILES} ${CU_FILES})

target_include_directories(vplib PUBLIC "${SRC_DIR}")

target_compile_definitions(vplib PUBLIC
    LOGGING=$<BOOL:${VPLIB_LOGGING}>
    PROFILING=$<BOOL:${VPLIB_PROFILING}>
)

target_compile_options(vplib PUBLIC
    $<$<CONFIG:Release>:-O3 -DNDEBUG --expt-relaxed-constexpr>
    $<$<CONFIG:Debug>:-g -O0 --expt-relaxed-constexpr>
)
