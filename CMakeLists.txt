cmake_minimum_required(VERSION 3.18)

project(cuda-surface-voxelization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 61)
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)

add_executable(main src/main.cpp)

set_source_files_properties(src/main.cpp PROPERTIES
    LANGUAGE CUDA
)

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-w -O3 --ptxas-options=-v>
)

target_link_libraries(main PRIVATE CUDA::cudart)