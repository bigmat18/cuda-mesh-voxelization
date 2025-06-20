#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include "profiling.h"
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>
#include <voxels_grid.h>
#include <mesh/mesh.h>
#include <cuda_runtime.h>


__device__ __host__ inline std::tuple<float, float, float> 
CalculateEdgeTerms(Position& V0, Position& V1) {
    float A = V0.Z - V1.Z;
    float B = V1.Y - V0.Y;
    float C = -A * V0.Y - B * V0.Z;    
    return {A, B, C};
}

__device__ __host__ inline float 
CalculateEdgeFunction(float A, float B, float C, float y, float z)
{ return (A * y) + (B * z) + C; }


template <typename T>
__global__ void NaiveKernel(size_t trianglesSize, uint32_t* triangleCoords, 
                            Position* coords, VoxelsGrid<T, true> grid);


template <typename T>
__host__ void Sequential(VoxelsGrid<T, false>& grid, 
                         const std::vector<uint32_t>& triangleCoords,
                         const std::vector<Position>& coords);

class Voxelization {

    static inline std::tuple<int, uint32_t*, Position*> InitKernel(
        const Mesh& mesh, int device, int blockSize, int trianglesSize) 
    {
        int maxBlockDimX, maxBlockDimY, maxBlockDimZ;
        cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, device);
        cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, device);
        cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, device);
   
        uint32_t* devTriangles;
        gpuAssert(cudaMalloc((void**) &devTriangles, mesh.FacesCoords.size() * sizeof(uint32_t)));
        gpuAssert(cudaMemcpy(devTriangles, &mesh.FacesCoords[0], mesh.FacesCoords.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

        Position* devCoords;
        gpuAssert(cudaMalloc((void**) &devCoords, mesh.Coords.size() * sizeof(Position)));
        gpuAssert(cudaMemcpy(devCoords, &mesh.Coords[0], mesh.Coords.size() * sizeof(Position), cudaMemcpyHostToDevice));

        int gridSize = (trianglesSize + blockSize - 1) / blockSize;
        return {gridSize, devTriangles, devCoords};

    }
    static inline void WaitKernel() {
        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize(); 
    }

public:
    enum class Types {
        SEQUENTIAL, NAIVE, TAILED
    };

    template<Types type, typename T>
    static void Compute(HostVoxelsGrid<T>& grid, const Mesh& mesh) 
    requires (type == Types::SEQUENTIAL)
    {
        PROFILING_SCOPE("Sequential Voxelization");
        Sequential<T>(grid.View(), mesh.FacesCoords, mesh.Coords);
    }


    template<Types type, typename T>
    static void Compute(DeviceVoxelsGrid<T>& grid, const Mesh& mesh, int device, int blockSize) 
    requires (type == Types::NAIVE)
    {
        PROFILING_SCOPE("Naive Voxelization");
        const size_t trianglesSize = mesh.FacesSize() * 2;  
        auto[gridSize, devTriangles, devCoords] = InitKernel(mesh, device, blockSize, trianglesSize);
        NaiveKernel<T><<< gridSize, blockSize >>>(trianglesSize, devTriangles, devCoords, grid.View());
        WaitKernel();
    }


    template<Types type, typename T>
    static void Compute(DeviceVoxelsGrid<T>& grid, const Mesh& mesh, int device, int blockSize, int tileSize)
    requires (type == Types::TAILED) 
    {
    }
};


#endif // !VOXELIZATION_H,
