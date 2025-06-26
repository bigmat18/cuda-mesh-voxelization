#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include <driver_types.h>
#include <strings.h>
#include <vector>

#include <voxels_grid.h>
#include <mesh/mesh.h>
#include <profiling.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>


namespace Voxelization {

__device__ __host__ inline float 
CalculateEdgeFunctionZY(Position& V0, Position& V1, float y, float z)
{ return ((z - V0.Z) * (V1.Y - V0.Y)) - ((y - V0.Y) * (V1.Z - V0.Z)); }
    
__device__ __host__ inline Normal 
CalculateNormalZY(Position& V0, Position& V1)
    { return Position(0, V1.Z - V0.Z, -(V1.Y - V0.Y));}
    
__device__ __host__ inline Normal 
CalculateFaceNormal(Position& V0, Position& V1, Position& V2)
{ return Vec3<float>::Cross(V1 - V0, V2 - V1); }
    
    
template <typename T>
__host__ void Sequential(const std::vector<uint32_t>& triangleCoords,
                         const std::vector<Position>& coords,
                         VoxelsGrid<T, false>& grid);
    
template <typename T>
__global__ void NaiveKernel(const size_t numTriangles, 
                            const uint32_t* triangleCoords, 
                            const Position* coords, 
                            VoxelsGrid<T, true> grid);
    
template <typename T>
__global__ void CalculateNumOverlapPerTriangle(const size_t numTriangles, 
                                               const uint32_t* triangleCoords,
                                               const Position* coords, 
                                               const VoxelsGrid<T, true> grid,
                                               uint32_t* overlapPerTriangle);
    
template <typename T>
__global__ void WorkQueuePopulation(const size_t numTriangles, 
                                        const uint32_t* triangleCoords,
                                        const Position* coords, 
                                        const uint32_t* offsets, 
                                        const VoxelsGrid<T, true> grid, 
                                        const size_t workQueueSize,
                                        uint32_t* workQueueKeys, 
                                        uint32_t* workQueueValues);
    
    
template <typename T, int BATCH_SIZE = 14>
__global__ void TiledProcessing(const uint32_t* triangleCoords, 
                                    const Position* coords, 
                                    const uint32_t* workQueue, 
                                    const uint32_t* activeTilesList, 
                                    const uint32_t* activeTilesListTriangleCount,
                                    const uint32_t* activeTilesListOffset, 
                                    VoxelsGrid<T, true> grid);

inline void TileAssignmentCalculateOverlap() {};

inline void TileAssignmentExclusiveScan() {};

inline void TileAssignmentWorkQueuPopulation() {};

inline void TileAssignmentWorkQueueSorting() {};

inline void TileAssignmentCompactResult() {};

enum class Types {
    SEQUENTIAL, NAIVE, TILED
};

template<Types type, typename T>
void Compute(HostVoxelsGrid<T>& grid, const Mesh& mesh) 
requires (type == Types::SEQUENTIAL)
{
    PROFILING_SCOPE("Sequential Voxelization");
    Sequential<T>(mesh.FacesCoords, mesh.Coords, grid.View());
}


template<Types type, typename T>
void Compute(DeviceVoxelsGrid<T>& grid, const Mesh& mesh, int device, int blockSize) 
requires (type == Types::NAIVE)
{
    PROFILING_SCOPE("Naive Voxelization");
    const size_t numTriangles = mesh.FacesSize() * 2;
    const size_t gridSize = (numTriangles + blockSize - 1) / blockSize;

    uint32_t* devTrianglesCoords; 
    gpuAssert(cudaMalloc((void**) &devTrianglesCoords, mesh.FacesCoords.size() * sizeof(uint32_t)));
    gpuAssert(cudaMemcpy(devTrianglesCoords, &mesh.FacesCoords[0], mesh.FacesCoords.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    Position* devCoords;
    gpuAssert(cudaMalloc((void**) &devCoords, mesh.Coords.size() * sizeof(Position)));
    gpuAssert(cudaMemcpy(devCoords, &mesh.Coords[0], mesh.Coords.size() * sizeof(Position), cudaMemcpyHostToDevice));

    NaiveKernel<T><<< gridSize, blockSize >>>(numTriangles, devTrianglesCoords, devCoords, grid.View());

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize(); 
    cudaFree(devTrianglesCoords);
    cudaFree(devCoords);
}


template<Types type, typename T>
void Compute(DeviceVoxelsGrid<T>& grid, const Mesh& mesh, int device, int blockSize)
requires (type == Types::TILED) 
{
    PROFILING_SCOPE("Tiled Voxelization");
    const size_t numTriangles = mesh.FacesSize() * 2;
    const int gridSize = (numTriangles + blockSize - 1) / blockSize;

    // ----- Calculate Number of tiles overlap for each triangle -----
    uint32_t* devTrianglesCoords;
    gpuAssert(cudaMalloc((void**) &devTrianglesCoords, mesh.FacesCoords.size() * sizeof(uint32_t)));
    gpuAssert(cudaMemcpy(devTrianglesCoords, &mesh.FacesCoords[0], mesh.FacesCoords.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));    

    Position* devCoords;
    gpuAssert(cudaMalloc((void**) &devCoords, mesh.Coords.size() * sizeof(Position)));
    gpuAssert(cudaMemcpy(devCoords, &mesh.Coords[0], mesh.Coords.size() * sizeof(Position), cudaMemcpyHostToDevice));
 
    uint32_t* devOverlapPerTriangle;  
    gpuAssert(cudaMalloc((void**) &devOverlapPerTriangle, numTriangles * sizeof(uint32_t)));
    gpuAssert(cudaMemset(devOverlapPerTriangle, 0, numTriangles * sizeof(uint32_t)));
    
    CalculateNumOverlapPerTriangle<T><<< gridSize, blockSize >>>(
        numTriangles, 
        devTrianglesCoords, 
        devCoords, 
        grid.View(),
        devOverlapPerTriangle
    );

    cudaDeviceSynchronize();
    // ----- Calculate Number of tiles overlap for each triangle -----    

        
    // ----- Calculate the hypotetics offset in an array with (tile, triangle) -----
    uint32_t* devOffsets;
    gpuAssert(cudaMalloc((void**) &devOffsets, numTriangles * sizeof(uint32_t)));

    void* devTempStorage = nullptr;
    size_t tempStorageBytes = 0;
    cub::DeviceScan::ExclusiveSum(
        devTempStorage, tempStorageBytes,
        devOverlapPerTriangle, devOffsets, numTriangles
    );

    gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));
    cub::DeviceScan::ExclusiveSum(
        devTempStorage, tempStorageBytes,
        devOverlapPerTriangle, devOffsets, numTriangles
    );

    cudaDeviceSynchronize();
    cudaFree(devTempStorage);
    devTempStorage = nullptr;
    tempStorageBytes = 0;
    // ----- Calculate the hypotetics offset in an array with (tile, triangle) -----


    // ----- Alloc two workQueue first for tileId second for triangleId and fill them -----
    int lastOverlapTriangle, lastOffset;
    gpuAssert(cudaMemcpy(&lastOverlapTriangle, devOverlapPerTriangle + (numTriangles - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&lastOffset, devOffsets + (numTriangles - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const size_t workQueueSize = lastOverlapTriangle + lastOffset;

    uint32_t* devWorkQueueKeys;
    uint32_t* devWorkQueueValues;
    gpuAssert(cudaMalloc((void**) &devWorkQueueKeys, workQueueSize * sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**) &devWorkQueueValues, workQueueSize * sizeof(uint32_t)));
        
    WorkQueuePopulation<T><<< gridSize, blockSize >>>(
        numTriangles, devTrianglesCoords, devCoords, 
        devOffsets, grid.View(), workQueueSize,
        devWorkQueueKeys, devWorkQueueValues
    );

    cudaDeviceSynchronize();
    cudaFree(devOverlapPerTriangle);
    cudaFree(devOffsets);
    // ----- Alloc two workQueue first for tileId second for triangleId and fill them -----
        

    // ----- Sorte the two work queue previus created -----
    uint32_t* devWorkQueueKeysSorted;
    uint32_t* devWorkQueueValuesSorted;
    gpuAssert(cudaMalloc((void**) &devWorkQueueKeysSorted, workQueueSize * sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**) &devWorkQueueValuesSorted, workQueueSize * sizeof(uint32_t)));

    cub::DeviceRadixSort::SortPairs(
        devTempStorage, tempStorageBytes,
        devWorkQueueKeys, devWorkQueueKeysSorted,
        devWorkQueueValues, devWorkQueueValuesSorted, workQueueSize
    );

    gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));
        
    cub::DeviceRadixSort::SortPairs(
        devTempStorage, tempStorageBytes,
        devWorkQueueKeys, devWorkQueueKeysSorted,
        devWorkQueueValues, devWorkQueueValuesSorted, workQueueSize
    );
    
    cudaDeviceSynchronize();
    cudaFree(devWorkQueueValues);
    cudaFree(devWorkQueueKeys);
    cudaFree(devTempStorage);
    devTempStorage = nullptr;
    tempStorageBytes = 0;
    // ----- Sorte the two work queue previus created -----
        

    // ----- From workQueueKey sorted we calculate activeTilesList, activeTilesOffset -----
    uint32_t* devActiveTilesList;
    uint32_t* devActiveTilesNum;
    uint32_t* devActiveTilesTrianglesCount;
    const size_t numTiled = (grid.View().VoxelsPerSide() * grid.View().VoxelsPerSide()) / 4;

    gpuAssert(cudaMalloc((void**) &devActiveTilesList, numTiled * sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**) &devActiveTilesTrianglesCount, numTiled * sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**) &devActiveTilesNum, sizeof(uint32_t)));

    cub::DeviceRunLengthEncode::Encode(
        devTempStorage, tempStorageBytes,
        devWorkQueueKeysSorted, devActiveTilesList, 
        devActiveTilesTrianglesCount,
        devActiveTilesNum, workQueueSize
    );

    gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));

    cub::DeviceRunLengthEncode::Encode(
        devTempStorage, tempStorageBytes,
        devWorkQueueKeysSorted, devActiveTilesList, 
        devActiveTilesTrianglesCount,
        devActiveTilesNum, workQueueSize
    );

    cudaDeviceSynchronize();
    cudaFree(devTempStorage);
    devTempStorage = nullptr;
    tempStorageBytes = 0;

    uint32_t numActiveTiles;
    gpuAssert(cudaMemcpy(&numActiveTiles, devActiveTilesNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(devActiveTilesNum);

    uint32_t* devActiveTilesOffset;
    gpuAssert(cudaMalloc((void**) &devActiveTilesOffset, numActiveTiles * sizeof(uint32_t)));

    cub::DeviceScan::ExclusiveSum(
        devTempStorage, tempStorageBytes,
        devActiveTilesTrianglesCount, 
        devActiveTilesOffset, 
        numActiveTiles
    );

    gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));

    cub::DeviceScan::ExclusiveSum(
        devTempStorage, tempStorageBytes,
        devActiveTilesTrianglesCount, devActiveTilesOffset, numActiveTiles
    );

    cudaDeviceSynchronize();
    cudaFree(devTempStorage);
    devTempStorage = nullptr;
    tempStorageBytes = 0;
    // ----- From workQueueKey sorted we calculate activeTilesList, activeTilesOffset -----

    TiledProcessing<T><<< numActiveTiles, 32 >>>(
        devTrianglesCoords, 
        devCoords, 
        devWorkQueueValuesSorted, 
        devActiveTilesList, 
        devActiveTilesTrianglesCount, 
        devActiveTilesOffset, 
        grid.View()
    );  

    cudaDeviceSynchronize();
    cudaFree(devTrianglesCoords);
    cudaFree(devCoords);
    cudaFree(devWorkQueueKeysSorted);
    cudaFree(devWorkQueueValuesSorted);
    cudaFree(devActiveTilesList);
    cudaFree(devActiveTilesTrianglesCount);
    cudaFree(devActiveTilesOffset);
}

};

#endif // !VOXELIZATION_H
