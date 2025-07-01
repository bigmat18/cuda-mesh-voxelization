#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
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

template <typename T>
void TileAssignmentCalculateOverlap(const size_t numTriangles, 
                                    const Mesh& mesh,
                                    uint32_t** devTrianglesCoords, 
                                    Position** devCoords,
                                    uint32_t** devOverlapPerTriangle,
                                    VoxelsGrid<T, true>& grid);

void TileAssignmentExclusiveScan(const size_t numTriangles,
                                 uint32_t** devOffsets,
                                 uint32_t** devOverlapPerTriangle);

template <typename T>
void TileAssignmentWorkQueuePopulation(const size_t numTriangles,
                                       const size_t workQueueSize,
                                       uint32_t** devTrianglesCoords,
                                       Position** devCoords,
                                       uint32_t** devOffsets,
                                       VoxelsGrid<T, true>& grid,
                                       uint32_t** devWorkQueueKeys,
                                       uint32_t** devWorkQueueValues);

void TileAssignmentWorkQueueSorting(const size_t workQueueSize,
                                    uint32_t** devWorkQueueKeys,
                                    uint32_t** devWorkQueueValues,
                                    uint32_t** devWorkQueueKeysSorted,
                                    uint32_t** devWorkQueueValuesSorted);
    
void TileAssignmentCompactResult(const size_t workQueueSize,
                                 const size_t numTiled,
                                 uint32_t& numActiveTiles,
                                 uint32_t** devWorkQueueKeysSorted,
                                 uint32_t** devActiveTilesList,
                                 uint32_t** devActiveTilesTrianglesCount,
                                 uint32_t** devActiveTilesOffset);

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
void Compute(DeviceVoxelsGrid<T>& grid, const Mesh& mesh, int device) 
requires (type == Types::NAIVE)
{
    PROFILING_SCOPE("Naive Voxelization");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const size_t numTriangles = mesh.FacesSize() * 2;
    const size_t blockSize = NextPow2(numTriangles, prop.maxThreadsDim[0] / 2);
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
void Compute(DeviceVoxelsGrid<T>& grid, const Mesh& mesh, int device)
requires (type == Types::TILED) 
{
    PROFILING_SCOPE("Tiled Voxelization");
    const size_t numTriangles = mesh.FacesSize() * 2;


    // =========================================================================
    uint32_t* devTrianglesCoords = nullptr;
    Position* devCoords = nullptr;
    uint32_t* devOverlapPerTriangle = nullptr;
    TileAssignmentCalculateOverlap<T>(
        numTriangles, mesh, &devTrianglesCoords, 
        &devCoords, &devOverlapPerTriangle, grid.View()
    );
    // =========================================================================


    // =========================================================================
    uint32_t* devOffsets = nullptr;
    TileAssignmentExclusiveScan(numTriangles, &devOffsets, &devOverlapPerTriangle);
    // =========================================================================



    // =========================================================================
    int lastOverlapTriangle, lastOffset;
    gpuAssert(cudaMemcpy(&lastOverlapTriangle, devOverlapPerTriangle + (numTriangles - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&lastOffset, devOffsets + (numTriangles - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    const size_t workQueueSize = lastOverlapTriangle + lastOffset;

    uint32_t* devWorkQueueKeys = nullptr;
    uint32_t* devWorkQueueValues = nullptr;

    TileAssignmentWorkQueuePopulation<T>(
        numTriangles, workQueueSize, &devTrianglesCoords, 
        &devCoords, &devOffsets, grid.View(), 
        &devWorkQueueKeys, &devWorkQueueValues
    );
    // =========================================================================
        


    // =========================================================================
    uint32_t* devWorkQueueKeysSorted = nullptr;
    uint32_t* devWorkQueueValuesSorted = nullptr;

    TileAssignmentWorkQueueSorting(
        workQueueSize, &devWorkQueueKeys, &devWorkQueueValues, 
        &devWorkQueueKeysSorted, &devWorkQueueValuesSorted
    );    
    // =========================================================================



    // =========================================================================
    const size_t numTiled = (grid.View().VoxelsPerSide() * grid.View().VoxelsPerSide()) / 4;
    uint32_t numActiveTiles = 0;
    uint32_t* devActiveTilesList = nullptr;
    uint32_t* devActiveTilesTrianglesCount = nullptr;
    uint32_t* devActiveTilesOffset = nullptr;

    TileAssignmentCompactResult(
        workQueueSize, numTiled, numActiveTiles, 
        &devWorkQueueKeysSorted, &devActiveTilesList, 
        &devActiveTilesTrianglesCount, &devActiveTilesOffset
    );
    // =========================================================================

    TiledProcessing<T><<< numActiveTiles, 32 >>>(
        devTrianglesCoords, 
        devCoords, 
        devWorkQueueValuesSorted, 
        devActiveTilesList, 
        devActiveTilesTrianglesCount, 
        devActiveTilesOffset, 
        grid.View()
    );  

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cudaFree(devTrianglesCoords);
    cudaFree(devCoords);
    cudaFree(devOverlapPerTriangle);
    cudaFree(devOffsets);
    cudaFree(devWorkQueueValues);
    cudaFree(devWorkQueueKeys);
    cudaFree(devWorkQueueKeysSorted);
    cudaFree(devWorkQueueValuesSorted);
    cudaFree(devActiveTilesList);
    cudaFree(devActiveTilesTrianglesCount);
    cudaFree(devActiveTilesOffset);
}

};

#endif // !VOXELIZATION_H
