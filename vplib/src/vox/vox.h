#ifndef VOX_H
#define VOX_H

#include "cuda_ptr.h"
#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <strings.h>
#include <vector>

#include <grid/voxels_grid.h>
#include <mesh/mesh.h>
#include <profiling.h>
#include <proc_utils.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>


namespace VOX {

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
__global__ void NaiveProcessing(const size_t numTriangles, 
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
                                    CudaPtr<uint32_t>& devTrianglesCoords, 
                                    CudaPtr<Position>& devCoords,
                                    CudaPtr<uint32_t>& devOverlapPerTriangle,
                                    VoxelsGrid<T, true>& grid);

void TileAssignmentExclusiveScan(const size_t numTriangles,
                                 CudaPtr<uint32_t>& devOffsets,
                                 CudaPtr<uint32_t>& devOverlapPerTriangle);

template <typename T>
void TileAssignmentWorkQueuePopulation(const size_t numTriangles,
                                       const size_t workQueueSize,
                                       CudaPtr<uint32_t>& devTrianglesCoords,
                                       CudaPtr<Position>& devCoords,
                                       CudaPtr<uint32_t>& devOffsets,
                                       VoxelsGrid<T, true>& grid,
                                       CudaPtr<uint32_t>& devWorkQueueKeys,
                                       CudaPtr<uint32_t>& devWorkQueueValues);

void TileAssignmentWorkQueueSorting(const size_t workQueueSize,
                                    CudaPtr<uint32_t>& devWorkQueueKeys,
                                    CudaPtr<uint32_t>& devWorkQueueValues,
                                    CudaPtr<uint32_t>& devWorkQueueKeysSorted,
                                    CudaPtr<uint32_t>& devWorkQueueValuesSorted);
    
void TileAssignmentCompactResult(const size_t workQueueSize,
                                 const size_t numTiled,
                                 uint32_t& numActiveTiles,
                                 CudaPtr<uint32_t>& devWorkQueueKeysSorted,
                                 CudaPtr<uint32_t>& devActiveTilesList,
                                 CudaPtr<uint32_t>& devActiveTilesTrianglesCount,
                                 CudaPtr<uint32_t>& devActiveTilesOffset);


template <Types type, typename T>
void Compute(HostVoxelsGrid<T>& grid, const Mesh& mesh); 

template <Types type, typename T>
void Compute(DeviceVoxelsGrid<T>& grid, const Mesh& mesh); 

};

#endif // !VOX_H
