#ifndef VOXELIZATION_H
#define VOXELIZATION_H

#include "profiling.h"
#include <cstddef>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <strings.h>
#include <tuple>
#include <vector>
#include <voxels_grid.h>
#include <mesh/mesh.h>
#include <cuda_runtime.h>
#include <bounding_box.h>

#include <cub/cub.cuh>



__device__ __host__ inline float 
CalculateEdgeFunction(Position& V0, Position& V1, float y, float z)
{ return ((z - V0.Z) * (V1.Y - V0.Y)) - ((y - V0.Y) * (V1.Z - V0.Z)); }

__device__ __host__ inline Position
CalculateNormalOfEdgeFunction(Position& V0, Position& V1)
{ return Position(0, -(V1.Y - V0.Y), V1.Z - V0.Z);}


template <typename T>
__global__ void NaiveKernel(size_t trianglesSize, uint32_t* triangleCoords, 
                            Position* coords, VoxelsGrid<T, true> grid);


template <typename T>
__host__ void Sequential(VoxelsGrid<T, false>& grid, 
                         const std::vector<uint32_t>& triangleCoords,
                         const std::vector<Position>& coords);

template <typename T>
__global__ void TiledCalculateOverlap(const size_t numTriangles, uint32_t* triangleCoords,
                                      Position* coords, uint32_t* overlapPerTriangle,
                                      VoxelsGrid<T, true> grid)
{
 
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= numTriangles)
        return;

    Position V0 = coords[triangleCoords[(index * 3)]];
    Position V1 = coords[triangleCoords[(index * 3) + 1]];
    Position V2 = coords[triangleCoords[(index * 3) + 2]];

    Position facesVertices[3] = {V0, V1, V2};
    std::pair<float, float> BB_X, BB_Y, BB_Z;
    CalculateBoundingBox(std::span<Position>(&facesVertices[0], 3), BB_X, BB_Y, BB_Z);

    const float tileSize = grid.VoxelSize() * 4;
    int startY = static_cast<int>(std::floorf((BB_Y.first - grid.OriginY()) / tileSize));
    int endY   = static_cast<int>(std::ceilf((BB_Y.second - grid.OriginY()) / tileSize));
    int startZ = static_cast<int>(std::floorf((BB_Z.first - grid.OriginZ()) / tileSize));
    int endZ   = static_cast<int>(std::ceilf((BB_Z.second - grid.OriginZ()) / tileSize));

    Position N0 = CalculateNormalOfEdgeFunction(V0, V1);
    Position N1 = CalculateNormalOfEdgeFunction(V1, V2);
    Position N2 = CalculateNormalOfEdgeFunction(V2, V0);

    int numOverlap = 0;
    for(int y = startY; y < endY; ++y) 
    {
        for(int z = startZ; z < endZ; ++z) 
        {
            float minY = grid.OriginY() + (y * tileSize);
            float minZ = grid.OriginZ() + (z * tileSize);
            float maxY = minY + tileSize;
            float maxZ = minZ + tileSize;

            float E0 = CalculateEdgeFunction(V0, V1, N0.Y > 0 ? minY : maxY, N0.Z > 0 ? minZ : maxZ);
            float E1 = CalculateEdgeFunction(V1, V2, N1.Y > 0 ? minY : maxY, N1.Z > 0 ? minZ : maxZ);
            float E2 = CalculateEdgeFunction(V2, V0, N2.Y > 0 ? minY : maxY, N2.Z > 0 ? minZ : maxZ);

            bool ccw_test = (E0 >= 0 && E1 >= 0 && E2 >= 0);
            bool cw_test = (E0 <= 0 && E1 <= 0 && E2 <= 0);

            if (ccw_test) {
                numOverlap++;
                for(int i = 0; i < 4; ++i)
                    for(int j=0; j<4; ++j)
                        grid(0, (y*4) + i, (z*4)+j) = true;
            }
        }
    }

    overlapPerTriangle[index] = numOverlap;

    __syncthreads();
    if (index == 1) {
        int counter = 0;
        for(int i=0; i<numTriangles; i++) {
            LOG_INFO("%d: %d", i, overlapPerTriangle[i]);
            counter+= overlapPerTriangle[i];
        }
        LOG_INFO("%d", counter);
    }
}

template <typename T>
__global__ void TiledWorkQueuePopulation(const size_t numTriangles, uint32_t* triangleCoords,
                                         Position* coords, uint32_t* offsets, 
                                         uint32_t* workQueueKeys, uint32_t* workQueueValues,
                                         VoxelsGrid<T, true> grid, int workQueueSize)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= numTriangles)
        return;

    Position V0 = coords[triangleCoords[(index * 3)]];
    Position V1 = coords[triangleCoords[(index * 3) + 1]];
    Position V2 = coords[triangleCoords[(index * 3) + 2]];
        
    Position facesVertices[3] = {V0, V1, V2};
    std::pair<float, float> BB_X, BB_Y, BB_Z;
    CalculateBoundingBox(std::span<Position>(&facesVertices[0], 3), BB_X, BB_Y, BB_Z);

    const float tileSize = grid.VoxelSize() * 4;
    const uint tilePerSide = grid.VoxelsPerSide() / 4;

    int startY = static_cast<int>(std::floorf((BB_Y.first - grid.OriginY()) / tileSize));
    int endY   = static_cast<int>(std::ceilf((BB_Y.second - grid.OriginY()) / tileSize));
    int startZ = static_cast<int>(std::floorf((BB_Z.first - grid.OriginZ()) / tileSize));
    int endZ   = static_cast<int>(std::ceilf((BB_Z.second - grid.OriginZ()) / tileSize));

    Position N0 = CalculateNormalOfEdgeFunction(V0, V1);
    Position N1 = CalculateNormalOfEdgeFunction(V1, V2);
    Position N2 = CalculateNormalOfEdgeFunction(V2, V0);

    int numOverlap = 0;
    for(int y = startY; y < endY; ++y)
    {
        for(int z = startZ; z < endZ; ++z)
        {
            float minY = grid.OriginY() + (y * tileSize);
            float minZ = grid.OriginZ() + (z * tileSize);
            float maxY = minY + tileSize;
            float maxZ = minZ + tileSize;

            float E0 = CalculateEdgeFunction(V0, V1, N0.Y > 0 ? minY : maxY, N0.Z > 0 ? minZ : maxZ);
            float E1 = CalculateEdgeFunction(V1, V2, N1.Y > 0 ? minY : maxY, N1.Z > 0 ? minZ : maxZ);
            float E2 = CalculateEdgeFunction(V2, V0, N2.Y > 0 ? minY : maxY, N2.Z > 0 ? minZ : maxZ);

            bool ccw_test = (E0 >= 0 && E1 >= 0 && E2 >= 0);
            bool cw_test = (E0 <= 0 && E1 <= 0 && E2 <= 0);

            if (ccw_test || cw_test) {
                workQueueKeys[offsets[index] + numOverlap] = (z * tilePerSide) + y;
                workQueueValues[offsets[index] + numOverlap] = index;
                numOverlap++;

                //for(int i = 0; i < 4; ++i)
                    //for(int j=0; j<4; ++j)
                        //grid(0, (y*4) + i, (z*4)+j) = true;
            }
        }
    }

    __syncthreads();
    if(index == 1) {
        for (int i=0; i<workQueueSize; ++i) {
            LOG_INFO("%d: %d", workQueueKeys[i], workQueueValues[i]); 
        }
    }
}

template <typename T, int BATCH_SIZE = 14>
__global__ void TiledProcessing(uint32_t* triangleCoords, Position* coords, uint32_t* workQueue, 
                                uint32_t* activeTilesList, uint32_t* activeTilesListTriangleCount,
                                uint32_t* activeTilesListOffset, VoxelsGrid<T, true> grid)
{
    if (threadIdx.x >= 16)
        return;

    __shared__ Position sharedVertices[BATCH_SIZE * 3];
    
    const int activeTileIndex = blockIdx.x;
    const int voxelIndex = threadIdx.x;

    const int numTriangles = activeTilesListTriangleCount[activeTileIndex];     
    const int tileOffset = activeTilesListOffset[activeTileIndex];
    const int tileIndex = activeTilesList[activeTileIndex];

    int tileZ = tileIndex % (grid.VoxelsPerSide() / 4);
    int tileY = tileIndex / (grid.VoxelsPerSide() / 4);

    int voxelZ = (voxelIndex % 16) % 4;
    int voxelY = (voxelIndex % 16) / 4;

    int z = (tileZ * 4) + voxelZ;
    int y = (tileY * 4) + voxelY;

    float centerZ = grid.OriginZ() + (z * grid.VoxelSize()) + (grid.VoxelSize() / 2);
    float centerY = grid.OriginY() + (y * grid.VoxelSize()) + (grid.VoxelSize() / 2);


    for(int batch = 0; batch < numTriangles; batch += BATCH_SIZE)
    {
        if (voxelIndex < BATCH_SIZE && (voxelIndex + batch) < numTriangles) {
            const int indexV0 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3)];
            const int indexV1 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 1];
            const int indexV2 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 2];
            
            sharedVertices[(voxelIndex * 3)]     = coords[indexV0];
            sharedVertices[(voxelIndex * 3) + 1] = coords[indexV1];
            sharedVertices[(voxelIndex * 3) + 2] = coords[indexV2];
        }

        __syncthreads();

        //if(voxelIndex == 1) {
            //for(int i=0; i<numTriangles; ++i){
                //LOG_INFO("%d:\n V0: (%f, %f, %f)\nV1: (%f, %f, %f)\nV2: (%f,%f,%f)", activeTileIndex,
                         //sharedVertices[(i * 3)].X, sharedVertices[(i * 3)].Y, sharedVertices[(i * 3)].Z,
                         //sharedVertices[(i * 3) + 1].X, sharedVertices[(i * 3) + 1].Y, sharedVertices[(i * 3) + 1].Z,
                         //sharedVertices[(i * 3) + 2].X, sharedVertices[(i * 3) + 2].Y, sharedVertices[(i * 3) + 2].Z);
            //}
        //}

        int sharedSize = min(BATCH_SIZE, numTriangles - batch);
        //int voxelHalf = voxelIndex / 16;
        //int startTriangle = voxelHalf * (sharedSize / 2);
        //int endTriangle = (sharedSize * (voxelHalf + 1)) / 2;


        for(int triangle = 0; triangle < sharedSize; triangle++)
        {         
            Position V0 = sharedVertices[(triangle * 3)];
            Position V1 = sharedVertices[(triangle * 3) + 1];
            Position V2 = sharedVertices[(triangle * 3) + 2];

            float E0 = CalculateEdgeFunction(V0, V1, centerY, centerZ);
            float E1 = CalculateEdgeFunction(V1, V2, centerY, centerZ);
            float E2 = CalculateEdgeFunction(V2, V0, centerY, centerZ);

            bool ccw_test = (E0 >= 0 && E1 >= 0 && E2 >= 0);
            bool cw_test = (E0 <= 0 && E1 <= 0 && E2 <= 0);

            if (ccw_test || cw_test) {
                //grid(0, y, z) = true;
                //Position edge0 = V1 - V0;
                //Position edge1 = V2 - V0;

                //auto [A, B, C] = Position::Cross(edge0, edge1);
                //float D = Position::Dot({A, B, C}, V0);

                //float intersection = (D - (B * centerY) - (C * centerZ)) / A;

                //int startX = static_cast<int>((intersection - grid.OriginX()) / grid.VoxelSize());
                //int endX = grid.VoxelsPerSide();
                //for(int x = startX; x < endX; ++x)
                //grid(x, y, z) ^= true;
            }
        }
        __syncthreads();
    }
}

class Voxelization {

    static inline std::tuple<int, uint32_t*, Position*> InitKernel(
        const Mesh& mesh, int device, int blockSize, int trianglesSize) 
    { 
        uint32_t* devTriangles;
        gpuAssert(cudaMalloc((void**) &devTriangles, mesh.FacesCoords.size() * sizeof(uint32_t)));
        gpuAssert(cudaMemcpy(devTriangles, &mesh.FacesCoords[0], mesh.FacesCoords.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

        Position* devCoords;
        gpuAssert(cudaMalloc((void**) &devCoords, mesh.Coords.size() * sizeof(Position)));
        gpuAssert(cudaMemcpy(devCoords, &mesh.Coords[0], mesh.Coords.size() * sizeof(Position), cudaMemcpyHostToDevice));

        int gridSize = (trianglesSize + blockSize - 1) / blockSize;
        return {gridSize, devTriangles, devCoords};

    }
    static inline void WaitKernel(uint32_t* devTriangles, Position* devCoords) {
        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize(); 
        cudaFree(devTriangles);
        cudaFree(devCoords);
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
        WaitKernel(devTriangles, devCoords);
    }


    template<Types type, typename T>
    static void Compute(DeviceVoxelsGrid<T>& grid, const Mesh& mesh, int device, int blockSize)
    requires (type == Types::TAILED) 
    {
        PROFILING_SCOPE("Tiled Voxelization");
        const size_t numTriangles = mesh.FacesSize() * 2;
        const int gridSize = (numTriangles + blockSize - 1) / blockSize;

        // ----- Calculate Number of tiles overlap for each triangle -----
        printf("================ Calculate size for each triangle ======================\n");
        uint32_t* devTriangles;
        gpuAssert(cudaMalloc((void**) &devTriangles, mesh.FacesCoords.size() * sizeof(uint32_t)));
        gpuAssert(cudaMemcpy(devTriangles, &mesh.FacesCoords[0], mesh.FacesCoords.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

        Position* devCoords;
        gpuAssert(cudaMalloc((void**) &devCoords, mesh.Coords.size() * sizeof(Position)));
        gpuAssert(cudaMemcpy(devCoords, &mesh.Coords[0], mesh.Coords.size() * sizeof(Position), cudaMemcpyHostToDevice));

        uint32_t* devOverlapPerTriangle;
        gpuAssert(cudaMalloc((void**) &devOverlapPerTriangle, numTriangles * sizeof(uint32_t)));
        gpuAssert(cudaMemset(devOverlapPerTriangle, 0, numTriangles * sizeof(uint32_t)));
    
        TiledCalculateOverlap<T><<< gridSize, blockSize >>>(numTriangles, devTriangles, devCoords, devOverlapPerTriangle, grid.View());
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
        printf("================ Work queue filled ======================\n");
        int lastOverlapTriangle, lastOffset;
        gpuAssert(cudaMemcpy(&lastOverlapTriangle, devOverlapPerTriangle + (numTriangles - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&lastOffset, devOffsets + (numTriangles - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        const int workQueueSize = lastOverlapTriangle + lastOffset;
        LOG_INFO("%d", workQueueSize);

        uint32_t* devWorkQueueKeys;
        uint32_t* devWorkQueueValues;
        gpuAssert(cudaMalloc((void**) &devWorkQueueKeys, workQueueSize * sizeof(uint32_t)));
        gpuAssert(cudaMalloc((void**) &devWorkQueueValues, workQueueSize * sizeof(uint32_t)));
        
        TiledWorkQueuePopulation<T><<< gridSize, blockSize >>>(numTriangles, devTriangles, devCoords, devOffsets, devWorkQueueKeys, devWorkQueueValues, grid.View(), workQueueSize);
        cudaDeviceSynchronize();
        cudaFree(devOverlapPerTriangle);
        cudaFree(devOffsets);
        // ----- Alloc two workQueue first for tileId second for triangleId and fill them -----
        

        // ----- Sorte the two work queue previus created -----
        printf("================ Work queue sorted ======================\n");
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
        cudaFree(devTempStorage);
        devTempStorage = nullptr;
        tempStorageBytes = 0;

        uint32_t* keys = new uint32_t[workQueueSize];
        uint32_t* values = new uint32_t[workQueueSize];

        cudaMemcpy(keys, devWorkQueueKeysSorted, workQueueSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(values, devWorkQueueValuesSorted, workQueueSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        for(int i = 0; i < workQueueSize; ++i)
            LOG_INFO("%d: %d", keys[i], values[i]);
        // ----- Sorte the two work queue previus created -----
        

        // ----- From workQueueKey sorted we calculate activeTilesList, activeTilesOffset -----
        printf("================ Calculate active, offset, size ======================\n");
        uint32_t* devActiveTilesList;
        uint32_t* devActiveTilesNum;
        uint32_t* devActiveTilesTrianglesCount;
        const int NUM_TILED = (grid.View().VoxelsPerSide() * grid.View().VoxelsPerSide()) / 4;
        LOG_INFO("%d", NUM_TILED);

        gpuAssert(cudaMalloc((void**) &devActiveTilesList, NUM_TILED * sizeof(uint32_t)));
        gpuAssert(cudaMalloc((void**) &devActiveTilesTrianglesCount, NUM_TILED * sizeof(uint32_t)));
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
        uint32_t* devActiveTilesOffset;
        gpuAssert(cudaMalloc((void**) &devActiveTilesOffset, numActiveTiles * sizeof(uint32_t)));


        LOG_INFO("%d", numActiveTiles);
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
        devTempStorage = nullptr;
        tempStorageBytes = 0;

        uint32_t* activeTilesList = new uint32_t[numActiveTiles];
        uint32_t* activeTilesOffset = new uint32_t[numActiveTiles];
        uint32_t* activeTilesTrianglesCount = new uint32_t[numActiveTiles]();

        cudaMemcpy(activeTilesList, devActiveTilesList, numActiveTiles * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(activeTilesOffset, devActiveTilesOffset, numActiveTiles * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(activeTilesTrianglesCount, devActiveTilesTrianglesCount, numActiveTiles * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        for (int i = 0; i < numActiveTiles; ++i) {
            LOG_INFO("%d: start: %d, num: %d", activeTilesList[i], activeTilesOffset[i], activeTilesTrianglesCount[i]);
        }
        // ----- From workQueueKey sorted we calculate activeTilesList, activeTilesOffset -----

        TiledProcessing<T><<< numActiveTiles, 32 >>>(
            devTriangles, devCoords, 
            devWorkQueueValuesSorted, 
            devActiveTilesList, 
            devActiveTilesTrianglesCount, 
            devActiveTilesOffset, grid.View()
        );  

        cudaDeviceSynchronize();
    }
};


#endif // !VOXELIZATION_H
