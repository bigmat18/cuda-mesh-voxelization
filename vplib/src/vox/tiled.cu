#include "cuda_ptr.h"
#include "grid/voxels_grid.h"
#include "mesh/mesh.h"
#include "proc_utils.h"
#include "profiling.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vox/vox.h>
#include <bounding_box.h>

namespace VOX {

template <typename T>
void TileAssignmentCalculateOverlap(const size_t numTriangles, 
                                    const Mesh& mesh,
                                    CudaPtr<uint32_t>& devTrianglesCoords, 
                                    CudaPtr<Position>& devCoords,
                                    CudaPtr<uint32_t>& devOverlapPerTriangle,
                                    VoxelsGrid<T, true>& grid) 
{
    {
        PROFILING_SCOPE("TiledVox::Memory");
        devTrianglesCoords.CopyFromHost(&mesh.FacesCoords[0], mesh.FacesCoords.size());
        devCoords.CopyFromHost(&mesh.Coords[0], mesh.Coords.size());
        devOverlapPerTriangle = CudaPtr<uint32_t>(numTriangles); 
        devOverlapPerTriangle.SetMemoryToZero();
    }

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::CalculateOverlap");
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        const size_t blockSize = NextPow2(numTriangles, prop.maxThreadsDim[0] / 2);
        const size_t gridSize = (numTriangles + blockSize - 1) / blockSize;
        CalculateNumOverlapPerTriangle<T><<< gridSize, blockSize >>>(
            numTriangles, devTrianglesCoords.get(), 
            devCoords.get(), grid, devOverlapPerTriangle.get()
        );

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }
}

void TileAssignmentExclusiveScan(const size_t numTriangles,
                                 CudaPtr<uint32_t>& devOffsets,
                                 CudaPtr<uint32_t>& devOverlapPerTriangle) 
{
    {
        PROFILING_SCOPE("TiledVox::Memory");
        devOffsets = CudaPtr<uint32_t>(numTriangles);
    }

    void* devTempStorage = nullptr;
    size_t tempStorageBytes = 0;

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::ExclusiveScan");
        cub::DeviceScan::ExclusiveSum(
            devTempStorage, tempStorageBytes,
            devOverlapPerTriangle.get(), devOffsets.get(), numTriangles
        );
    }

    {
        PROFILING_SCOPE("TiledVox::Memory");
        gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));
    }

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::ExclusiveScan");
        cub::DeviceScan::ExclusiveSum(
            devTempStorage, tempStorageBytes,
            devOverlapPerTriangle.get(), devOffsets.get(), numTriangles
        );


        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        cudaFree(devTempStorage);
    }
}

template <typename T>
void TileAssignmentWorkQueuePopulation(const size_t numTriangles,
                                       const size_t workQueueSize,
                                       CudaPtr<uint32_t>& devTrianglesCoords,
                                       CudaPtr<Position>& devCoords,
                                       CudaPtr<uint32_t>& devOffsets,
                                       VoxelsGrid<T, true>& grid,
                                       CudaPtr<uint32_t>& devWorkQueueKeys,
                                       CudaPtr<uint32_t>& devWorkQueueValues) 
{
    {
        PROFILING_SCOPE("TiledVox::Memory");
        devWorkQueueKeys = CudaPtr<uint32_t>(workQueueSize);
        devWorkQueueValues = CudaPtr<uint32_t>(workQueueSize);
    }
 
    {
        PROFILING_SCOPE("TiledVox::TileAssignment::WorkQueuePopulation");
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        const size_t blockSize = NextPow2(numTriangles, prop.maxThreadsDim[0] / 2);
        const size_t gridSize = (numTriangles + blockSize - 1) / blockSize;

        WorkQueuePopulation<T><<< gridSize, blockSize >>>(
            numTriangles, devTrianglesCoords.get(), devCoords.get(), 
            devOffsets.get(), grid, workQueueSize,
            devWorkQueueKeys.get(), devWorkQueueValues.get()
        );

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }
}

void TileAssignmentWorkQueueSorting(const size_t workQueueSize,
                                    CudaPtr<uint32_t>& devWorkQueueKeys,
                                    CudaPtr<uint32_t>& devWorkQueueValues,
                                    CudaPtr<uint32_t>& devWorkQueueKeysSorted,
                                    CudaPtr<uint32_t>& devWorkQueueValuesSorted) 
{
    PROFILING_SCOPE("TiledVox::TileAssignment::WorkQueueSorting");
    void* devTempStorage = nullptr;
    size_t tempStorageBytes = 0;

    {
        PROFILING_SCOPE("TiledVox::Memory");
        devWorkQueueKeysSorted = CudaPtr<uint32_t>(workQueueSize);
        devWorkQueueValuesSorted = CudaPtr<uint32_t>(workQueueSize);
    }

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::WorkQueueSorting");
        cub::DeviceRadixSort::SortPairs(
            devTempStorage, tempStorageBytes,
            devWorkQueueKeys.get(), devWorkQueueKeysSorted.get(),
            devWorkQueueValues.get(), devWorkQueueValuesSorted.get(), workQueueSize
        );
    }

    {
        PROFILING_SCOPE("TiledVox::Memory");
        gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));
    }

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::WorkQueueSorting");
        cub::DeviceRadixSort::SortPairs(
            devTempStorage, tempStorageBytes,
            devWorkQueueKeys.get(), devWorkQueueKeysSorted.get(),
            devWorkQueueValues.get(), devWorkQueueValuesSorted.get(), workQueueSize
        );

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        cudaFree(devTempStorage);
    }
}

void TileAssignmentCompactResult(const size_t workQueueSize,
                                 const size_t numTiled,
                                 uint32_t& numActiveTiles,
                                 CudaPtr<uint32_t>& devWorkQueueKeysSorted,
                                 CudaPtr<uint32_t>& devActiveTilesList,
                                 CudaPtr<uint32_t>& devActiveTilesTrianglesCount,
                                 CudaPtr<uint32_t>& devActiveTilesOffset) 
{
    CudaPtr<uint32_t> devActiveTilesNum;
    {
        PROFILING_SCOPE("TiledVox::Memory");
        devActiveTilesNum = CudaPtr<uint32_t>(1);
        devActiveTilesList = CudaPtr<uint32_t>(numTiled);
        devActiveTilesTrianglesCount = CudaPtr<uint32_t>(numTiled);
    }

    void* devTempStorage = nullptr;
    size_t tempStorageBytes = 0;

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::CompactResult");
        cub::DeviceRunLengthEncode::Encode(
            devTempStorage, tempStorageBytes,
            devWorkQueueKeysSorted.get(), devActiveTilesList.get(), 
            devActiveTilesTrianglesCount.get(),
            devActiveTilesNum.get(), workQueueSize
        );
    }

    {
        PROFILING_SCOPE("TiledVox::Memory");
        gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));
    }

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::CompactResult");
        cub::DeviceRunLengthEncode::Encode(
            devTempStorage, tempStorageBytes,
            devWorkQueueKeysSorted.get(), devActiveTilesList.get(), 
            devActiveTilesTrianglesCount.get(), 
            devActiveTilesNum.get(), workQueueSize
        );

        cudaDeviceSynchronize();
        cudaFree(devTempStorage);
        devTempStorage = nullptr;
        tempStorageBytes = 0;
    }

    {
        PROFILING_SCOPE("TiledVox::Memory");
        devActiveTilesNum.CopyToHost(&numActiveTiles, 1);
        devActiveTilesOffset = CudaPtr<uint32_t>(numActiveTiles);
    }

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::CompactResult");
        cub::DeviceScan::ExclusiveSum(
            devTempStorage, tempStorageBytes,
            devActiveTilesTrianglesCount.get(), 
            devActiveTilesOffset.get(), 
            numActiveTiles
        );
    }

    {
        PROFILING_SCOPE("TiledVox::Memory");
        gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));
    }

    {
        PROFILING_SCOPE("TiledVox::TileAssignment::CompactResult");
        cub::DeviceScan::ExclusiveSum(
            devTempStorage, tempStorageBytes,
            devActiveTilesTrianglesCount.get(), 
            devActiveTilesOffset.get(), 
            numActiveTiles
        );

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        cudaFree(devTempStorage);
    }
}

template <typename T>
__global__ void CalculateNumOverlapPerTriangle(const size_t numTriangles, 
                                               const uint32_t* triangleCoords,
                                               const Position* coords, 
                                               const VoxelsGrid<T, true> grid,
                                               uint32_t* overlapPerTriangle)
{ 
    int trianglendex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (trianglendex >= numTriangles)
        return;

    Position V0 = coords[triangleCoords[(trianglendex * 3)]];
    Position V1 = coords[triangleCoords[(trianglendex * 3) + 1]];
    Position V2 = coords[triangleCoords[(trianglendex * 3) + 2]];

    Normal normal = CalculateFaceNormal(V0, V1, V2);
    int sign = 2 * (normal.X >= 0) - 1;

    Position facesVertices[3] = {V0, V1, V2};
    std::pair<float, float> BB_X, BB_Y, BB_Z;
    CalculateBoundingBox(std::span<Position>(&facesVertices[0], 3), BB_X, BB_Y, BB_Z);

    const float tileSize = grid.VoxelSize() * 4;
    int startY = static_cast<int>(std::floor((BB_Y.first - grid.OriginY()) / tileSize));
    int endY   = static_cast<int>(std::ceil((BB_Y.second - grid.OriginY()) / tileSize));
    int startZ = static_cast<int>(std::floor((BB_Z.first - grid.OriginZ()) / tileSize));
    int endZ   = static_cast<int>(std::ceil((BB_Z.second - grid.OriginZ()) / tileSize));

    Normal N0 = CalculateNormalZY(V0, V1) * sign;
    Normal N1 = CalculateNormalZY(V1, V2) * sign;
    Normal N2 = CalculateNormalZY(V2, V0) * sign;

    int numOverlap = 0;
    for(int y = startY; y < endY; ++y) 
    {
        for(int z = startZ; z < endZ; ++z) 
        {
            float minY = grid.OriginY() + (y * tileSize);
            float minZ = grid.OriginZ() + (z * tileSize);
            float maxY = minY + tileSize;
            float maxZ = minZ + tileSize;

            float E0 = CalculateEdgeFunctionZY(V0, V1, N0.Y >= 0 ? minY : maxY, N0.Z >= 0 ? minZ : maxZ) * sign;
            float E1 = CalculateEdgeFunctionZY(V1, V2, N1.Y >= 0 ? minY : maxY, N1.Z >= 0 ? minZ : maxZ) * sign;
            float E2 = CalculateEdgeFunctionZY(V2, V0, N2.Y >= 0 ? minY : maxY, N2.Z >= 0 ? minZ : maxZ) * sign;

            if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
                numOverlap++;
            }
        }
    }
    overlapPerTriangle[trianglendex] = numOverlap;
}


template <typename T>
__global__ void WorkQueuePopulation(const size_t numTriangles, 
                                    const uint32_t* triangleCoords,
                                    const Position* coords, 
                                    const uint32_t* offsets, 
                                    const VoxelsGrid<T, true> grid, 
                                    const size_t workQueueSize,
                                    uint32_t* workQueueKeys, 
                                    uint32_t* workQueueValues)
{
    int triangleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (triangleIndex >= numTriangles)
        return;

    Position V0 = coords[triangleCoords[(triangleIndex * 3)]];
    Position V1 = coords[triangleCoords[(triangleIndex * 3) + 1]];
    Position V2 = coords[triangleCoords[(triangleIndex * 3) + 2]];

    Normal normal = CalculateFaceNormal(V0, V1, V2);
    int sign = 2 * (normal.X >= 0) - 1;

    Position facesVertices[3] = {V0, V1, V2};
    std::pair<float, float> BB_X, BB_Y, BB_Z;
    CalculateBoundingBox(std::span<Position>(&facesVertices[0], 3), BB_X, BB_Y, BB_Z);

    const float tileSize = grid.VoxelSize() * 4;
    const uint tilePerSide = grid.VoxelsPerSide() / 4;

    int startY = static_cast<int>(std::floor((BB_Y.first - grid.OriginY()) / tileSize));
    int endY   = static_cast<int>(std::ceil((BB_Y.second - grid.OriginY()) / tileSize));
    int startZ = static_cast<int>(std::floor((BB_Z.first - grid.OriginZ()) / tileSize));
    int endZ   = static_cast<int>(std::ceil((BB_Z.second - grid.OriginZ()) / tileSize));

    Position N0 = CalculateNormalZY(V0, V1) * sign;
    Position N1 = CalculateNormalZY(V1, V2) * sign;
    Position N2 = CalculateNormalZY(V2, V0) * sign;

    int numOverlap = 0;
    for(int y = startY; y < endY; ++y)
    {
        for(int z = startZ; z < endZ; ++z)
        {
            float minY = grid.OriginY() + (y * tileSize);
            float minZ = grid.OriginZ() + (z * tileSize);
            float maxY = minY + tileSize;
            float maxZ = minZ + tileSize;

            float E0 = CalculateEdgeFunctionZY(V0, V1, N0.Y > 0 ? minY : maxY, N0.Z > 0 ? minZ : maxZ) * sign;
            float E1 = CalculateEdgeFunctionZY(V1, V2, N1.Y > 0 ? minY : maxY, N1.Z > 0 ? minZ : maxZ) * sign;
            float E2 = CalculateEdgeFunctionZY(V2, V0, N2.Y > 0 ? minY : maxY, N2.Z > 0 ? minZ : maxZ) * sign;

            if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
                workQueueKeys[offsets[triangleIndex] + numOverlap] = (y * tilePerSide) + z;
                workQueueValues[offsets[triangleIndex] + numOverlap] = triangleIndex;
                numOverlap++;
            }
        }
    }
}

template <typename T>
__global__ void TiledProcessing(const int BATCH_SIZE,
                                const uint32_t* triangleCoords, 
                                const Position* coords, 
                                const uint32_t* workQueue, 
                                const uint32_t* activeTilesList, 
                                const uint32_t* activeTilesListTriangleCount,
                                const uint32_t* activeTilesListOffset, 
                                VoxelsGrid<T, true> grid)
{
    extern __shared__ Position sharedVertices[];

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
        for(int i = voxelIndex; i < BATCH_SIZE * 3 && i < (numTriangles - batch) * 3; i+=blockDim.x)
        {
            const int posVertex = (i / 3);
            const int posCoord = (i % 3);
            const int indexV = triangleCoords[(workQueue[(tileOffset + posVertex + batch)] * 3) + posCoord];

            sharedVertices[(posVertex * 3) + (posCoord)] = coords[indexV];
        }

        // ================ OLD VERSION ========================
        //if (voxelIndex < BATCH_SIZE && (voxelIndex + batch) < numTriangles) {
            //const int indexV0 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3)];
            //const int indexV1 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 1];
            //const int indexV2 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 2];

            //sharedVertices[(voxelIndex * 3)]     = coords[indexV0];
            //sharedVertices[(voxelIndex * 3) + 1] = coords[indexV1];
            //sharedVertices[(voxelIndex * 3) + 2] = coords[indexV2];
        //}
        // ================ OLD VERSION ========================

        __syncthreads();

        int sharedSize = min(BATCH_SIZE, numTriangles - batch);
        int voxelHalf = voxelIndex / 16;
        int startTriangle = voxelHalf * (sharedSize / (blockDim.x / 16));
        int endTriangle = (sharedSize * (voxelHalf + 1)) / (blockDim.x / 16);


        for(int triangle = startTriangle; triangle < endTriangle; triangle++)
        {         
            Position V0 = sharedVertices[(triangle * 3)];
            Position V1 = sharedVertices[(triangle * 3) + 1];
            Position V2 = sharedVertices[(triangle * 3) + 2];

            Normal normal = CalculateFaceNormal(V0, V1, V2);
            int sign = 2 * (normal.X >= 0) - 1;

            float E0 = CalculateEdgeFunctionZY(V0, V1, centerY, centerZ) * sign;
            float E1 = CalculateEdgeFunctionZY(V1, V2, centerY, centerZ) * sign;
            float E2 = CalculateEdgeFunctionZY(V2, V0, centerY, centerZ) * sign;

            if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
                Position edge0 = V1 - V0;
                Position edge1 = V2 - V0;

                auto [A, B, C] = Position::Cross(edge0, edge1);   

                float D = Position::Dot({A, B, C}, V0);
                float intersection = ((D - (B * centerY) - (C * centerZ)) / A);

                int startX = static_cast<int>((intersection - grid.OriginX()) / grid.VoxelSize());
                int endX = grid.VoxelsPerSide();

                for(int x = (startX / grid.WordSize()) * grid.WordSize(); x < endX; x+=grid.WordSize())
                {
                    T newWord = 0;
                    for(int bit = startX % grid.WordSize(); bit < grid.WordSize(); ++bit) {
                        newWord |= (1 << bit);
                    }
                    // grid.Word(x, y, z) ^= newWord;
                    atomicXor(&grid.Word(x, y, z), newWord);

                    startX = 0;
                }
            }
        }
        __syncthreads();
    }
}


template<Types type, typename T>
void Compute<Types::TILED, T>(const size_t blockSize, HostVoxelsGrid<T>& grid, const Mesh& mesh)
{
    PROFILING_SCOPE("TiledVox(" + mesh.Name + ")");
    const size_t numTriangles = mesh.FacesSize() * 2;

    DeviceVoxelsGrid<T> devGrid;
    {
        PROFILING_SCOPE("TiledVox::Memory");
        devGrid = DeviceVoxelsGrid<T>(grid.View().VoxelsPerSide(), grid.View().VoxelSize());
        devGrid.View().SetOrigin(grid.View().OriginX(), grid.View().OriginY(), grid.View().OriginZ());
    }

    CudaPtr<uint32_t> devTrianglesCoords;
    CudaPtr<Position> devCoords;
    CudaPtr<uint32_t> devOverlapPerTriangle;
    TileAssignmentCalculateOverlap<T>(
        numTriangles, mesh, devTrianglesCoords, 
        devCoords, devOverlapPerTriangle, devGrid.View()
    );


    CudaPtr<uint32_t> devOffsets;
    TileAssignmentExclusiveScan(numTriangles, devOffsets, devOverlapPerTriangle);


    int lastOverlapTriangle, lastOffset;
    { 
        PROFILING_SCOPE("TiledVox::Memory");
        gpuAssert(cudaMemcpy(&lastOverlapTriangle, devOverlapPerTriangle.get() + (numTriangles - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpuAssert(cudaMemcpy(&lastOffset, devOffsets.get() + (numTriangles - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    const size_t workQueueSize = lastOverlapTriangle + lastOffset;

    CudaPtr<uint32_t> devWorkQueueKeys;
    CudaPtr<uint32_t> devWorkQueueValues;

    TileAssignmentWorkQueuePopulation<T>(
        numTriangles, workQueueSize, devTrianglesCoords, 
        devCoords, devOffsets, devGrid.View(), 
        devWorkQueueKeys, devWorkQueueValues
    );


    CudaPtr<uint32_t> devWorkQueueKeysSorted;
    CudaPtr<uint32_t> devWorkQueueValuesSorted;

    TileAssignmentWorkQueueSorting(
        workQueueSize, devWorkQueueKeys, devWorkQueueValues, 
        devWorkQueueKeysSorted, devWorkQueueValuesSorted
    );    


    const size_t numTiled = (grid.View().VoxelsPerSide() * grid.View().VoxelsPerSide()) / 4;
    uint32_t numActiveTiles = 0;
    CudaPtr<uint32_t> devActiveTilesList;
    CudaPtr<uint32_t> devActiveTilesTrianglesCount;
    CudaPtr<uint32_t> devActiveTilesOffset;

    TileAssignmentCompactResult(
        workQueueSize, numTiled, numActiveTiles, 
        devWorkQueueKeysSorted, devActiveTilesList, 
        devActiveTilesTrianglesCount, devActiveTilesOffset
    );

    {
        PROFILING_SCOPE("TiledVox::Processing");

        //cudaDeviceProp prop;
        //cudaGetDeviceProperties(&prop, 0);
        //const size_t maxSMEM = prop.sharedMemPerBlock / (sizeof(Position) * 3);
        const int defaultSize = 14;

        TiledProcessing<T><<< numActiveTiles, blockSize, defaultSize * 3 * sizeof(Position) >>>(
            defaultSize,
            devTrianglesCoords.get(), 
            devCoords.get(), 
            devWorkQueueValuesSorted.get(), 
            devActiveTilesList.get(), 
            devActiveTilesTrianglesCount.get(), 
            devActiveTilesOffset.get(), 
            devGrid.View()
        );  

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }

    {
        PROFILING_SCOPE("TiledVox::Memory");
        grid = HostVoxelsGrid<T>(devGrid);
    }
}


template void TileAssignmentCalculateOverlap<uint32_t>
(const size_t, const Mesh&, CudaPtr<uint32_t>&, CudaPtr<Position>&, 
 CudaPtr<uint32_t>&, VoxelsGrid<uint32_t, true>&);
template void TileAssignmentCalculateOverlap<uint64_t>
(const size_t, const Mesh&, CudaPtr<uint32_t>&, CudaPtr<Position>&, 
 CudaPtr<uint32_t>&, VoxelsGrid<uint64_t, true>&);


template void TileAssignmentWorkQueuePopulation<uint32_t>
(const size_t, const size_t, CudaPtr<uint32_t>&, CudaPtr<Position>&, 
 CudaPtr<uint32_t>&, VoxelsGrid<uint32_t, true>&, CudaPtr<uint32_t>&, CudaPtr<uint32_t>&);
template void TileAssignmentWorkQueuePopulation<uint64_t>
(const size_t, const size_t, CudaPtr<uint32_t>&, CudaPtr<Position>&, 
 CudaPtr<uint32_t>&, VoxelsGrid<uint64_t, true>&, CudaPtr<uint32_t>&, CudaPtr<uint32_t>&);


template __global__ void CalculateNumOverlapPerTriangle<uint32_t>
(const size_t, const uint32_t*, const Position*, 
 const VoxelsGrid<uint32_t, true>, uint32_t*);
template __global__ void CalculateNumOverlapPerTriangle<uint64_t>
(const size_t, const uint32_t*, const Position*, 
 const VoxelsGrid<uint64_t, true>, uint32_t*);


template __global__ void WorkQueuePopulation<uint32_t>
(const size_t, const uint32_t*, const Position*, const uint32_t*, 
 const VoxelsGrid<uint32_t, true>, const size_t, uint32_t*, uint32_t*);
template __global__ void WorkQueuePopulation<uint64_t>
(const size_t, const uint32_t*, const Position*, const uint32_t*, 
 const VoxelsGrid<uint64_t, true>, const size_t, uint32_t*, uint32_t*);


template __global__ void TiledProcessing<uint32_t>
(const int, const uint32_t*, const Position*, const uint32_t*, const uint32_t*, 
 const uint32_t*, const uint32_t*, VoxelsGrid<uint32_t, true>);
template __global__ void TiledProcessing<uint64_t>
(const int, const uint32_t*, const Position*, const uint32_t*, const uint32_t*, 
 const uint32_t*, const uint32_t*, VoxelsGrid<uint64_t, true>);


template void Compute<Types::TILED, uint32_t>
(const size_t, HostVoxelsGrid<uint32_t>&, const Mesh&); 
template void Compute<Types::TILED, uint64_t>
(const size_t, HostVoxelsGrid<uint64_t>&, const Mesh&); 

}


// ========================== TEST TiledProcessing ==========================
//template <typename T, int BATCH_SIZE>
//__global__ void TiledProcessing(const uint32_t numActiveTiles, 
                                //const uint32_t* triangleCoords, 
                                //const Position* coords, 
                                //const uint32_t* workQueue, 
                                //const uint32_t* activeTilesList, 
                                //const uint32_t* activeTilesListTriangleCount,
                                //const uint32_t* activeTilesListOffset, 
                                //VoxelsGrid<T, true> grid)
//{
    //__shared__ Position smem[BATCH_SIZE * 3 * 2];

    //Position* sharedVertices = &smem[0];
    //int activeTileIndex = blockIdx.x;
    //int voxelIndex = threadIdx.x;

    //if (voxelIndex >= 16) {
        //sharedVertices = &smem[BATCH_SIZE * 3];
        //activeTileIndex = gridDim.x + blockIdx.x;    
        //voxelIndex = threadIdx.x % 16;
    //}

    //if(activeTileIndex >= numActiveTiles)
        //return;

    //const int numTriangles = activeTilesListTriangleCount[activeTileIndex];     
    //const int tileOffset = activeTilesListOffset[activeTileIndex];
    //const int tileIndex = activeTilesList[activeTileIndex];

    //int tileZ = tileIndex % (grid.VoxelsPerSide() / 4);
    //int tileY = tileIndex / (grid.VoxelsPerSide() / 4);

    //int voxelZ = (voxelIndex % 16) % 4;
    //int voxelY = (voxelIndex % 16) / 4;

    //int z = (tileZ * 4) + voxelZ;
    //int y = (tileY * 4) + voxelY;

    //float centerZ = grid.OriginZ() + (z * grid.VoxelSize()) + (grid.VoxelSize() / 2);
    //float centerY = grid.OriginY() + (y * grid.VoxelSize()) + (grid.VoxelSize() / 2);


    //for(int batch = 0; batch < numTriangles; batch += BATCH_SIZE)
    //{
        //for(int i = voxelIndex; i < BATCH_SIZE * 3 && i < (numTriangles - batch) * 3; i+=16)
        //{
            //const int posVertex = (i / 3);
            //const int posCoord = (i % 3);
            //const int indexV = triangleCoords[(workQueue[(tileOffset + posVertex + batch)] * 3) + posCoord];

            //sharedVertices[(posVertex * 3) + (posCoord)] = coords[indexV];
        //}

        //// ================ OLD VERSION ========================
        ////if (voxelIndex < BATCH_SIZE && (voxelIndex + batch) < numTriangles) {
            ////const int indexV0 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3)];
            ////const int indexV1 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 1];
            ////const int indexV2 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 2];

            ////sharedVertices[(voxelIndex * 3)]     = coords[indexV0];
            ////sharedVertices[(voxelIndex * 3) + 1] = coords[indexV1];
            ////sharedVertices[(voxelIndex * 3) + 2] = coords[indexV2];
        ////}
        //// ================ OLD VERSION ========================

        //__syncthreads();

        //int sharedSize = min(BATCH_SIZE, numTriangles - batch);

        //for(int triangle = 0; triangle < sharedSize; triangle++)
        //{         
            //Position V0 = sharedVertices[(triangle * 3)];
            //Position V1 = sharedVertices[(triangle * 3) + 1];
            //Position V2 = sharedVertices[(triangle * 3) + 2];

            //Normal normal = CalculateFaceNormal(V0, V1, V2);
            //int sign = 2 * (normal.X >= 0) - 1;

            //float E0 = CalculateEdgeFunctionZY(V0, V1, centerY, centerZ) * sign;
            //float E1 = CalculateEdgeFunctionZY(V1, V2, centerY, centerZ) * sign;
            //float E2 = CalculateEdgeFunctionZY(V2, V0, centerY, centerZ) * sign;

            //if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
                //Position edge0 = V1 - V0;
                //Position edge1 = V2 - V0;

                //auto [A, B, C] = Position::Cross(edge0, edge1);   

                //float D = Position::Dot({A, B, C}, V0);
                //float intersection = ((D - (B * centerY) - (C * centerZ)) / A);

                //int startX = static_cast<int>((intersection - grid.OriginX()) / grid.VoxelSize());
                //int endX = grid.VoxelsPerSide();

                //for(int x = (startX / grid.WordSize()) * grid.WordSize(); x < endX; x+=grid.WordSize())
                //{       
                    //T newWord = 0;
                    //for(int bit = startX % grid.WordSize(); bit < grid.WordSize(); ++bit) {
                        //newWord |= (1 << bit);
                    //}
                    ////grid.Word(x, y, z) ^= newWord;
                    //atomicXor(&grid.Word(x, y, z), newWord);
                    //startX = 0;
                //}
            //}
        //} 
        //__syncthreads();
    //}
//}
// ========================== TEST TiledProcessing ==========================

