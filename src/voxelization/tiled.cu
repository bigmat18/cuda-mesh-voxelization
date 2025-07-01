#include <cmath>
#include <cstdint>
#include <voxelization/voxelization.cuh>
#include <bounding_box.h>

namespace Voxelization {

template <typename T>
void TileAssignmentCalculateOverlap(const size_t numTriangles, 
                                           const Mesh& mesh,
                                           uint32_t** devTrianglesCoords, 
                                           Position** devCoords,
                                           uint32_t** devOverlapPerTriangle,
                                           VoxelsGrid<T, true>& grid) 
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    const size_t blockSize = NextPow2(numTriangles, prop.maxThreadsDim[0] / 2);
    const size_t gridSize = (numTriangles + blockSize - 1) / blockSize;

    gpuAssert(cudaMalloc((void**) devTrianglesCoords, mesh.FacesCoords.size() * sizeof(uint32_t)));
    gpuAssert(cudaMemcpy(*devTrianglesCoords, &mesh.FacesCoords[0], mesh.FacesCoords.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));    

    gpuAssert(cudaMalloc((void**) devCoords, mesh.Coords.size() * sizeof(Position)));
    gpuAssert(cudaMemcpy(*devCoords, &mesh.Coords[0], mesh.Coords.size() * sizeof(Position), cudaMemcpyHostToDevice));
 
    gpuAssert(cudaMalloc((void**) devOverlapPerTriangle, numTriangles * sizeof(uint32_t)));
    gpuAssert(cudaMemset(*devOverlapPerTriangle, 0, numTriangles * sizeof(uint32_t)));
    
    CalculateNumOverlapPerTriangle<T><<< gridSize, blockSize >>>(
        numTriangles, 
        *devTrianglesCoords, 
        *devCoords, 
        grid,
        *devOverlapPerTriangle
    );

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

void TileAssignmentExclusiveScan(const size_t numTriangles,
                                        uint32_t** devOffsets,
                                        uint32_t** devOverlapPerTriangle) 
{
    gpuAssert(cudaMalloc((void**) devOffsets, numTriangles * sizeof(uint32_t)));

    void* devTempStorage = nullptr;
    size_t tempStorageBytes = 0;

    cub::DeviceScan::ExclusiveSum(
        devTempStorage, tempStorageBytes,
        *devOverlapPerTriangle, *devOffsets, numTriangles
    );

    gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));

    cub::DeviceScan::ExclusiveSum(
        devTempStorage, tempStorageBytes,
        *devOverlapPerTriangle, *devOffsets, numTriangles
    );


    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    cudaFree(devTempStorage);
}

template <typename T>
void TileAssignmentWorkQueuePopulation(const size_t numTriangles,
                                             const size_t workQueueSize,
                                             uint32_t** devTrianglesCoords,
                                             Position** devCoords,
                                             uint32_t** devOffsets,
                                             VoxelsGrid<T, true>& grid,
                                             uint32_t** devWorkQueueKeys,
                                             uint32_t** devWorkQueueValues) 
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    const size_t blockSize = NextPow2(numTriangles, prop.maxThreadsDim[0] / 2);
    const size_t gridSize = (numTriangles + blockSize - 1) / blockSize;

    gpuAssert(cudaMalloc((void**) devWorkQueueKeys, workQueueSize * sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**) devWorkQueueValues, workQueueSize * sizeof(uint32_t)));
        
    WorkQueuePopulation<T><<< gridSize, blockSize >>>(
        numTriangles, *devTrianglesCoords, *devCoords, 
        *devOffsets, grid, workQueueSize,
        *devWorkQueueKeys, *devWorkQueueValues
    );

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

void TileAssignmentWorkQueueSorting(const size_t workQueueSize,
                                           uint32_t** devWorkQueueKeys,
                                           uint32_t** devWorkQueueValues,
                                           uint32_t** devWorkQueueKeysSorted,
                                           uint32_t** devWorkQueueValuesSorted) 
{

    void* devTempStorage = nullptr;
    size_t tempStorageBytes = 0;

    gpuAssert(cudaMalloc((void**) devWorkQueueKeysSorted, workQueueSize * sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**) devWorkQueueValuesSorted, workQueueSize * sizeof(uint32_t)));

    cub::DeviceRadixSort::SortPairs(
        devTempStorage, tempStorageBytes,
        *devWorkQueueKeys, *devWorkQueueKeysSorted,
        *devWorkQueueValues, *devWorkQueueValuesSorted, workQueueSize
    );

    gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));
        
    cub::DeviceRadixSort::SortPairs(
        devTempStorage, tempStorageBytes,
        *devWorkQueueKeys, *devWorkQueueKeysSorted,
        *devWorkQueueValues, *devWorkQueueValuesSorted, workQueueSize
    );
    
    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    cudaFree(devTempStorage);
}

void TileAssignmentCompactResult(const size_t workQueueSize,
                                        const size_t numTiled,
                                        uint32_t& numActiveTiles,
                                        uint32_t** devWorkQueueKeysSorted,
                                        uint32_t** devActiveTilesList,
                                        uint32_t** devActiveTilesTrianglesCount,
                                        uint32_t** devActiveTilesOffset) 
{
    uint32_t* devActiveTilesNum;
    void* devTempStorage = nullptr;
    size_t tempStorageBytes = 0;

    gpuAssert(cudaMalloc((void**) devActiveTilesList, numTiled * sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**) devActiveTilesTrianglesCount, numTiled * sizeof(uint32_t)));
    gpuAssert(cudaMalloc((void**) &devActiveTilesNum, sizeof(uint32_t)));

    cub::DeviceRunLengthEncode::Encode(
        devTempStorage, tempStorageBytes,
        *devWorkQueueKeysSorted, *devActiveTilesList, 
        *devActiveTilesTrianglesCount,
        devActiveTilesNum, workQueueSize
    );

    gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));

    cub::DeviceRunLengthEncode::Encode(
        devTempStorage, tempStorageBytes,
        *devWorkQueueKeysSorted, 
        *devActiveTilesList, 
        *devActiveTilesTrianglesCount,
        devActiveTilesNum, 
        workQueueSize
    );

    cudaDeviceSynchronize();
    cudaFree(devTempStorage);
    devTempStorage = nullptr;
    tempStorageBytes = 0;

    gpuAssert(cudaMemcpy(&numActiveTiles, devActiveTilesNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(devActiveTilesNum);

    gpuAssert(cudaMalloc((void**) devActiveTilesOffset, numActiveTiles * sizeof(uint32_t)));
    cub::DeviceScan::ExclusiveSum(
        devTempStorage, tempStorageBytes,
        *devActiveTilesTrianglesCount, 
        *devActiveTilesOffset, 
        numActiveTiles
    );

    gpuAssert(cudaMalloc(&devTempStorage, tempStorageBytes));

    cub::DeviceScan::ExclusiveSum(
        devTempStorage, tempStorageBytes,
        *devActiveTilesTrianglesCount, 
        *devActiveTilesOffset, 
        numActiveTiles
    );

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    cudaFree(devTempStorage);
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

template <typename T, int BLOCK_SIZE>
__global__ void TiledProcessing(const uint32_t* triangleCoords, 
                                const Position* coords, 
                                const uint32_t* workQueue, 
                                const uint32_t* activeTilesList, 
                                const uint32_t* activeTilesListTriangleCount,
                                const uint32_t* activeTilesListOffset, 
                                VoxelsGrid<T, true> grid)
{
    __shared__ Position sharedVertices[BLOCK_SIZE * 3];

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


    for(int batch = 0; batch < numTriangles; batch += BLOCK_SIZE)
    {
        for(int i = voxelIndex; i < BLOCK_SIZE * 3 && i < (numTriangles - batch) * 3; i+=32)
        {
            const int posVertex = (i / 3);
            const int posCoord = (i % 3);
            const int indexV = triangleCoords[(workQueue[(tileOffset + posVertex + batch)] * 3) + posCoord];

            sharedVertices[(posVertex * 3) + (posCoord)] = coords[indexV];
        }

        // ================ OLD VERSION ========================
        //if (voxelIndex < BLOCK_SIZE && (voxelIndex + batch) < numTriangles) {
            //const int indexV0 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3)];
            //const int indexV1 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 1];
            //const int indexV2 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 2];

            //sharedVertices[(voxelIndex * 3)]     = coords[indexV0];
            //sharedVertices[(voxelIndex * 3) + 1] = coords[indexV1];
            //sharedVertices[(voxelIndex * 3) + 2] = coords[indexV2];
        //}
        // ================ OLD VERSION ========================

        __syncthreads();

        int sharedSize = min(BLOCK_SIZE, numTriangles - batch);
        int voxelHalf = voxelIndex / 16;
        int startTriangle = voxelHalf * (sharedSize / 2);
        int endTriangle = (sharedSize * (voxelHalf + 1)) / 2;


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
                    grid.XorWord(x, y, z, newWord);
                    startX = 0;
                }
            }
        }
        __syncthreads();
    }
}


template void TileAssignmentCalculateOverlap<uint32_t>
(const size_t, const Mesh&, uint32_t**, Position**, 
 uint32_t**, VoxelsGrid<uint32_t, true>&);


template void TileAssignmentWorkQueuePopulation<uint32_t>
(const size_t, const size_t, uint32_t**, Position**, 
 uint32_t**, VoxelsGrid<uint32_t, true>&, uint32_t**, uint32_t**);


template __global__ void CalculateNumOverlapPerTriangle<uint32_t>
(const size_t, const uint32_t*, const Position*, 
 const VoxelsGrid<uint32_t, true>, uint32_t*);


template __global__ void WorkQueuePopulation<uint32_t>
(const size_t, const uint32_t*, const Position*, const uint32_t*, 
 const VoxelsGrid<uint32_t, true>, const size_t, uint32_t*, uint32_t*);


template __global__ void TiledProcessing<uint32_t>
(const uint32_t*, const Position*, const uint32_t*, const uint32_t*, 
 const uint32_t*, const uint32_t*, VoxelsGrid<uint32_t, true>);

}
