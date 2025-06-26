#include <voxelization/voxelization.h>
#include <bounding_box.h>

namespace Voxelization {

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
    int startY = static_cast<int>(std::floorf((BB_Y.first - grid.OriginY()) / tileSize));
    int endY   = static_cast<int>(std::ceilf((BB_Y.second - grid.OriginY()) / tileSize));
    int startZ = static_cast<int>(std::floorf((BB_Z.first - grid.OriginZ()) / tileSize));
    int endZ   = static_cast<int>(std::ceilf((BB_Z.second - grid.OriginZ()) / tileSize));

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

    int startY = static_cast<int>(std::floorf((BB_Y.first - grid.OriginY()) / tileSize));
    int endY   = static_cast<int>(std::ceilf((BB_Y.second - grid.OriginY()) / tileSize));
    int startZ = static_cast<int>(std::floorf((BB_Z.first - grid.OriginZ()) / tileSize));
    int endZ   = static_cast<int>(std::ceilf((BB_Z.second - grid.OriginZ()) / tileSize));

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
        if (voxelIndex < BLOCK_SIZE && (voxelIndex + batch) < numTriangles) {
            const int indexV0 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3)];
            const int indexV1 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 1];
            const int indexV2 = triangleCoords[(workQueue[(tileOffset + voxelIndex + batch)] * 3) + 2];

            sharedVertices[(voxelIndex * 3)]     = coords[indexV0];
            sharedVertices[(voxelIndex * 3) + 1] = coords[indexV1];
            sharedVertices[(voxelIndex * 3) + 2] = coords[indexV2];
        }

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

                for(int x = startX; x < endX; ++x)
                    grid(x, y, z) ^= true;
            }
        }
        __syncthreads();
    }
}


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
