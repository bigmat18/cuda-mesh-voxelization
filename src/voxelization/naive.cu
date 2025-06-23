#include <span>
#include <voxelization/voxelization.h>
#include <bounding_box.h>

template <typename T>
__global__ void NaiveKernel(size_t trianglesSize, uint32_t* triangleCoords, 
                            Position* coords, VoxelsGrid<T, true> grid)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= trianglesSize)
        return;

    Position V0 = coords[triangleCoords[(index * 3)]];
    Position V1 = coords[triangleCoords[(index * 3) + 1]];
    Position V2 = coords[triangleCoords[(index * 3) + 2]];
        
    Position facesVertices[3] = {V0, V1, V2};
    std::pair<float, float> BB_X, BB_Y, BB_Z;
    CalculateBoundingBox(std::span<Position>(&facesVertices[0], 3), BB_X, BB_Y, BB_Z);

    int startY = static_cast<int>(std::floorf((BB_Y.first - grid.OriginY()) / grid.VoxelSize()));
    int endY   = static_cast<int>(std::ceilf((BB_Y.second - grid.OriginY()) / grid.VoxelSize()));
    int startZ = static_cast<int>(std::floorf((BB_Z.first - grid.OriginZ()) / grid.VoxelSize()));
    int endZ   = static_cast<int>(std::ceilf((BB_Z.second - grid.OriginZ()) / grid.VoxelSize()));

    Position edge0 = V1 - V0;
    Position edge1 = V2 - V0;
    auto [A, B, C] = Position::Cross(edge0, edge1);
    float D = Position::Dot({A, B, C}, V0);

    for(int y = startY; y < endY; ++y)
    {
        for(int z = startZ; z < endZ; ++z)
        {
            float centerY = grid.OriginY() + ((y * grid.VoxelSize()) + (grid.VoxelSize() / 2));
            float centerZ = grid.OriginZ() + ((z * grid.VoxelSize()) + (grid.VoxelSize() / 2));

            float E0 = CalculateEdgeFunction(V0, V1, centerY, centerZ);
            float E1 = CalculateEdgeFunction(V1, V2, centerY, centerZ);
            float E2 = CalculateEdgeFunction(V2, V0, centerY, centerZ);

            bool ccw_test = (E0 >= 0 && E1 >= 0 && E2 >= 0);
            bool cw_test = (E0 <= 0 && E1 <= 0 && E2 <= 0);
            
            if (ccw_test || cw_test) {
                float intersection = (D - (B * centerY) - (C * centerZ)) / A;

                int startX = static_cast<int>((intersection - grid.OriginX()) / grid.VoxelSize());
                int endX = grid.VoxelsPerSide();
                for(int x = startX; x < endX; ++x)
                    grid(x, y, z) ^= true;
            }
        }
    }
}


template __global__ void NaiveKernel<uint32_t>
 (size_t, uint32_t*, Position*, VoxelsGrid<uint32_t, true>);

