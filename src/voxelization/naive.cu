#include <voxelization/voxelization.cuh>
#include <bounding_box.h>

namespace Voxelization {

template <typename T>
__global__ void NaiveKernel(const size_t numTriangles, 
                            const uint32_t* triangleCoords, 
                            const Position* coords, 
                            VoxelsGrid<T, true> grid)
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

    int startY = static_cast<int>(std::floor((BB_Y.first - grid.OriginY()) / grid.VoxelSize()));
    int endY   = static_cast<int>(std::ceil((BB_Y.second - grid.OriginY()) / grid.VoxelSize()));
    int startZ = static_cast<int>(std::floor((BB_Z.first - grid.OriginZ()) / grid.VoxelSize()));
    int endZ   = static_cast<int>(std::ceil((BB_Z.second - grid.OriginZ()) / grid.VoxelSize()));

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

            float E0 = CalculateEdgeFunctionZY(V0, V1, centerY, centerZ) * sign;
            float E1 = CalculateEdgeFunctionZY(V1, V2, centerY, centerZ) * sign;
            float E2 = CalculateEdgeFunctionZY(V2, V0, centerY, centerZ) * sign;

            if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
                float intersection = (D - (B * centerY) - (C * centerZ)) / A;

                int startX = static_cast<int>((intersection - grid.OriginX()) / grid.VoxelSize());
                int endX = grid.VoxelsPerSide();

                for(int x = (startX / grid.WordSize()) * grid.WordSize(); x < endX; x+=grid.WordSize())
                {
                    T newWord = 0;
                    for(int bit = startX % grid.WordSize(); bit < grid.WordSize(); ++bit) {
                        newWord |= (1 << bit);
                    }
                    atomicXor(&grid.Word(x, y, z), newWord);
                    startX = 0;
                }
            }
        }
    }
}


template __global__ void NaiveKernel<uint32_t>
(const size_t, const uint32_t*, const Position*, VoxelsGrid<uint32_t, true>);

}

