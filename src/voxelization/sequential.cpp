#include <voxelization/voxelization.h>
#include <bounding_box.h>

namespace Voxelization {

template <typename T>
__host__ void Sequential(const std::vector<uint32_t>& triangleCoords,
                         const std::vector<Position>& coords,
                         VoxelsGrid<T, false>& grid)
{
    const int numTriangle = triangleCoords.size() / 3;
    for(int i = 0; i < numTriangle; ++i) {
        Position V0 = coords[triangleCoords[(i * 3)]];
        Position V1 = coords[triangleCoords[(i * 3) + 1]];
        Position V2 = coords[triangleCoords[(i * 3) + 2]];

        Normal normal = CalculateFaceNormal(V0, V1, V2);
        int sign = 2 * (normal.X >= 0) - 1;

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

                float E0 = CalculateEdgeFunctionZY(V0, V1, centerY, centerZ) * sign;
                float E1 = CalculateEdgeFunctionZY(V1, V2, centerY, centerZ) * sign;
                float E2 = CalculateEdgeFunctionZY(V2, V0, centerY, centerZ) * sign;

                if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
                    float intersection = (D - (B * centerY) - (C * centerZ)) / A;

                    int startX = static_cast<int>((intersection - grid.OriginX()) / grid.VoxelSize());
                    int endX = grid.VoxelsPerSide();
                    for(int x = startX; x < endX; ++x)
                        grid(x, y, z) ^= true;
                }
            }
        }
    }
}

template __host__ void Sequential<uint8_t>
(const std::vector<uint32_t>&, const std::vector<Position>&, VoxelsGrid<uint8_t, false>&);

template __host__ void Sequential<uint16_t>
(const std::vector<uint32_t>&, const std::vector<Position>&, VoxelsGrid<uint16_t, false>&);

template __host__ void Sequential<uint32_t>
(const std::vector<uint32_t>&, const std::vector<Position>&, VoxelsGrid<uint32_t, false>&);

template __host__ void Sequential<uint64_t>
(const std::vector<uint32_t>&, const std::vector<Position>&, VoxelsGrid<uint64_t, false>&);

}
