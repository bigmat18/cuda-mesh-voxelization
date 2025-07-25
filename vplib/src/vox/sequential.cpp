#include <vox/vox.h>
#include <bounding_box.h>

namespace VOX {

template<Types type, typename T>
void Compute<Types::SEQUENTIAL, T>(HostVoxelsGrid<T>& grid, const Mesh& mesh) 
{
    PROFILING_SCOPE("SequentialVox(" + mesh.Name + ")");

    {
        PROFILING_SCOPE("SequentialVox::Processing");
        auto& triangleCoords = mesh.FacesCoords;
        auto& coords = mesh.Coords;
        auto& v = grid.View();
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

            int startY = static_cast<int>(std::floor((BB_Y.first - v.OriginY()) / v.VoxelSize()));
            int endY   = static_cast<int>(std::ceil((BB_Y.second - v.OriginY()) / v.VoxelSize()));
            int startZ = static_cast<int>(std::floor((BB_Z.first - v.OriginZ()) / v.VoxelSize()));
            int endZ   = static_cast<int>(std::ceil((BB_Z.second - v.OriginZ()) / v.VoxelSize()));

            Position edge0 = V1 - V0;
            Position edge1 = V2 - V0;
            auto [A, B, C] = Position::Cross(edge0, edge1);
            float D = Position::Dot({A, B, C}, V0);

            for(int y = startY; y < endY; ++y)
            {
                for(int z = startZ; z < endZ; ++z)
                {
                    float centerY = v.OriginY() + ((y * v.VoxelSize()) + (v.VoxelSize() / 2));
                    float centerZ = v.OriginZ() + ((z * v.VoxelSize()) + (v.VoxelSize() / 2));

                    float E0 = CalculateEdgeFunctionZY(V0, V1, centerY, centerZ) * sign;
                    float E1 = CalculateEdgeFunctionZY(V1, V2, centerY, centerZ) * sign;
                    float E2 = CalculateEdgeFunctionZY(V2, V0, centerY, centerZ) * sign;

                    if (E0 >= 0 && E1 >= 0 && E2 >= 0) {
                        float intersection = (D - (B * centerY) - (C * centerZ)) / A;

                        int startX = static_cast<int>((intersection - v.OriginX()) / v.VoxelSize());
                        int endX = v.VoxelsPerSide();
                        for(int x = startX; x < endX; ++x)
                            v.Voxel(x, y, z) ^= true;
                    }
                }
            }
        }
    }
}

template void Compute<Types::SEQUENTIAL, uint32_t>
(HostVoxelsGrid<uint32_t>&, const Mesh&);    

template void Compute<Types::SEQUENTIAL, uint64_t>
(HostVoxelsGrid<uint64_t>&, const Mesh&); 

}
