#include "cuda_ptr.h"
#include "profiling.h"
#include <cstdint>
#include <iostream>
#include <vox/vox.h>
#include <bounding_box.h>
#include <omp.h>

namespace VOX {

template<Types type, typename T>
void Compute<Types::OPENMP, T>(HostVoxelsGrid<T>& grid, const Mesh& mesh) 
{
    PROFILING_SCOPE("OpenMPVox(" + mesh.Name + ")");

    auto& v = grid.View();
    auto& triangleCoords = mesh.FacesCoords;
    auto& coords = mesh.Coords;
    const int numTriangle = triangleCoords.size() / 3;

    int maxThreads = omp_get_max_threads();
    std::vector<HostVoxelsGrid<T>> appGrids;
    {
        PROFILING_SCOPE("OpenMPVox::Memory");
        for (int i = 0; i < maxThreads; ++i) {
            appGrids.push_back(HostVoxelsGrid<T>(v.VoxelsPerSide(), v.VoxelSize()));
        }
    }

    {
        PROFILING_SCOPE("OpenMPVox::Processing");
        #pragma omp parallel for schedule(dynamic)
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
                        for(int x = startX; x < endX; ++x) {
                            appGrids[omp_get_thread_num()].View().Word(x, y, z) ^= true;
                        }
                    }
                }
            }
        }
        using uint = long unsigned int;

        #pragma omp parallel for schedule(dynamic)
        for(uint i = 0; i < v.CalculateStorageSize(v.VoxelsPerSide()); ++i) {
            uint idx = i * v.WordSize();
            uint x = (idx % v.SizeX());
            uint y = (idx / v.SizeX()) % v.SizeY();
            uint z = idx / (v.SizeX() * v.SizeY());

            for(int t = 0; t < maxThreads; ++t) {
                grid.View().Word(x, y, z) ^= appGrids[t].View().Word(x, y, z);
            }
        }
    }
}

template void Compute<Types::OPENMP, uint32_t>
(HostVoxelsGrid<uint32_t>&, const Mesh&);    

template void Compute<Types::OPENMP, uint64_t>
(HostVoxelsGrid<uint64_t>&, const Mesh&); 

}
