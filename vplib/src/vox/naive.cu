#include "grid/voxels_grid.h"
#include "mesh/mesh.h"
#include "profiling.h"
#include <cstdint>
#include <vox/vox.h>
#include <bounding_box.h>
#include <proc_utils.h>
#include <cuda_ptr.h>

namespace VOX {

template <typename T>
__global__ void NaiveProcessing(const size_t numTriangles, 
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

template <Types type, typename T>
void Compute<Types::NAIVE, T>(HostVoxelsGrid<T>& grid, const Mesh& mesh) 
{
    PROFILING_SCOPE("NaiveVox(" + mesh.Name + ")");
    
    DeviceVoxelsGrid<T> devGrid;
    CudaPtr<uint32_t> devTrianglesCoords;
    CudaPtr<Position> devCoords;
    {
        PROFILING_SCOPE("NaiveVox::Memory");
        devGrid = DeviceVoxelsGrid<T>(grid.View().VoxelsPerSide(), grid.View().VoxelSize());
        devGrid.View().SetOrigin(grid.View().OriginX(), grid.View().OriginY(), grid.View().OriginZ());

        devTrianglesCoords = CudaPtr<uint32_t>(&mesh.FacesCoords[0], mesh.FacesCoords.size()); 
        devCoords = CudaPtr<Position>(&mesh.Coords[0], mesh.Coords.size());
    }

    {
        PROFILING_SCOPE("NaiveVox::Processing");
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        const size_t numTriangles = mesh.FacesSize() * 2;
        const size_t blockSize = NextPow2(numTriangles, prop.maxThreadsDim[0] / 2);
        const size_t gridSize = (numTriangles + blockSize - 1) / blockSize;

        NaiveProcessing<T><<< gridSize, blockSize >>>(numTriangles, devTrianglesCoords.get(), devCoords.get(), devGrid.View());

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize(); 
    }

    {
        PROFILING_SCOPE("NaiveVox::Memory");
        grid = HostVoxelsGrid<T>(devGrid);
    }
}


template __global__ void NaiveProcessing<uint32_t>
(const size_t, const uint32_t*, const Position*, VoxelsGrid<uint32_t, true>);

template void Compute<Types::NAIVE, uint32_t>
(HostVoxelsGrid<uint32_t>&, const Mesh&); 

}
