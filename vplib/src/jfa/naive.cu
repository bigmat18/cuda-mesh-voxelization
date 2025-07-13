#include "grid/voxels_grid.h"
#include "mesh/mesh.h"
#include <cmath>
#include <jfa/jfa.h>
#include <limits>

namespace JFA {

template <typename T>
__global__ void InizializationNaive(const VoxelsGrid<T, true> grid, Grid<float> sdf, Grid<Position> positions) 
{
    const int voxelIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(voxelIndex >= grid.Size())
        return;

    const int voxelZ = voxelIndex / (grid.VoxelsPerSide() * grid.VoxelsPerSide());
    const int voxelY = (voxelIndex % (grid.VoxelsPerSide() * grid.VoxelsPerSide())) / grid.VoxelsPerSide();
    const int voxelX = voxelIndex % grid.VoxelsPerSide();

    if(!grid.Voxel(voxelX, voxelY, voxelZ))
        return;


    bool found = false;
    Position pos;

    for(int z = -1; z <= 1; z++) {
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                if(x == 0 && y == 0 && z == 0)
                    continue;

                int nx = voxelX + x;
                int ny = voxelY + y;
                int nz = voxelZ + z;

                bool isBorder = nx < 0 || nx >= grid.VoxelsPerSide() || 
                                ny < 0 || ny >= grid.VoxelsPerSide() || 
                                nz < 0 || nz >= grid.VoxelsPerSide();

                if(isBorder || !grid.Voxel(nx, ny, nz)) {
                    found = true;
                    pos = Position(grid.OriginX() + (voxelX * grid.VoxelSize()),
                                   grid.OriginY() + (voxelY * grid.VoxelSize()),
                                   grid.OriginZ() + (voxelZ * grid.VoxelSize()));
                }
            }
        }
    }
    if(found) {
        sdf(voxelX, voxelY, voxelZ) = 0.0f;
        positions(voxelX, voxelY, voxelZ) = pos;
    } else {
        sdf(voxelX, voxelY, voxelZ) = std::numeric_limits<float>::infinity();
    }
}

template <typename T>
__global__ void ProcessingNaive(const int K, const VoxelsGrid<T, true> grid, 
                                const Grid<float> inSDF, const Grid<Position> inPositions,
                                Grid<float> outSDF, Grid<Position> outPositions) {

    const int voxelIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(voxelIndex >= grid.Size())
        return;

    const int voxelZ = voxelIndex / (grid.VoxelsPerSide() * grid.VoxelsPerSide());
    const int voxelY = (voxelIndex % (grid.VoxelsPerSide() * grid.VoxelsPerSide())) / grid.VoxelsPerSide();
    const int voxelX = voxelIndex % grid.VoxelsPerSide();

    for(int z = -1; z <= 1; z++) {
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                if(x == 0 && y == 0 && z == 0)
                    continue;

                int nx = voxelX + (x * K);
                int ny = voxelY + (y * K);
                int nz = voxelZ + (z * K);

                if(nx < 0 || nx >= grid.VoxelsPerSide() ||
                   ny < 0 || ny >= grid.VoxelsPerSide() ||
                   nz < 0 || nz >= grid.VoxelsPerSide())
                    continue;

                float seed = inSDF(nx, ny, nz);
                float voxel = inSDF(voxelX, voxelY, voxelZ);

                if(std::abs(seed) < std::numeric_limits<float>::infinity()) {
                    Position seedPos = inPositions(nx, ny, nz);
                    Position voxelPos = Position(grid.OriginX() + (voxelX * grid.VoxelSize()),
                                                 grid.OriginY() + (voxelY * grid.VoxelSize()),
                                                 grid.OriginZ() + (voxelZ * grid.VoxelSize()));

                    float distance = CalculateDistance(voxelPos, seedPos);
                    if(distance < std::abs(voxel)) {
                        outSDF(voxelX, voxelY, voxelZ) = std::copysign(distance, voxel);
                        outPositions(voxelX, voxelY, voxelZ) = seedPos;
                    }
                }
            }
        }
    }
}

template <Types type, typename T>
void Compute<Types::NAIVE, T>(DeviceVoxelsGrid<T>& grid, DeviceGrid<float>& sdf, DeviceGrid<Position>& positions)
{ 
    PROFILING_SCOPE("NaiveJFA");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const size_t numVoxels = grid.View().Size();
    const size_t blockSize = NextPow2(numVoxels, prop.maxThreadsDim[0] / 2);
    const size_t gridSize = (numVoxels + blockSize - 1) / blockSize;

    {
        PROFILING_SCOPE("NaiveJFA::Inizialization");
        InizializationNaive<T><<< gridSize, blockSize >>>(grid.View(), sdf.View(), positions.View());
        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }
    
    {
        PROFILING_SCOPE("NaiveJFA::Processing");
        DeviceGrid<float> sdfApp(sdf);
        DeviceGrid<Position> positionsApp(positions);

        for(int k = grid.View().VoxelsPerSide() / 2; k >= 1; k /= 2) { 
            ProcessingNaive<T><<< gridSize, blockSize >>>(
                k, grid.View(), 
                sdf.View(), positions.View(), 
                sdfApp.View(), positionsApp.View()
            );

            gpuAssert(cudaPeekAtLastError()); 
            cudaDeviceSynchronize();

            sdf = sdfApp;
            positions = positionsApp;
        }
    }
};


template void Compute<Types::NAIVE, uint32_t>
(DeviceVoxelsGrid<uint32_t>&, DeviceGrid<float>&, DeviceGrid<Position>&);

template void Compute<Types::NAIVE, uint64_t>
(DeviceVoxelsGrid<uint64_t>&, DeviceGrid<float>&, DeviceGrid<Position>&);


template __global__ void InizializationNaive<uint32_t>
(const VoxelsGrid<uint32_t, true>, Grid<float>, Grid<Position>);

template __global__ void InizializationNaive<uint64_t>
(const VoxelsGrid<uint64_t, true>, Grid<float>, Grid<Position>);


template __global__ void ProcessingNaive<uint32_t>
(const int, const VoxelsGrid<uint32_t, true>, const Grid<float>, const Grid<Position>, Grid<float>, Grid<Position>);

template __global__ void ProcessingNaive<uint64_t>
(const int, const VoxelsGrid<uint64_t, true>, const Grid<float>, const Grid<Position>, Grid<float>, Grid<Position>);

}
