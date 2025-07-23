#include "grid/grid.h"
#include "grid/voxels_grid.h"
#include "mesh/mesh.h"
#include <cmath>
#include <iostream>
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
    Position pos = Position(grid.OriginX() + (voxelX * grid.VoxelSize()),
                            grid.OriginY() + (voxelY * grid.VoxelSize()),
                            grid.OriginZ() + (voxelZ * grid.VoxelSize()));

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

                if(isBorder || !grid.Voxel(nx, ny, nz))
                    found = true;
            }
        }
    }
    if(found) {
        sdf(voxelX, voxelY, voxelZ) = 0.0f;
        positions(voxelX, voxelY, voxelZ) = pos;
    } else {
        sdf(voxelX, voxelY, voxelZ) = INFINITY;
    }
}

template <typename T>
__global__ void ProcessingNaive(const VoxelsGrid<T, true> grid, 
                                Grid<float> sdf, Grid<Position> positions) {

    const int voxelIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(voxelIndex >= grid.Size())
        return;

    const int voxelZ = voxelIndex / (grid.VoxelsPerSide() * grid.VoxelsPerSide());
    const int voxelY = (voxelIndex % (grid.VoxelsPerSide() * grid.VoxelsPerSide())) / grid.VoxelsPerSide();
    const int voxelX = voxelIndex % grid.VoxelsPerSide();

    Position voxelPos = Position(grid.OriginX() + (voxelX * grid.VoxelSize()),
                                 grid.OriginY() + (voxelY * grid.VoxelSize()),
                                 grid.OriginZ() + (voxelZ * grid.VoxelSize()));

    for(int k = grid.VoxelsPerSide() / 2; k >= 1; k/=2)
    {
        bool findNewBest = false;
        float bestDistance = sdf(voxelX, voxelY, voxelZ);
        Position bestPosition;
        for(int z = -1; z <= 1; z++) {
            for(int y = -1; y <= 1; y++) {
                for(int x = -1; x <= 1; x++) {
                    if(x == 0 && y == 0 && z == 0)
                        continue;

                    int nx = voxelX + (x * k);
                    int ny = voxelY + (y * k);
                    int nz = voxelZ + (z * k);

                    if(nx < 0 || nx >= grid.VoxelsPerSide() ||
                       ny < 0 || ny >= grid.VoxelsPerSide() ||
                       nz < 0 || nz >= grid.VoxelsPerSide())
                        continue;

                    float seed = sdf(nx, ny, nz);
                    if(fabs(seed) < INFINITY) {
                        Position seedPos = positions(nx, ny, nz);

                        float distance = CalculateDistance(voxelPos, seedPos);
                        if(distance < fabs(bestDistance)) {
                            findNewBest = true;
                            bestDistance = copysignf(distance, bestDistance);
                            bestPosition = seedPos;
                        }
                    }
                }
            }
        }

        __syncthreads();
        if (findNewBest) {
            sdf(voxelX, voxelY, voxelZ) = bestDistance;
            positions(voxelX, voxelY, voxelZ) = bestPosition;
        }
        __syncthreads();
    }
}

template <Types type, typename T>
void Compute<Types::NAIVE, T>(HostVoxelsGrid<T>& grid, HostGrid<float>& sdf)
{ 
    PROFILING_SCOPE("NaiveJFA");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const size_t numVoxels = grid.View().Size();
    const size_t blockSize = NextPow2(numVoxels, prop.maxThreadsDim[0] / 2);
    const size_t gridSize = (numVoxels + blockSize - 1) / blockSize;

    DeviceVoxelsGrid<T> devGrid;
    DeviceGrid<float> devSDF;
    DeviceGrid<Position> devPositions;
    {
        PROFILING_SCOPE("NaiveJFA::Memory");
        devGrid = DeviceVoxelsGrid<T>(grid);
        devSDF = DeviceGrid<float>(sdf);
        devPositions = DeviceGrid<Position>(grid.View().VoxelsPerSide());
    }

    {
        PROFILING_SCOPE("NaiveJFA::Inizialization");
        InizializationNaive<T><<< gridSize, blockSize >>>(
            devGrid.View(), devSDF.View(), devPositions.View()
        );
        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }
    
    {
        PROFILING_SCOPE("NaiveJFA::Processing");

        ProcessingNaive<T><<< gridSize, blockSize >>>(
            devGrid.View(), devSDF.View(), devPositions.View()
        );
        gpuAssert(cudaPeekAtLastError()); 
        cudaDeviceSynchronize();
    }

    {
        PROFILING_SCOPE("NaiveJFA::Memory");
        sdf = HostGrid<float>(devSDF);
    }
};


template void Compute<Types::NAIVE, uint32_t>
(HostVoxelsGrid<uint32_t>&, HostGrid<float>&);

template void Compute<Types::NAIVE, uint64_t>
(HostVoxelsGrid<uint64_t>&, HostGrid<float>&);


template __global__ void InizializationNaive<uint32_t>
(const VoxelsGrid<uint32_t, true>, Grid<float>, Grid<Position>);

template __global__ void InizializationNaive<uint64_t>
(const VoxelsGrid<uint64_t, true>, Grid<float>, Grid<Position>);


template __global__ void ProcessingNaive<uint32_t>
(const VoxelsGrid<uint32_t, true>, Grid<float>, Grid<Position>);

template __global__ void ProcessingNaive<uint64_t>
(const VoxelsGrid<uint64_t, true>, Grid<float>, Grid<Position>);

}
