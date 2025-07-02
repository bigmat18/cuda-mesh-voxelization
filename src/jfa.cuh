#include "mesh/mesh.h"
#include <cmath>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <voxels_grid.h>
#include <profiling.h>

#ifndef JFA_H
#define JFA_H

namespace JFA {

struct SDF {
    int x = -1;
    int y = -1;
    int z = -1;
    float distance = std::numeric_limits<float>::infinity();
};

enum class Types {
    SEQUENTIAL, NAIVE, TILED
};

__host__ __device__ inline float CalculateDistance(Position p0, Position p1) 
{ return std::sqrt((p1.X - p0.X) + (p1.Y - p0.Y) + (p1.Z - p0.Z)); }

template <typename T>
__global__ void JFAInizializationNaive(const VoxelsGrid<T, true> grid, SDF* SDFValues) 
{
    const int voxelIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(voxelIndex >= grid.SpaceSize())
        return;

    const int voxelZ = voxelIndex / (grid.VoxelsPerSide() * grid.VoxelsPerSide());
    const int voxelY = (voxelIndex % (grid.VoxelsPerSide() * grid.VoxelsPerSide())) / grid.VoxelsPerSide();
    const int voxelX = voxelIndex % grid.VoxelsPerSide();

    if(!grid(voxelX, voxelY, voxelZ))
        return;

    for(int z = -1; z <= 1; z++) {
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                if(x == 0 && y == 0 && z == 0)
                    continue;

                int nx = voxelX + x;
                int ny = voxelY + y;
                int nz = voxelZ + z;

                if(nx < 0 || nx >= grid.VoxelsPerSide() ||
                   ny < 0 || ny >= grid.VoxelsPerSide() ||
                   nz < 0 || nz >= grid.VoxelsPerSide())
                    continue;

                if(!grid(nx, ny, nz)) {
                    SDFValues[voxelIndex] = SDF({voxelX, voxelY, voxelZ, 0});
                    return;
                }
            }
        }
    }
}

template <typename T>
__global__ void JPAProcessingNaive(const int K, const VoxelsGrid<T, true> grid, const SDF* valuesIn, SDF* valuesOut) {

    const int voxelIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(voxelIndex >= grid.SpaceSize())
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

                int seedIndex = (nz * grid.VoxelsPerSide() * grid.VoxelsPerSide()) + (ny * grid.VoxelsPerSide());
                SDF seed = valuesIn[seedIndex];
                SDF voxel = valuesIn[voxelIndex];

                if(seed.distance < std::numeric_limits<float>::infinity()) {
                    float distance = CalculateDistance(Position(voxelX, voxelY, voxelZ), Position(seed.x, seed.y, seed.z));
                    if(distance < voxel.distance) {
                        valuesOut[voxelIndex] = SDF({seed.x, seed.y, seed.z, distance});
                    }
                }
            }
        }
    }
}

template <Types type, typename T>
void Compute(VoxelsGrid<T, true>& grid, std::vector<SDF>& sdfValues)
requires (type == Types::NAIVE)
{ 
    PROFILING_SCOPE("Naive JFA");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const size_t numVoxels = grid.SpaceSize();
    const size_t blockSize = NextPow2(numVoxels, prop.maxThreadsDim[0] / 2);
    const size_t gridSize = (numVoxels + blockSize - 1) / blockSize;


    SDF* devSDFValues;
    gpuAssert(cudaMalloc((void**) &devSDFValues, sdfValues.size() * sizeof(SDF)));
    gpuAssert(cudaMemcpy(devSDFValues, &sdfValues[0], sdfValues.size() * sizeof(SDF), cudaMemcpyHostToDevice));

    JFAInizializationNaive<T><<< gridSize, blockSize >>>(grid, devSDFValues);

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    SDF* devSDFValuesApp;
    gpuAssert(cudaMalloc((void**) &devSDFValuesApp, sdfValues.size() * sizeof(SDF)));
    gpuAssert(cudaMemcpy(devSDFValuesApp, devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToDevice));

    for(int k = grid.VoxelsPerSide() / 2; k >= 1; k /= 2) {
        JPAProcessingNaive<T><<< gridSize, blockSize >>>(k, grid, devSDFValues, devSDFValuesApp);
        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
        gpuAssert(cudaMemcpy(devSDFValues, devSDFValuesApp, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToDevice));
    }

    gpuAssert(cudaMemcpy(sdfValues.data(), devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToHost));
};

};

#endif // !JFA_H
