#include "grid/voxels_grid.h"
#include "proc_utils.h"
#include <cstdint>
#include <jfa/jfa.h>

namespace JFA {

template <typename T>
__global__ void InizializationNaive(const VoxelsGrid<T, true> grid, SDF* SDFValues) 
{
    const int voxelIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(voxelIndex >= grid.Size())
        return;

    const int voxelZ = voxelIndex / (grid.VoxelsPerSide() * grid.VoxelsPerSide());
    const int voxelY = (voxelIndex % (grid.VoxelsPerSide() * grid.VoxelsPerSide())) / grid.VoxelsPerSide();
    const int voxelX = voxelIndex % grid.VoxelsPerSide();

    if(!grid.Voxel(voxelX, voxelY, voxelZ))
        return;

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
                    SDFValues[voxelIndex] = SDF({voxelX, voxelY, voxelZ, 0});
                    return;
                }
            }
        }
    }
}

template <typename T>
__global__ void ProcessingNaive(const int K, const VoxelsGrid<T, true> grid, const SDF* valuesIn, SDF* valuesOut) {

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

                int seedIndex = (nz * grid.VoxelsPerSide() * grid.VoxelsPerSide()) + (ny * grid.VoxelsPerSide()) + nx;
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
void Compute<Types::NAIVE, T>(DeviceVoxelsGrid<T>& grid, std::vector<SDF>& sdfValues)
{ 
    PROFILING_SCOPE("NaiveJFA");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const size_t numVoxels = grid.View().Size();
    const size_t blockSize = NextPow2(numVoxels, prop.maxThreadsDim[0] / 2);
    const size_t gridSize = (numVoxels + blockSize - 1) / blockSize;

    SDF* devSDFValues;

    {
        PROFILING_SCOPE("NaiveJFA::Inizialization");
        gpuAssert(cudaMalloc((void**) &devSDFValues, sdfValues.size() * sizeof(SDF)));
        gpuAssert(cudaMemcpy(devSDFValues, &sdfValues[0], sdfValues.size() * sizeof(SDF), cudaMemcpyHostToDevice));

        InizializationNaive<T><<< gridSize, blockSize >>>(grid.View(), devSDFValues);

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }
    
    {
        PROFILING_SCOPE("NaiveJFA::Processing");
        SDF* devSDFValuesApp;
        gpuAssert(cudaMalloc((void**) &devSDFValuesApp, sdfValues.size() * sizeof(SDF)));
        gpuAssert(cudaMemcpy(devSDFValuesApp, devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToDevice));

        for(int k = grid.View().VoxelsPerSide() / 2; k >= 1; k /= 2) { 
            ProcessingNaive<T><<< gridSize, blockSize >>>(k, grid.View(), devSDFValues, devSDFValuesApp); 
            gpuAssert(cudaPeekAtLastError()); 
            cudaDeviceSynchronize();
            gpuAssert(cudaMemcpy(devSDFValues, devSDFValuesApp, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToDevice));
        }

    }

    gpuAssert(cudaMemcpy(sdfValues.data(), devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToHost));
};


template void Compute<Types::NAIVE, uint32_t>
(DeviceVoxelsGrid<uint32_t>&, std::vector<SDF>&);

template void Compute<Types::NAIVE, uint64_t>
(DeviceVoxelsGrid<uint64_t>&, std::vector<SDF>&);


template __global__ void InizializationNaive<uint32_t>
(const VoxelsGrid<uint32_t, true>, SDF*);

template __global__ void InizializationNaive<uint64_t>
(const VoxelsGrid<uint64_t, true>, SDF*);


template __global__ void ProcessingNaive<uint32_t>
(const int, const VoxelsGrid<uint32_t, true>, const SDF*, SDF*);

template __global__ void ProcessingNaive<uint64_t>
(const int, const VoxelsGrid<uint64_t, true>, const SDF*, SDF*);

}
