#include "mesh/mesh.h"
#include "mesh/mesh_io.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>
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
{ return std::sqrt(std::pow(p1.X - p0.X, 2) + std::pow(p1.Y - p0.Y, 2) + std::pow(p1.Z - p0.Z, 2)); }


template <typename T, int TILE_DIM = 1>
__global__ void JFAInizializationTiled(const VoxelsGrid<T, true> grid, SDF* SDFValues)
{
    constexpr int OUT_TILE_DIM = TILE_DIM + 2;
    constexpr int SMEM_DIM = OUT_TILE_DIM * OUT_TILE_DIM;

    __shared__ T SMEM[SMEM_DIM * 3];

    VoxelsGrid<T, true> gridSMEM(&SMEM[0], grid.WordSize() * 3, OUT_TILE_DIM, OUT_TILE_DIM);

    const int voxelX = blockIdx.x * (grid.WordSize() * TILE_DIM) + threadIdx.x;
    const int voxelY = blockIdx.y * TILE_DIM + threadIdx.y;
    const int voxelZ = blockIdx.z * TILE_DIM + threadIdx.z;
    const int voxelIndex = grid.Index(voxelX, voxelY, voxelZ);

    if(voxelIndex >= grid.SpaceSize())
        return;
    
    const int blockIndex = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; 

    if(blockIndex < SMEM_DIM) {
        int smemZ = (blockIndex % OUT_TILE_DIM);
        int smemY = (blockIndex / OUT_TILE_DIM);

        int dz = -(OUT_TILE_DIM / 2) + smemZ;
        int dy = -(OUT_TILE_DIM / 2) + smemY;

        int x = voxelX - grid.WordSize(); 
        int y = voxelY + dy;    
        int z = voxelZ + dz;

        if(z >= 0 && z < grid.VoxelsPerSide() && y >= 0 && y < grid.VoxelsPerSide() && x >= 0 && x < grid.VoxelsPerSide()) {
            gridSMEM.SetWord(0, smemY, smemZ, grid.GetWord(x, y, z));
        } else {
            gridSMEM.SetWord(0, smemY, smemZ, 0);
        }

        x = voxelX;
        if(z >= 0 && z < grid.VoxelsPerSide() && y >= 0 && y < grid.VoxelsPerSide() && x >= 0 && x < grid.VoxelsPerSide()) {
            gridSMEM.SetWord(grid.WordSize(), smemY, smemZ, grid.GetWord(x, y, z));
        } else {
            gridSMEM.SetWord(grid.WordSize(), smemY, smemZ, 0);
        }
    }

    for(int depth = 0; depth < 1; ++depth) {

        if(blockIndex < SMEM_DIM) {
            int smemZ = (blockIndex % OUT_TILE_DIM);
            int smemY = (blockIndex / OUT_TILE_DIM);

            int dz = -(OUT_TILE_DIM / 2) + smemZ;
            int dy = -(OUT_TILE_DIM / 2) + smemY;


            int x = voxelX + ((depth + 1) * grid.WordSize());
            int y = voxelY + dy;    
            int z = voxelZ + dz;

            if(z >= 0 && z < grid.VoxelsPerSide() && y >= 0 && y < grid.VoxelsPerSide() && x >= 0 && x < grid.VoxelsPerSide()) {
                gridSMEM.SetWord(grid.WordSize() * 2, smemY, smemZ, grid.GetWord(x, y, z));
            } else {
                gridSMEM.SetWord(grid.WordSize() * 2, smemY, smemZ, 0);
            }
        }

        __syncthreads();
        
        if(blockIndex == 0) {
            gridSMEM.Print();
        }

        //int tileX = grid.WordSize() + threadIdx.x;
        //int tileY = (OUT_TILE_DIM / 2);
        //int tileZ = (OUT_TILE_DIM / 2);


        //if(gridSMEM(tileX, tileY, tileZ)) {
            //bool found = false;
            //for(int z = -1; z <= 1 && !found; z++) {
                //for(int y = -1; y <= 1 && !found; y++) {
                    //for(int x = -1; x <= 1 && !found; x++) {
                        //if(x == 0 && y == 0 && z == 0)
                            //continue;

                        //int nx = tileX + x;
                        //int ny = tileY + y;
                        //int nz = tileZ + z;

                        //printf("ciao");
                        //if(!gridSMEM(nx, ny, nz)) {
                            //SDFValues[voxelIndex] = SDF({voxelX, voxelY, voxelZ, 0});
                            //found = true;
                        //}
                    //}
                //}
            //}
        //}

        //__syncthreads();

        //if(blockIndex < SMEM_DIM) { 
            //int smemZ = (blockIndex % OUT_TILE_DIM);
            //int smemY = (blockIndex / OUT_TILE_DIM);

            //gridSMEM(smemX, smemY, 0) = gridSMEM(smemX, smemY, 1);
            //gridSMEM(smemX, smemY, 1) = gridSMEM(smemX, smemY, 2);
        //}
    }
}   

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

                bool isBorder = nx < 0 || nx >= grid.VoxelsPerSide() || 
                                ny < 0 || ny >= grid.VoxelsPerSide() || 
                                nz < 0 || nz >= grid.VoxelsPerSide();

                if(isBorder || !grid(nx, ny, nz)) {
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
void Compute(DeviceVoxelsGrid<T>& grid, std::vector<SDF>& sdfValues)
requires (type == Types::NAIVE)
{ 
    PROFILING_SCOPE("Naive JFA");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const size_t numVoxels = grid.View().SpaceSize();
    const size_t blockSize = NextPow2(numVoxels, prop.maxThreadsDim[0] / 2);
    const size_t gridSize = (numVoxels + blockSize - 1) / blockSize;


    SDF* devSDFValues;
    gpuAssert(cudaMalloc((void**) &devSDFValues, sdfValues.size() * sizeof(SDF)));
    gpuAssert(cudaMemcpy(devSDFValues, &sdfValues[0], sdfValues.size() * sizeof(SDF), cudaMemcpyHostToDevice));

    JFAInizializationNaive<T><<< gridSize, blockSize >>>(grid.View(), devSDFValues);

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    SDF* devSDFValuesApp;
    gpuAssert(cudaMalloc((void**) &devSDFValuesApp, sdfValues.size() * sizeof(SDF)));
    gpuAssert(cudaMemcpy(devSDFValuesApp, devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToDevice));

    for(int k = grid.View().VoxelsPerSide() / 2; k >= 1; k /= 2) { 
        JPAProcessingNaive<T><<< gridSize, blockSize >>>(k, grid.View(), devSDFValues, devSDFValuesApp); 
        gpuAssert(cudaPeekAtLastError()); 
        cudaDeviceSynchronize();
        gpuAssert(cudaMemcpy(devSDFValues, devSDFValuesApp, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToDevice));
    }

    gpuAssert(cudaMemcpy(sdfValues.data(), devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToHost));
};


template <Types type, typename T>
void Compute(DeviceVoxelsGrid<T>& grid, std::vector<SDF>& sdfValues)
requires (type == Types::TILED)
{   
    assert(grid.View().VoxelsPerSide() % grid.View().WordSize() == 0);
    PROFILING_SCOPE("TILED JFA");

    SDF* devSDFValues;
    gpuAssert(cudaMalloc((void**) &devSDFValues, sdfValues.size() * sizeof(SDF)));
    gpuAssert(cudaMemcpy(devSDFValues, &sdfValues[0], sdfValues.size() * sizeof(SDF), cudaMemcpyHostToDevice));

    const size_t TILE_DIM = 3;
    dim3 blockSize(grid.View().WordSize(), TILE_DIM, TILE_DIM);
    dim3 gridSize(
        (grid.View().VoxelsPerSide() + grid.View().WordSize() * TILE_DIM - 1) / (grid.View().WordSize() * TILE_DIM),
        (grid.View().VoxelsPerSide() + TILE_DIM - 1) / TILE_DIM,
        (grid.View().VoxelsPerSide() + TILE_DIM - 1) / TILE_DIM
    );
                                                                            
    
    JFAInizializationTiled<T, TILE_DIM><<< 1, blockSize >>>(grid.View(), devSDFValues);

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    gpuAssert(cudaMemcpy(sdfValues.data(), devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToHost));
};

};

#endif // !JFA_H
