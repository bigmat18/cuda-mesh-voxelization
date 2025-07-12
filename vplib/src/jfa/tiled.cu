#include "grid/voxels_grid.h"
#include "proc_utils.h"
#include <jfa/jfa.h>

namespace JFA {

template <typename T, int TILE_DIM>
__global__ void InizializationTiled(const VoxelsGrid<T, true> grid, SDF* SDFValues)
{
    constexpr int OUT_TILE_DIM = TILE_DIM + 2;
    constexpr int SMEM_DIM = OUT_TILE_DIM * OUT_TILE_DIM;

    __shared__ T SMEM[OUT_TILE_DIM * 3];

    VoxelsGrid<T, true> gridSMEM(&SMEM[0], grid.WordSize() * 3, OUT_TILE_DIM, OUT_TILE_DIM);

    const int voxelX = blockIdx.x * (grid.WordSize() * TILE_DIM) + threadIdx.x;
    const int voxelY = blockIdx.y * TILE_DIM + threadIdx.y;
    const int voxelZ = blockIdx.z * TILE_DIM + threadIdx.z;
    const int voxelIndex = grid.Index(voxelX, voxelY, voxelZ);

    if(voxelIndex >= grid.Size())
        return;
    
    const int blockIndex = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    const int centerTileX = (blockIdx.x * blockDim.x * TILE_DIM);
    const int centerTileY = (blockIdx.y * blockDim.y) + (TILE_DIM / 2);
    const int centerTileZ = (blockIdx.z * blockDim.z) + (TILE_DIM / 2);

    if(blockIndex < SMEM_DIM) {
        int smemZ = (blockIndex % OUT_TILE_DIM);
        int smemY = (blockIndex / OUT_TILE_DIM);

        int dz = -(OUT_TILE_DIM / 2) + smemZ;
        int dy = -(OUT_TILE_DIM / 2) + smemY;

        int x = centerTileX - grid.WordSize(); 
        int y = centerTileY + dy;    
        int z = centerTileZ + dz;

        if(z >= 0 && z < grid.VoxelsPerSide() && y >= 0 && y < grid.VoxelsPerSide() && x >= 0 && x < grid.VoxelsPerSide()) {
            gridSMEM.Word(0, smemY, smemZ) = grid.Word(x, y, z);
        } else {
            gridSMEM.Word(0, smemY, smemZ) = 0;
        }

        x = centerTileX;
        if(z >= 0 && z < grid.VoxelsPerSide() && y >= 0 && y < grid.VoxelsPerSide() && x >= 0 && x < grid.VoxelsPerSide()) {
            gridSMEM.Word(grid.WordSize(), smemY, smemZ) = grid.Word(x, y, z);
        } else {
            gridSMEM.Word(grid.WordSize(), smemY, smemZ) = 0;
        }
    }

    for(int depth = 0; depth < 1; ++depth) {

        if(blockIndex < SMEM_DIM) {
            int smemZ = (blockIndex % OUT_TILE_DIM);
            int smemY = (blockIndex / OUT_TILE_DIM);

            int dz = -(OUT_TILE_DIM / 2) + smemZ;
            int dy = -(OUT_TILE_DIM / 2) + smemY;

            int x = centerTileX - grid.WordSize(); 
            int y = centerTileY + dy;    
            int z = centerTileZ + dz;

            if(z >= 0 && z < grid.VoxelsPerSide() && y >= 0 && y < grid.VoxelsPerSide() && x >= 0 && x < grid.VoxelsPerSide()) {
                gridSMEM.Word(grid.WordSize() * 2, smemY, smemZ) = grid.Word(x, y, z);
            } else {
                gridSMEM.Word(grid.WordSize() * 2, smemY, smemZ) = 0;
            }
        }

        __syncthreads();
        
        int tileX = grid.WordSize() + threadIdx.x;
        int tileY = (OUT_TILE_DIM / 2) + threadIdx.y;
        int tileZ = (OUT_TILE_DIM / 2) + threadIdx.z;    

        if(gridSMEM.Voxel(tileX, tileY, tileZ)) {
            bool found = false;
            for(int z = -1; z <= 1 && !found; z++) {
                for(int y = -1; y <= 1 && !found; y++) {
                    for(int x = -1; x <= 1 && !found; x++) {
                        if(x == 0 && y == 0 && z == 0)
                            continue;

                        int nx = tileX + x;
                        int ny = tileY + y;
                        int nz = tileZ + z;

                        if(!gridSMEM.Voxel(nx, ny, nz)) {
                            SDFValues[voxelIndex] = SDF({voxelX, voxelY, voxelZ, 0});
                            found = true;
                        }
                    }
                }
            }
        }

        __syncthreads();

        if(blockIndex < SMEM_DIM) { 
            int smemZ = (blockIndex % OUT_TILE_DIM);
            int smemY = (blockIndex / OUT_TILE_DIM);

            gridSMEM.Word(0, smemY, smemZ) = gridSMEM.Word(grid.WordSize(), smemY, smemZ);
            gridSMEM.Word(grid.WordSize(), smemY, smemZ) = gridSMEM.Word(grid.WordSize() * 2, smemY, smemZ);
        }
    }
}


template <Types type, typename T>
void Compute<Types::TILED, T>(DeviceVoxelsGrid<T>& grid, std::vector<SDF>& sdfValues)
{   
    assert(grid.View().VoxelsPerSide() % grid.View().WordSize() == 0);
    PROFILING_SCOPE("TiledJFA");
    
    SDF* devSDFValues;
    {
        PROFILING_SCOPE("TiledJFA::Inizialization");
        gpuAssert(cudaMalloc((void**) &devSDFValues, sdfValues.size() * sizeof(SDF)));
        gpuAssert(cudaMemcpy(devSDFValues, &sdfValues[0], sdfValues.size() * sizeof(SDF), cudaMemcpyHostToDevice));

        const size_t TILE_DIM = 1;
        dim3 blockSize(grid.View().WordSize(), TILE_DIM, TILE_DIM);
        dim3 gridSize(
            (grid.View().VoxelsPerSide() + grid.View().WordSize() * TILE_DIM - 1) / (grid.View().WordSize() * TILE_DIM),
            (grid.View().VoxelsPerSide() + TILE_DIM - 1) / TILE_DIM,
            (grid.View().VoxelsPerSide() + TILE_DIM - 1) / TILE_DIM
        );                                          

        InizializationTiled<T, TILE_DIM><<< gridSize, blockSize >>>(grid.View(), devSDFValues);

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }

    {
        PROFILING_SCOPE("TiledJFA::Processing");
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        const size_t numVoxels = grid.View().Size();
        const size_t blockSizeProc = NextPow2(numVoxels, prop.maxThreadsDim[0] / 2);
        const size_t gridSizeProc = (numVoxels + blockSizeProc - 1) / blockSizeProc;

        SDF* devSDFValuesApp;
        gpuAssert(cudaMalloc((void**) &devSDFValuesApp, sdfValues.size() * sizeof(SDF)));
        gpuAssert(cudaMemcpy(devSDFValuesApp, devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToDevice));

        for(int k = grid.View().VoxelsPerSide() / 2; k >= 1; k /= 2) { 
            ProcessingNaive<T><<< gridSizeProc, blockSizeProc >>>(k, grid.View(), devSDFValues, devSDFValuesApp); 
            gpuAssert(cudaPeekAtLastError()); 
            cudaDeviceSynchronize();
            gpuAssert(cudaMemcpy(devSDFValues, devSDFValuesApp, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToDevice));
        }
    }

    gpuAssert(cudaMemcpy(sdfValues.data(), devSDFValues, sdfValues.size() * sizeof(SDF), cudaMemcpyDeviceToHost));
}


template void Compute<Types::TILED, uint32_t>
(DeviceVoxelsGrid<uint32_t>&, std::vector<SDF>&);

template void Compute<Types::TILED, uint64_t>
(DeviceVoxelsGrid<uint64_t>&, std::vector<SDF>&);

template __global__ void InizializationTiled<uint32_t>
(const VoxelsGrid<uint32_t, true>, SDF*);

template __global__ void InizializationTiled<uint64_t>
(const VoxelsGrid<uint64_t, true>, SDF*);

}
