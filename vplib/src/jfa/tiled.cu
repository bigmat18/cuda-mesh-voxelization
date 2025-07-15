#include "grid/grid.h"
#include "grid/voxels_grid.h"
#include "mesh/mesh.h"
#include "proc_utils.h"
#include <jfa/jfa.h>
#include <string>

namespace JFA {

template <typename T, int TILE_DIM>
__global__ void InizializationTiled(const VoxelsGrid<T, true> grid, Grid<float> sdf, Grid<Position> positions)
{
    static_assert(TILE_DIM % 2 != 0, "TILE_DIM must be odd");

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

    for(int depth = 0; depth < TILE_DIM; ++depth) {

        if(blockIndex < SMEM_DIM) {
            int smemZ = (blockIndex % OUT_TILE_DIM);
            int smemY = (blockIndex / OUT_TILE_DIM);

            int dz = -(OUT_TILE_DIM / 2) + smemZ;
            int dy = -(OUT_TILE_DIM / 2) + smemY;

            int x = centerTileX + ((depth + 1) * grid.WordSize()); 
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
            Position pos;

            for(int z = -1; z <= 1; z++) {
                for(int y = -1; y <= 1; y++) {
                    for(int x = -1; x <= 1; x++) {
                        if(x == 0 && y == 0 && z == 0)
                            continue;

                        int nx = tileX + x;
                        int ny = tileY + y;
                        int nz = tileZ + z;

                        if(!gridSMEM.Voxel(nx, ny, nz)) {
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

        __syncthreads();

        if(blockIndex < SMEM_DIM) { 
            int smemZ = (blockIndex % OUT_TILE_DIM);
            int smemY = (blockIndex / OUT_TILE_DIM);

            gridSMEM.Word(0, smemY, smemZ) = gridSMEM.Word(grid.WordSize(), smemY, smemZ);
            gridSMEM.Word(grid.WordSize(), smemY, smemZ) = gridSMEM.Word(grid.WordSize() * 2, smemY, smemZ);
        }
    }
}


template <typename T>
__global__ void ProcessingTiled(const int K, const VoxelsGrid<T, true> grid,
                                const Grid<float> inSDF, const Grid<Position> inPositions,
                                Grid<float> outSDF, Grid<Position> outPositions) 
{

}

template <Types type, typename T>
void Compute<Types::TILED, T>(DeviceVoxelsGrid<T>& grid, DeviceGrid<float>& sdf, DeviceGrid<Position>& positions)
{   
    assert(grid.View().VoxelsPerSide() % grid.View().WordSize() == 0);
    PROFILING_SCOPE("TiledJFA");
    
    {
        PROFILING_SCOPE("TiledJFA::Inizialization");

        const size_t TILE_DIM = 1;
        dim3 blockSize(grid.View().WordSize(), TILE_DIM, TILE_DIM);
        dim3 gridSize(
            (grid.View().VoxelsPerSide() + grid.View().WordSize() * TILE_DIM - 1) / (grid.View().WordSize() * TILE_DIM),
            (grid.View().VoxelsPerSide() + TILE_DIM - 1) / TILE_DIM,
            (grid.View().VoxelsPerSide() + TILE_DIM - 1) / TILE_DIM
        );                                          

        InizializationTiled<T, TILE_DIM><<< gridSize, blockSize >>>(grid.View(), sdf.View(), positions.View());

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }

    {
        PROFILING_SCOPE("TiledJFA::Processing");

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        const size_t numVoxels = grid.View().Size();
        const size_t blockSize = NextPow2(numVoxels, prop.maxThreadsDim[0] / 2);
        const size_t gridSize = (numVoxels + blockSize - 1) / blockSize;
        
        DeviceGrid<float> sdfApp(sdf);
        DeviceGrid<Position> positionsApp(positions);

        for(int k = grid.View().VoxelsPerSide() / 2; k >= 1; k /= 2) { 
            PROFILING_SCOPE(std::to_string(k));
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
}


template void Compute<Types::TILED, uint32_t>
(DeviceVoxelsGrid<uint32_t>&, DeviceGrid<float>&, DeviceGrid<Position>&);

template void Compute<Types::TILED, uint64_t>
(DeviceVoxelsGrid<uint64_t>&, DeviceGrid<float>&, DeviceGrid<Position>&);

template __global__ void InizializationTiled<uint32_t>
(const VoxelsGrid<uint32_t, true>, Grid<float>, Grid<Position>);

template __global__ void InizializationTiled<uint64_t>
(const VoxelsGrid<uint64_t, true>, Grid<float>, Grid<Position>);

template __global__ void ProcessingTiled<uint32_t>
(const int, const VoxelsGrid<uint32_t, true>, const Grid<float>, const Grid<Position>, Grid<float>, Grid<Position>);

template __global__ void ProcessingTiled<uint64_t>
(const int, const VoxelsGrid<uint64_t, true>, const Grid<float>, const Grid<Position>, Grid<float>, Grid<Position>);

}
