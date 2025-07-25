#include "cuda_ptr.h"
#include "grid/grid.h"
#include "grid/voxels_grid.h"
#include "mesh/mesh.h"
#include "proc_utils.h"
#include <cmath>
#include <cstddef>
#include <jfa/jfa.h>
#include <string>
#include <vector_types.h>

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
                sdf(voxelX, voxelY, voxelZ) = INFINITY;
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


template <typename T, int TILE_DIM>
__global__ void ProcessingTiled(const int K, const VoxelsGrid<T, true> grid,
                                const Grid<float> inSDF, const Grid<Position> inPositions,
                                Grid<float> outSDF, Grid<Position> outPositions) 
{
    extern __shared__ unsigned char smem[];
    const int OUT_TILE_DIM = TILE_DIM;
    const int IN_TILE_DIM = OUT_TILE_DIM - (K * 2);

    Grid<float> sdfSMEM = Grid<float>((float*)smem, OUT_TILE_DIM);
    Grid<Position> positionSMEM = Grid<Position>((Position*)(smem + (sdfSMEM.Size() * sizeof(float))), OUT_TILE_DIM);

    for(int i = threadIdx.x; i < sdfSMEM.Size(); i+= blockDim.x) { 
        int smemZ = i / (sdfSMEM.SizeX() * sdfSMEM.SizeX());
        int smemY = (i % (sdfSMEM.SizeX() * sdfSMEM.SizeX())) / sdfSMEM.SizeX();
        int smemX = i % sdfSMEM.SizeX();

        int x = (blockIdx.x * IN_TILE_DIM) + smemX - K;
        int y = (blockIdx.y * IN_TILE_DIM) + smemY - K;
        int z = (blockIdx.z * IN_TILE_DIM) + smemZ - K;

        if(x >= 0 && x < inSDF.SizeX() && y >= 0 && y < inSDF.SizeY() && z >= 0 && z < inSDF.SizeZ()) {
            sdfSMEM(smemX, smemY, smemZ) = inSDF(x, y, z);
            positionSMEM(smemX, smemY, smemZ) = inPositions(x, y, z);   
        } else {
            sdfSMEM(smemX, smemY, smemZ) = -INFINITY;
        }
    }
    
    __syncthreads();

    if(threadIdx.x >= IN_TILE_DIM * IN_TILE_DIM * IN_TILE_DIM) 
        return;

    int smemZ = threadIdx.x / (IN_TILE_DIM * IN_TILE_DIM);
    int smemY = (threadIdx.x % (IN_TILE_DIM * IN_TILE_DIM)) / IN_TILE_DIM;
    int smemX = threadIdx.x % IN_TILE_DIM;

    int globalX = (blockIdx.x * IN_TILE_DIM) + smemX;
    int globalY = (blockIdx.y * IN_TILE_DIM) + smemY;
    int globalZ = (blockIdx.z * IN_TILE_DIM) + smemZ;

    smemX += K;
    smemY += K;
    smemZ += K;

    if(globalX >= inSDF.SizeX() || globalY >= inSDF.SizeY() || globalZ >= inSDF.SizeZ())
        return;

    Position voxelPos = Position(grid.OriginX() + (globalX * grid.VoxelSize()),
                                 grid.OriginY() + (globalY * grid.VoxelSize()),
                                 grid.OriginZ() + (globalZ * grid.VoxelSize()));
 
    bool findNewBest = false;
    float bestDistance = sdfSMEM(smemX, smemY, smemZ);
    Position bestPosition;
    for(int z = -1; z <= 1; z++) {
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                if(x == 0 && y == 0 && z == 0)
                    continue;

                int nx = smemX + (x * K);
                int ny = smemY + (y * K);
                int nz = smemZ + (z * K);

                float seed = sdfSMEM(nx, ny, nz);
                if(fabsf(seed) < INFINITY) {
                    Position seedPos = positionSMEM(nx, ny, nz);

                    float distance = CalculateDistance(voxelPos, seedPos);
                    if(distance < fabsf(bestDistance)) {
                        findNewBest = true;
                        bestDistance = copysignf(distance, bestDistance);
                        bestPosition = seedPos;
                    }
                }
            }
        }
    }

    if (findNewBest) {
        outSDF(globalX, globalY, globalZ) = bestDistance;
        outPositions(globalX, globalY, globalZ) = bestPosition;
    }
}

template <Types type, typename T>
void Compute<Types::TILED, T>(HostVoxelsGrid<T>& grid, HostGrid<float>& sdf)
{   
    assert(grid.View().VoxelsPerSide() % grid.View().WordSize() == 0);
    PROFILING_SCOPE("TiledJFA");

    DeviceVoxelsGrid<T> devGrid;
    DeviceGrid<float> devSDF;
    DeviceGrid<Position> devPositions;
    {
        PROFILING_SCOPE("TiledJFA::Memory");
        devGrid = DeviceVoxelsGrid<T>(grid);
        devSDF = DeviceGrid<float>(sdf);
        devPositions = DeviceGrid<Position>(grid.View().VoxelsPerSide());
    }
    
    {
        PROFILING_SCOPE("TiledJFA::Inizialization");

        const size_t TILE_DIM = 1;
        dim3 blockSize(grid.View().WordSize(), TILE_DIM, TILE_DIM);
        dim3 gridSize(
            (grid.View().VoxelsPerSide() + grid.View().WordSize() * TILE_DIM - 1) / (grid.View().WordSize() * TILE_DIM),
            (grid.View().VoxelsPerSide() + TILE_DIM - 1) / TILE_DIM,
            (grid.View().VoxelsPerSide() + TILE_DIM - 1) / TILE_DIM
        );                                          

        InizializationTiled<T, TILE_DIM><<< gridSize, blockSize >>>(devGrid.View(), devSDF.View(), devPositions.View());

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }


    DeviceGrid<float> appSDF;
    DeviceGrid<Position> appPositions;
    {
        PROFILING_SCOPE("NaiveJFA::Memory");

        appSDF = DeviceGrid<float>(devSDF);
        appPositions = DeviceGrid<Position>(devPositions);
    }

    const int OUT_TILE_DIM = 14;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const size_t numVoxels = grid.View().Size();
    const size_t naiveBlockSize = NextPow2(numVoxels, prop.maxThreadsDim[0] / 2);
    const size_t naiveGridSize = (numVoxels + naiveBlockSize - 1) / naiveBlockSize;


    for (int k = grid.View().VoxelsPerSide() / 2; k >= 1; k /= 2) {
        PROFILING_SCOPE("TiledJFA::Processing::Iteration(" + std::to_string(k) + ")");

        if (k <= 2) {

            const size_t IN_TILE_DIM = OUT_TILE_DIM - (k * 2);
            const size_t tiledBlockSize = NextPow2(IN_TILE_DIM * IN_TILE_DIM * IN_TILE_DIM, 1024); 
            const dim3 tiledGridSize( 
                (grid.View().SizeX() + IN_TILE_DIM - 1) / IN_TILE_DIM,  
                (grid.View().SizeY() + IN_TILE_DIM - 1) / IN_TILE_DIM,  
                (grid.View().SizeZ() + IN_TILE_DIM - 1) / IN_TILE_DIM   
            );  

            ProcessingTiled<T, OUT_TILE_DIM><<< tiledGridSize, tiledBlockSize, (OUT_TILE_DIM * OUT_TILE_DIM * OUT_TILE_DIM) * (sizeof(float) + sizeof(Position)) >>>(
                k, devGrid.View(),  
                devSDF.View(), devPositions.View(), 
                appSDF.View(), appPositions.View() 
            ); 
 
        } else {  
            ProcessingNaive<T><<< naiveGridSize, naiveBlockSize >>>(
                k, devGrid.View(), 
                devSDF.View(), devPositions.View(),
                appSDF.View(), appPositions.View()
            );
        }
        gpuAssert(cudaPeekAtLastError()); 
        cudaDeviceSynchronize();

        devSDF = appSDF;
        devPositions = appPositions;
    } 

    {
        PROFILING_SCOPE("TiledJFA::Memory");
        sdf = HostGrid<float>(devSDF);
    }
}


template void Compute<Types::TILED, uint32_t>
(HostVoxelsGrid<uint32_t>&, HostGrid<float>&);

template void Compute<Types::TILED, uint64_t>
(HostVoxelsGrid<uint64_t>&, HostGrid<float>&);

template __global__ void InizializationTiled<uint32_t>
(const VoxelsGrid<uint32_t, true>, Grid<float>, Grid<Position>);

template __global__ void InizializationTiled<uint64_t>
(const VoxelsGrid<uint64_t, true>, Grid<float>, Grid<Position>);

template __global__ void ProcessingTiled<uint32_t>
(const int, const VoxelsGrid<uint32_t, true>, const Grid<float>, const Grid<Position>, Grid<float>, Grid<Position>);

template __global__ void ProcessingTiled<uint64_t>
(const int, const VoxelsGrid<uint64_t, true>, const Grid<float>, const Grid<Position>, Grid<float>, Grid<Position>);

}
