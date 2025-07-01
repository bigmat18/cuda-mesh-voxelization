#include <functional>
#include <voxels_grid.h>
#include <debug_utils.h>
#ifndef CSG_H

namespace CSG {

template <typename T> 
struct Union {
    __host__ __device__
    void operator() (T& el, T value) { el |= value; }
};

template <typename T> 
struct Intersection {
    __host__ __device__
    void operator() (T& el, T value) { el &= value; }
};

template <typename T> 
struct Difference {
    __host__ __device__
    void operator() (T& el, T value) { el &= ~value; }
};

template <typename T, typename func>
__global__ void CSGProcessing(VoxelsGrid<T, true> grid1, VoxelsGrid<T, true> grid2, func Op)
{

    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * grid1.WordSize();
    const int y = (blockIdx.y * blockDim.y + threadIdx.y) * grid1.WordSize();
    const int z = (blockIdx.z * blockDim.z + threadIdx.z) * grid1.WordSize();

    if(x >= grid1.VoxelsPerSide() || y >= grid1.VoxelsPerSide() || z >= grid1.VoxelsPerSide())
        return;

    T word = 0;
    for(int i = 0; i < grid2.WordSize(); ++i) 
        word |= grid2(x + i, y, z) << i;
    
    grid1.SetWord(x, y, z, word, Op);
}

template <typename T, bool device, typename func>
void Compute(VoxelsGrid<T, device> grid1, VoxelsGrid<T, device> grid2, func Op)
{ 
    if constexpr (device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        const size_t numWord = grid1.VoxelsPerSide() / grid1.WordSize();
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks(
            (numWord + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (numWord + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (numWord + threadsPerBlock.z - 1) / threadsPerBlock.z
        );
        
        CSGProcessing<T><<< numBlocks, threadsPerBlock >>>(grid1, grid2, Op);

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize(); 
    }
}

}

#endif // !CSG_H

