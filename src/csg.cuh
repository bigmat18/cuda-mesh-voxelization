#include <functional>
#include <voxels_grid.h>
#include <debug_utils.h>
#ifndef CSG_H

namespace CSG {

template <typename T> __host__ __device__
void Union(T& el, T value) { el |= value; }

template <typename T> __host__ __device__
void Intersection(T& el, T value) { el &= value; }

template <typename T> __host__ __device__
void Difference(T& el, T value) { el &= !value; }

template <typename T>
__global__ void CSGProcessing(VoxelsGrid<T, true> grid1, VoxelsGrid<T, true> grid2, std::function<void(T&, T)> Op)
{
    const int wordX = blockIdx.x * blockDim.x + threadIdx.x;
    const int wordY = blockIdx.y * blockDim.y + threadIdx.y;
    const int wordZ = blockIdx.z * blockDim.z + threadIdx.z;

    T word = 0;
    for(int i = 0; i < grid2.WordSize(); ++i) 
        word |= grid2((wordX * grid2.WordSize()) + i, (wordY * grid2.WordSize()), (wordZ * grid2.WordSize())) << 0;
    
    grid1.SetWord(wordX * grid1.WordSize(), wordY * grid1.WordSize(), wordZ * grid1.WordSize(), word, Op);
}

template <typename T, bool device>
void Compute(VoxelsGrid<T, device> grid1, VoxelsGrid<T, device> grid2, std::function<void(T&, T)> Op)
{

    if constexpr (device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        const size_t numWord = grid1.VoxelsPerSide() / grid1.WordSize();
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks(
            (numWord + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (numWord + threadsPerBlock.y - 1) / threadsPerBlock.y,
            ((numWord + 31) / 32 + threadsPerBlock.z - 1) / threadsPerBlock.z
        );
        
        CSGProcessing<T><<< numBlocks, threadsPerBlock >>>(grid1, grid2, Op);

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize(); 
    }
}

}

#endif // !CSG_H

