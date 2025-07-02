#include <cstddef>
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

    const int numWord = grid1.SpaceSize() * grid1.WordSize();
    const int wordIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int bitIndex = wordIndex * grid1.WordSize();

    if(wordIndex >= numWord)
        return;

    const int z = bitIndex / (grid1.VoxelsPerSide() * grid1.VoxelsPerSide());
    const int y = (bitIndex % (grid1.VoxelsPerSide() * grid1.VoxelsPerSide())) / grid1.VoxelsPerSide();
    const int x = bitIndex % grid1.VoxelsPerSide();

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

        const size_t numWord = (grid1.SpaceSize() + grid1.WordSize() - 1) / grid1.WordSize();
        const size_t blockSize = NextPow2(numWord, prop.maxThreadsDim[0] / 2);
        const size_t gridSize = (numWord + blockSize - 1) / blockSize;
        
        CSGProcessing<T><<< gridSize, blockSize >>>(grid1, grid2, Op);

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize(); 
    }
}

}

#endif // !CSG_H

