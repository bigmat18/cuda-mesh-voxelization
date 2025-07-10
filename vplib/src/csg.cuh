#include <cassert>
#include <cstddef>
#include <voxels_grid.h>
#include <debug_utils.h>
#ifndef CSG_H

namespace CSG {

enum class Types {
    UNION, INTERSECTION, DIFFERENCE, VOID
};

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
    const int voxelIndex = wordIndex * grid1.WordSize();

    if(wordIndex >= numWord)
        return;

    const int z = voxelIndex / (grid1.VoxelsPerSide() * grid1.VoxelsPerSide());
    const int y = (voxelIndex % (grid1.VoxelsPerSide() * grid1.VoxelsPerSide())) / grid1.VoxelsPerSide();
    const int x = voxelIndex % grid1.VoxelsPerSide();

    T word = 0;
    for(int i = 0; i < grid2.WordSize(); ++i) 
        word |= grid2.Voxel(x + i, y, z) << i;
    
    Op(grid1.Word(x, y, z), word);
}

template <typename T, typename func>
void Compute(DeviceVoxelsGrid<T>& grid1, DeviceVoxelsGrid<T>& grid2, func Op)
{ 
    cpuAssert(grid1.View().VoxelsPerSide() == grid2.View().VoxelsPerSide(), 
              "grid1 and grid2 must have same voxels per side");
    cpuAssert(grid1.View().SideLength() == grid2.View().SideLength(),
              "grid1 and grid2 must have same side length");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const size_t numWord = (grid1.View().SpaceSize() + grid1.View().WordSize() - 1) / grid1.View().WordSize();
    const size_t blockSize = NextPow2(numWord, prop.maxThreadsDim[0] / 2);
    const size_t gridSize = (numWord + blockSize - 1) / blockSize;
        
    CSGProcessing<T><<< gridSize, blockSize >>>(grid1.View(), grid2.View(), Op);

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize(); 
}

}

#endif // !CSG_H

