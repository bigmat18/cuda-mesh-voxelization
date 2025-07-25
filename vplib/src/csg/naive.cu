#include "proc_utils.h"
#include "profiling.h"
#include <csg/csg.h>

namespace CSG {

template <typename T, typename func>
__global__ void ProcessingNaive(VoxelsGrid<T, true> grid1, VoxelsGrid<T, true> grid2, func Op)
{

    const int numWord = grid1.Size() * grid1.WordSize();
    const int wordIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int voxelIndex = wordIndex * grid1.WordSize();

    if(wordIndex >= numWord)
        return;

    const int z = voxelIndex / (grid1.VoxelsPerSide() * grid1.VoxelsPerSide());
    const int y = (voxelIndex % (grid1.VoxelsPerSide() * grid1.VoxelsPerSide())) / grid1.VoxelsPerSide();
    const int x = voxelIndex % grid1.VoxelsPerSide();

    Op(grid1.Word(x, y, z), grid2.Word(x, y, z));
}



template <Types type, typename T, typename func>
void Compute<Types::NAIVE, T, func>(HostVoxelsGrid<T>& grid1, HostVoxelsGrid<T>& grid2, func Op)
{ 
    PROFILING_SCOPE("NaiveCSG");
    cpuAssert(grid1.View().VoxelsPerSide() == grid2.View().VoxelsPerSide(), 
              "grid1 and grid2 must have same voxels per side");
    cpuAssert(grid1.View().VoxelSize() == grid2.View().VoxelSize(),
              "grid1 and grid2 must have same side length");

    DeviceVoxelsGrid<T> devGridFirst;
    DeviceVoxelsGrid<T> devGridSecond;
    {

        PROFILING_SCOPE("NaiveCSG::Memory");
        devGridFirst = DeviceVoxelsGrid<T>(grid1);
        devGridSecond = DeviceVoxelsGrid<T>(grid2);
    }


    {   
        PROFILING_SCOPE("NaiveCSG::Processing");
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        const size_t numWord = (grid1.View().Size() + grid1.View().WordSize() - 1) / grid1.View().WordSize();
        const size_t blockSize = NextPow2(numWord, prop.maxThreadsDim[0] / 2);
        const size_t gridSize = (numWord + blockSize - 1) / blockSize;

        ProcessingNaive<T><<< gridSize, blockSize >>>(devGridFirst.View(), devGridSecond.View(), Op);

        gpuAssert(cudaPeekAtLastError());
        cudaDeviceSynchronize(); 
    }

    {
        PROFILING_SCOPE("NaiveCSG::Memory");
        grid1 = HostVoxelsGrid<T>(devGridFirst);
    }
}

////////////////////////////// Union OP ///////////////////////////////
template __global__ void ProcessingNaive<uint32_t, Union<uint32_t>>
(VoxelsGrid<uint32_t, true>, VoxelsGrid<uint32_t, true>, Union<uint32_t>);

template __global__ void ProcessingNaive<uint64_t, Union<uint64_t>>
(VoxelsGrid<uint64_t, true>, VoxelsGrid<uint64_t, true>, Union<uint64_t>);

template void Compute<Types::NAIVE, uint32_t, Union<uint32_t>>
(HostVoxelsGrid<uint32_t>&, HostVoxelsGrid<uint32_t>&, Union<uint32_t>);

template void Compute<Types::NAIVE, uint64_t, Union<uint64_t>>
(HostVoxelsGrid<uint64_t>&, HostVoxelsGrid<uint64_t>&, Union<uint64_t>);
///////////////////////////// Union OP ///////////////////////////////


////////////////////////////// Intersection OP ///////////////////////////////
template __global__ void ProcessingNaive<uint32_t, Intersection<uint32_t>>
(VoxelsGrid<uint32_t, true>, VoxelsGrid<uint32_t, true>, Intersection<uint32_t>);

template __global__ void ProcessingNaive<uint64_t, Intersection<uint64_t>>
(VoxelsGrid<uint64_t, true>, VoxelsGrid<uint64_t, true>, Intersection<uint64_t>);

template void Compute<Types::NAIVE, uint32_t, Intersection<uint32_t>>
(HostVoxelsGrid<uint32_t>&, HostVoxelsGrid<uint32_t>&, Intersection<uint32_t>);

template void Compute<Types::NAIVE, uint64_t, Intersection<uint64_t>>
(HostVoxelsGrid<uint64_t>&, HostVoxelsGrid<uint64_t>&, Intersection<uint64_t>);
///////////////////////////// Intersection OP ///////////////////////////////


////////////////////////////// Difference OP ///////////////////////////////
template __global__ void ProcessingNaive<uint32_t, Difference<uint32_t>>
(VoxelsGrid<uint32_t, true>, VoxelsGrid<uint32_t, true>, Difference<uint32_t>);

template __global__ void ProcessingNaive<uint64_t, Difference<uint64_t>>
(VoxelsGrid<uint64_t, true>, VoxelsGrid<uint64_t, true>, Difference<uint64_t>);

template void Compute<Types::NAIVE, uint32_t, Difference<uint32_t>>
(HostVoxelsGrid<uint32_t>&, HostVoxelsGrid<uint32_t>&, Difference<uint32_t>);

template void Compute<Types::NAIVE, uint64_t, Difference<uint64_t>>
(HostVoxelsGrid<uint64_t>&, HostVoxelsGrid<uint64_t>&, Difference<uint64_t>);
///////////////////////////// Difference OP ///////////////////////////////
}

