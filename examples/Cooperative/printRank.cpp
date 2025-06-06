/* First use of cooperative groups */

#include<iostream>
#include<cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__device__ void printRank(cg::thread_group g)
{
    printf("Rank %d\n", g.thread_rank());
}

__global__ void allPrint()
{
    cg::thread_block b = cg::this_thread_block();
    printRank(b);
}

int main(int argc, char **argv)
{
    cudaSetDevice(0); // set the working device
    allPrint<<<1, 23>>>(); // launch the kernel
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize(); // wait for the kernel completion
}
