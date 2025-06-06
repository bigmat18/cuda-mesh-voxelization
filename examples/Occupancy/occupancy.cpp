/* Example of occupancy calculation */

#include<iostream>
#include<stdlib.h>

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    d[idx] = a[idx] * b[idx];
}

// Host code
int main() {
    int numBlocks;
    int device;
    cudaDeviceProp prop;
    int residentWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    for (int blockSize = 2; blockSize <= 1024; blockSize*=2) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, MyKernel, blockSize, 0);
        residentWarps = numBlocks * blockSize / prop.warpSize;
        double occup = (double) residentWarps / maxWarps * 100;
        printf("blockSize = %4d <-> Occupancy [numBlocks = %2d,  activeWarps = %2d]:\t%2.2f%%\n", blockSize, numBlocks, residentWarps, occup);
    }
    return 0;
}
