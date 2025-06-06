/* Example of using global device variables in CUDA */

#include<iostream>
#include<stdio.h>

__device__ int d_value[10];
int h_value[10];

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void write_value()
{
    d_value[threadIdx.x] += threadIdx.x;
    printf("Value GPU = %d\n", d_value[threadIdx.x]);
}

int main()
{
    for (int i = 0; i < 10; i++) {
        h_value[i] = 10*i;
    }

    cudaSetDevice(0); // set the working device
    gpuErrchk(cudaMemcpyToSymbol(d_value, h_value, sizeof(h_value)));
    write_value<<<1, 10>>>();
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpyFromSymbol(h_value, d_value, sizeof(h_value)));
    for (int i = 0; i < 10; i++) {
        printf("Value CPU [%d] = %d\n", i, h_value[i]);
    }
    return 0;
}
