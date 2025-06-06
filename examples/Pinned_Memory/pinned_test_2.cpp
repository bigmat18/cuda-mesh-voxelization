/* Test of CUDA Pinned Memory */

#include<iostream>
#include<stdlib.h>
#include <stdint.h>


inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

__global__ void print_kernel(float *elem)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx == 0) {
        printf("Value of float in pinned memory: %f\n", *elem);
    }
}

int main(int argc, char **argv)
{
    float *h_a;
    cudaMallocHost(&h_a, sizeof(float));
    *h_a = 10.725;
    print_kernel<<<1, 16>>>(h_a);
    cudaDeviceSynchronize();
    cudaFreeHost(h_a);
    return 0;
}
