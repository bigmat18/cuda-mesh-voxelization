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

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr<< "Usage: " << argv[0] << " <num_elements>" << std::endl;
        exit(1);
    }
    int size = atoi(argv[1]);

    float *h_a;
    float *d_a;
    cudaMallocHost(&h_a, sizeof(float) * size);
    cudaMalloc(&d_a, sizeof(float) * size);
    cudaMemcpy(d_a, h_a, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaFreeHost(h_a);
    cudaFree(d_a);
    return 0;
}
