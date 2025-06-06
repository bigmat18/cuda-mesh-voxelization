/* Simple kernel to profile CUDA streams (Version 2) */

#include<iostream>
#include<stdlib.h>

#define N 8192

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

__global__ void kernel(float *x, int n)
{
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(2,i));
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n_streams>" << std::endl;
        exit(1);
    }

    int num_streams = atoi(argv[1]);
    cudaStream_t *streams = (cudaStream_t *) malloc(sizeof(cudaStream_t) * num_streams);
    float **data;
    data = (float **) malloc(sizeof(float *) * num_streams);

    for (int i=0; i < num_streams; i++) {    
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&data[i], N * sizeof(float));
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);
        // kernel<<<1, 1>>>(0, 0);
    }
    return 0;
}
