/* Naive CUDA version of a kernel to tranpose a matrix */

#include<iostream>
#include<stdlib.h>

#define BLOCK_DIM 16 // blocks of BLOCK_DIM*BLOCKD_DIM threads

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

__global__ void naive_transpose(float *A, float *T, int N, int M)
{
    int ix = (blockDim.x * blockIdx.x) + threadIdx.x;
    int iy = (blockDim.y * blockIdx.y) + threadIdx.y;
    if (ix < M && iy < N) {
        T[(ix * N) + iy] = A[(iy * M) + ix];
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr<< "Usage: " << argv[0] << " <N> <M>" << std::endl;
        exit(1);
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    // initialization of the host arrays host_A and host_B
    float *host_A = (float *) malloc(sizeof(float) * N * M);
    float *host_T = (float *) malloc(sizeof(float) * N * M);

    for (int i=0; i<N*M; i++) {
        host_A[i] = rand() % 100;
    }

    // Allocate GPU arrays
    cudaSetDevice(0); // set the working device
    float *dev_A, *dev_T;
    gpuErrchk(cudaMalloc((void**) &dev_A, N*M*sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &dev_T, N*M*sizeof(float)));

    uint64_t initial_time = current_time_nsecs();

    // Copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_A, host_A, N*M*sizeof(float), cudaMemcpyHostToDevice));

    // Invoke kernels (define grid and block sizes)
    int rowBlocks = std::ceil(((double) M) / BLOCK_DIM);
    int columnBlocks = std::ceil(((double) N) / BLOCK_DIM);
    dim3 gridDim(rowBlocks, columnBlocks);
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);

    uint64_t initial_time2 = current_time_nsecs();

    naive_transpose<<<gridDim, blockDim>>>(dev_A, dev_T, N, M);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    uint64_t end_time2 = current_time_nsecs();

    // Copy results from GPU memory
    gpuErrchk(cudaMemcpy(host_T, dev_T, N*M*sizeof(float), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed2 = end_time2 - initial_time2;
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Kernel time: " << ((float) elapsed2)/1000.0 << " usec" << std::endl;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // Deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_T));

    float *check_T = (float *) malloc(sizeof(float)*N*M);
    for (int i=0; i<N; i++) {
        for (int j=0; j<M; j++) {
            check_T[(j*N) + i] = host_A[(i*M) + j];
            if (check_T[(j*N) + i] != host_T[(j*N) + i]) {
                std::cerr << "Error result" << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Result is ok!" << std::endl;

    // Deallocate host memory
    free(host_A);
    free(host_T);
    free(check_T);
    return 0;
}
