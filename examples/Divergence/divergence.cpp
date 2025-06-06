/* CUDA application testing warp divergence */

#include<iostream>
#include<stdio.h>
#include<stdlib.h>

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

__global__ void kernel_test(int *A, int *B, int *C, int size)
{
    unsigned int tUID = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tUID >= size) {
        return;
    }
    if (tUID % 2 == 0) { // event
        C[tUID] = A[tUID] + B[tUID];
    }
    else { // odd
        C[tUID] = A[tUID] - B[tUID];
    }
}

int main(int argc, char **argv)
{
    int *A_cpu, *B_cpu, *C_cpu;
    int *A_gpu, *B_gpu, *C_gpu;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <size> <blocksize>" << std::endl;
        exit(EXIT_FAILURE);
    }

    int size = atoi(argv[1]);
    int b = atoi(argv[2]);

    A_cpu = (int *) malloc(sizeof(int) * size);
    B_cpu = (int *) malloc(sizeof(int) * size);
    C_cpu = (int *) malloc(sizeof(int) * size);

    for (int i=0; i<size; i++) {
        A_cpu[i] = rand() % 100;
        B_cpu[i] = rand() % 100;
        C_cpu[i] = 0;
    }

    cudaSetDevice(0); // set the working device
    // Allocate device memory for the input and output images
    gpuErrchk(cudaMalloc((void**) &A_gpu, sizeof(int) * size));
    gpuErrchk(cudaMalloc((void**) &B_gpu, sizeof(int) * size));
    gpuErrchk(cudaMalloc((void**) &C_gpu, sizeof(int) * size));

    uint64_t initial_time = current_time_nsecs();

    // Copy host memory buffers into device memory buffers
    gpuErrchk(cudaMemcpy(A_gpu, A_cpu, sizeof(int) * size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(B_gpu, B_cpu, sizeof(int) * size, cudaMemcpyHostToDevice));

    uint64_t initial_time2 = current_time_nsecs();

    int n_blocks = std::ceil(((double) size) / b);
    kernel_test<<<n_blocks, b>>>(A_gpu, B_gpu, C_gpu, size);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    uint64_t end_time2 = current_time_nsecs();

    // Copy output (results) from GPU buffer to host (CPU) memory
    gpuErrchk(cudaMemcpy(C_cpu, C_gpu, sizeof(int) * size, cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed2 = end_time2 - initial_time2;
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Kernel time: " << ((float) elapsed2)/1000.0 << " usec" << std::endl;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    for (size_t i=0; i<size; i++) {
        if (i % 2 == 0) {
            if (C_cpu[i] != A_cpu[i] + B_cpu[i]) {
                std::cerr << "Error result" << std::endl;
                exit(1);
            }
        }
        else {
            if (C_cpu[i] != A_cpu[i] - B_cpu[i]) {
                std::cerr << "Error result" << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Result is ok!" << std::endl;

    // Deallocate CPU, GPU memory
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    free(A_cpu);
    free(B_cpu);
    free(C_cpu);
    return 0;
}
