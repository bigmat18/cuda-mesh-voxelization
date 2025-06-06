/* Solution of parallel reduction on GPU with a strided approach */

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

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

__global__ void strided_reduction(int *A, int L, int *sum)
{
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >=1; stride /= 2) {
        if (threadIdx.x < stride) {
            if (i+stride < L) {
                A[i] += A[i + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sum, A[i]);
    }
}

bool isPowerOfTwo(int n)
{
    return (ceil(log2(n)) == floor(log2(n)));
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <L> <block_size (power of two)>" << std::endl;
        exit(1);
    }

    int L = atoi(argv[1]);
    int block_size = atoi(argv[2]);

    if (!isPowerOfTwo(block_size)) {
        std::cerr << "block_size is not a power of two!" << std::endl;
        exit(1);
    }

    // initialization of the host array and result variable
    int *host_A = (int *) malloc(sizeof(int) * L);
    int *host_result = (int *) malloc(sizeof(int));
    *host_result = 0;

    for (int i=0; i<L; i++) {
        host_A[i] = rand() % 100;
    }

    // Allocate GPU array and result variable
    cudaSetDevice(0); // set the working device
    int *dev_A;
    int *dev_result;
    gpuErrchk(cudaMalloc((void**) &dev_A, L*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &dev_result, sizeof(int)));

    uint64_t initial_time = current_time_nsecs();

    // Copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_A, host_A, L *sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_result, host_result, sizeof(int), cudaMemcpyHostToDevice));

    uint64_t initial_time2 = current_time_nsecs();

    // Perform computation on GPU
    unsigned int numBlocks = std::ceil(((double) L) / (2*block_size));
    strided_reduction<<<numBlocks, block_size>>>(dev_A, L, dev_result);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    uint64_t end_time2 = current_time_nsecs();
    uint64_t elapsed2 = end_time2 - initial_time2;
    std::cout << "Kernel time: " << ((float) elapsed2)/1000.0 << " usec" << std::endl;

    // Copy results from GPU memory
    gpuErrchk(cudaMemcpy(host_result, dev_result, sizeof(int), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // Deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_result));

    int check = 0;
    for (int i=0; i<L; i++) {
        check += host_A[i];
    }
    if (check != *host_result) {
        std::cout << "Result error!" << std::endl;
    }
    else {
        std::cout << "Result is ok!" << std::endl;
    }

    // Deallocate host memory
    free(host_A);
    free(host_result);
    return 0;
}
