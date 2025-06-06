/* Streamed version of VectorSUM */

#include<iostream>
#include<stdlib.h>

#define BDIM 512

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

__global__ void vector_sum(int *A, int *B, int *C, int L)
{
    int threadUID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadUID < L) {
        C[threadUID] = A[threadUID] + B[threadUID];
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr<< "Usage: " << argv[0] << " <size> <n_streams>" << std::endl;
        exit(1);
    }
    int size = atoi(argv[1]);
    int n_streams = atoi(argv[2]);

    if (size % n_streams != 0) {
        std::cerr << "The size of the array must be divisible by the number of streams" << std::endl;
        exit(1);
    }

    int *host_A, *host_B, *host_C;
    // initialization of the host arrays host_A and host_B
    cudaMallocHost(&host_A, sizeof(int) * size);
    cudaMallocHost(&host_B, sizeof(int) * size);
    cudaMallocHost(&host_C, sizeof(int) * size);

    for (int i=0; i<size; i++) {
        host_A[i] = rand() % 100;
        host_B[i] = rand() % 100;
    }

    // Allocate GPU arrays
    cudaSetDevice(0); // set the working device
    int *dev_A, *dev_B, *dev_C;
    gpuErrchk(cudaMalloc((void**) &dev_A, size*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &dev_B, size *sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &dev_C, size *sizeof(int)));

    int iElem = size / n_streams;
    size_t iBytes = iElem * sizeof(int);

    cudaStream_t stream[n_streams];
    for (int i=0; i<n_streams; ++i) {
        cudaStreamCreate(&stream[i]); 
    }

    dim3 block(BDIM);
    dim3 grid (std::ceil(size / BDIM));

    uint64_t initial_time = current_time_nsecs();

    // initiate all work on the device asynchronously in depth-first order
    for (int i=0; i<n_streams; i++) {
        int ioffset = i * iElem;
        cudaMemcpyAsync(&dev_A[ioffset], &host_A[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);      
        cudaMemcpyAsync(&dev_B[ioffset], &host_B[ioffset], iBytes, cudaMemcpyHostToDevice, stream[i]);
        vector_sum<<<grid, block, 0, stream[i]>>>(&dev_A[ioffset], &dev_B[ioffset], &dev_C[ioffset], iElem);     
        cudaMemcpyAsync(&host_C[ioffset], &dev_C[ioffset], iBytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    for (int i=0; i < n_streams; i++) {
        cudaStreamSynchronize(stream[i]);
    }

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed/1000.0) << " usec" << std::endl;

    // Deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_B));
    gpuErrchk(cudaFree(dev_C));

    int *check = (int *) malloc(sizeof(int) * size);
    for (int i=0; i<size; i++) {
        check[i] = host_A[i] + host_B[i];
        if (check[i] != host_C[i]) {
            std::cerr << "Error result" << std::endl;
            exit(1);
        }
    }
    std::cout << "Result is ok!" << std::endl;

    // Deallocate host memory
    cudaFreeHost(host_A);
    cudaFreeHost(host_B);
    cudaFreeHost(host_C);
    free(check);

    // Deallocate streams
    for (int i=0; i < n_streams; i++) {
        cudaStreamDestroy(stream[i]);
    }

    return 0;
}
