/* Naive solution of 7-point stencil with 3D grid */

#include<iostream>
#include<stdlib.h>

#define C0 0.5
#define C1 0.5
#define C2 0.5
#define C3 0.5
#define C4 0.5
#define C5 0.5
#define C6 0.5

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

__global__ void naive_stencils(float *IN, float *OUT, unsigned int N)
{
    unsigned int iz = (blockIdx.z*blockDim.z) + threadIdx.z;
    unsigned int iy = (blockIdx.y*blockDim.y) + threadIdx.y;
    unsigned int ix = (blockIdx.x*blockDim.x) + threadIdx.x;
    if (iz >= 1 && iz < N-1 && iy >= 1 && iy < N-1 && ix >= 1 && ix < N-1) {
        OUT[iz*N*N + iy*N + ix] = C0 * IN[iz*N*N + iy*N + ix]
                             + C1 * IN[iz*N*N + iy*N + (ix - 1)]
                             + C2 * IN[iz*N*N + iy *N + (ix + 1)]
                             + C3 * IN[iz*N*N + (iy - 1)*N + ix]
                             + C4 * IN[iz*N*N + (iy + 1)*N + ix]
                             + C5 * IN[(iz - 1)*N*N + iy*N + ix]
                             + C6 * IN[(iz + 1)*N*N + iy*N + ix];
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <L> <B>" << std::endl;
        exit(1);
    }

    int L = atoi(argv[1]);
    int B = atoi(argv[2]);

    // initialization of the host arrays
    float *host_A = (float *) malloc(sizeof(float) * L * L * L);
    float *host_B = (float *) malloc(sizeof(float) * L * L * L);

    for (int i=0; i<L*L*L; i++) {
        host_A[i] = rand() % 100;
    }

    // Allocate GPU arrays
    cudaSetDevice(0); // set the working device
    float *dev_A;
    float *dev_B;
    gpuErrchk(cudaMalloc((void**) &dev_A, L*L*L*sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &dev_B, L*L*L*sizeof(float)));
    cudaMemset(dev_B, 0, sizeof(float)*L*L*L);

    uint64_t initial_time = current_time_nsecs();

    // Copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_A, host_A, L*L*L*sizeof(float), cudaMemcpyHostToDevice));

    uint64_t initial_time2 = current_time_nsecs();

    // Perform computation on GPU
    dim3 gridDim(std::ceil(((float) L)/B), std::ceil(((float) L)/B), std::ceil(((float) L)/B));
    dim3 blockDim(B, B, B);
    naive_stencils<<<gridDim, blockDim>>>(dev_A, dev_B, L);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    uint64_t end_time2 = current_time_nsecs();
    uint64_t elapsed2 = end_time2 - initial_time2;
    std::cout << "Kernel time: " << ((float) elapsed2)/1000.0 << " usec" << std::endl;

    // Copy results from GPU memory
    gpuErrchk(cudaMemcpy(host_B, dev_B, L*L*L*sizeof(float), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // Deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_B));

    float *check_B = (float *) malloc(sizeof(float) * L * L * L);
    memset(check_B, 0, sizeof(float) * L * L *L);

    for (int k=1; k<L-1; k++) {
        for (int i=1; i<L-1; i++) {
            for (int j=1; j<L-1; j++) {            
                check_B[k*L*L + i*L + j] = C0 * host_A[k*L*L + i*L + j]
                                     + C1 * host_A[k*L*L + i*L + j - 1]
                                     + C2 * host_A[k*L*L + i*L + j + 1]
                                     + C3 * host_A[k*L*L + (i-1)*L + j]
                                     + C4 * host_A[k*L*L + (i+1)*L + j]
                                     + C5 * host_A[(k-1)*L*L + i*L + j]
                                     + C6 * host_A[(k+1)*L*L + i*L + j];
            } 
        }
    }

    for (int i=0; i<L*L*L; i++) {
        if (check_B[i] != host_B[i]) {
            std::cout << "Result error!" << std::endl;
            abort();
        }
    }
    std::cout << "Result is ok!" << std::endl;

    // Deallocate host memory
    free(host_A);
    free(host_B);
    free(check_B);
    return 0;
}
