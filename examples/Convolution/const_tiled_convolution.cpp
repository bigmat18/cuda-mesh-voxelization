/* Solution of parallel convolution on GPU with constant memory and tiling */

#include<iostream>
#include<stdlib.h>

#define RADIUS 8
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM-2*RADIUS)
#define FILTER_SIZE (2*RADIUS+1)*(2*RADIUS+1)

// declaration of the filter in constant memory (global variable)
__constant__ float dev_F[2*RADIUS+1][2*RADIUS+1];

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

__global__ void tiled_conv(float *A, float *B, int w, int h)
{
    // loading the input tile
    __shared__ float A_smem[IN_TILE_DIM][IN_TILE_DIM];
    int ix = (blockIdx.x * OUT_TILE_DIM) + threadIdx.x - RADIUS;
    int iy = (blockIdx.y * OUT_TILE_DIM) + threadIdx.y - RADIUS;
    if (ix >= 0 && ix < w && iy >=0 && iy < h) {
        A_smem[threadIdx.y][threadIdx.x] = A[iy * w + ix];
    }
    else {
        A_smem[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    int tileX = threadIdx.x - RADIUS;
    int tileY = threadIdx.y - RADIUS;
    // writing the output tile
    if (ix >= 0 && ix < w && iy >=0 && iy < h) { // active thread
        if (tileX >= 0 && tileX < OUT_TILE_DIM && tileY >=0 && tileY < OUT_TILE_DIM) {
            float pvalue = 0;
            for (int fy=0; fy<2*RADIUS+1; fy++) {
                for(int fx=0; fx<2*RADIUS+1; fx++) {
                    pvalue += dev_F[fy][fx] * A_smem[(tileY + fy)][(tileX + fx)];
                }
            }
            B[iy * w + ix] = pvalue;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <L>" << std::endl;
        exit(1);
    }

    int L = atoi(argv[1]);

    // initialization of the host arrays
    float *host_A = (float *) malloc(sizeof(float) * L * L);
    float *host_B = (float *) malloc(sizeof(float) * L * L);
    float host_F[FILTER_SIZE];

    for (int i=0; i<L*L; i++) {
        host_A[i] = rand() % 100;
    }

    for (int i=0; i<FILTER_SIZE; i++) {
        host_F[i] = rand() % 100;
    }

    // Allocate GPU arrays
    cudaSetDevice(0); // set the working device
    float *dev_A;
    float *dev_B;
    gpuErrchk(cudaMalloc((void**) &dev_A, L*L*sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &dev_B, L*L*sizeof(float)));

    uint64_t initial_time = current_time_nsecs();

    // Copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_A, host_A, L*L*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(dev_F, host_F, FILTER_SIZE*sizeof(float)));

    uint64_t initial_time2 = current_time_nsecs();

    // Perform computation on GPU
    dim3 gridDim(std::ceil(((float) L)/OUT_TILE_DIM), std::ceil(((float) L)/OUT_TILE_DIM));
    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
 
    tiled_conv<<<gridDim, blockDim>>>(dev_A, dev_B, L, L);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    uint64_t end_time2 = current_time_nsecs();
    uint64_t elapsed2 = end_time2 - initial_time2;
    std::cout << "Kernel time: " << ((float) elapsed2)/1000.0 << " usec" << std::endl;

    // Copy results from GPU memory
    gpuErrchk(cudaMemcpy(host_B, dev_B, L*L*sizeof(float), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // Deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_B));

    float *check_B = (float *) malloc(sizeof(float) * L * L);

    for (int i=0; i<L; i++) {
        for (int j=0; j<L; j++) {
            float v = 0;
            for (int iff=0; iff<2*RADIUS+1; iff++) {
                for (int jff=0; jff<2*RADIUS+1; jff++) {
                    int fi = i - RADIUS + iff;
                    int fj = j - RADIUS + jff;
                    if (fi >= 0 && fi < L && fj >= 0 && fj < L) {
                        v += host_F[iff * (2*RADIUS+1) + jff] * host_A[fi * L + fj];
                    }
                }
            }
            check_B[i * L + j] = v;
        }
    }

    for (int i=0; i<L*L; i++) {
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
