/* First CUDA kernel printing "Hello World" on the STDOUT */

#include <iostream>

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void helloWorld_GPU(void)
{
    // int threadUID = warpSize;
    int local[300];
    for (int i =0; i < 300; ++i) {
        local[i] = i;
        printf("Hello world from the GPU by thread %d\n", local[i]);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Wrong number of parameters" << std::endl;
        exit(1);
    }
    int n = atoi(argv[1]);      // n is the number of threads of the kernel
    cudaSetDevice(0);           // set the working device (0 is the default one)
    helloWorld_GPU<<<1, n>>>(); // launch the kernel
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();    // wait for the kernel completion
}
