/* Hello world kernel with dynamic parallelism in CUDA */

#include<iostream>

extern __global__ void nestedHelloWorld(const int iSize, int iDepth);

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Wrong number of parameters" << std::endl;
        exit(1);
    }
    int n = atoi(argv[1]); // n is the number of threads of the kernel
    cudaSetDevice(0); // set the working device
    nestedHelloWorld<<<1, n>>>(n, 0); // launch the parent kernel
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize(); // wait for the parent kernel completion
}
