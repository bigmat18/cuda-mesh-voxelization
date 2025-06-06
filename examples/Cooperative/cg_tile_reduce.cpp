/* Tile reduce using cooperative groups */

#include<iostream>
#include<cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define TILE_SIZE 16

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

template<unsigned int size>
__device__ int tile_reduce(cg::thread_block_tile<size> g, int val)
{
    for (int i=g.size()/2; i>0; i /=2) {
        val += g.shfl_down(val, i);
    }
    return val;
}

__global__ void call_reduce(int *array, int *reduce_result)
{
    int myval = array[(blockIdx.x * blockDim.x) + threadIdx.x];
    array[(blockIdx.x * blockDim.x) + threadIdx.x] = 0;
    auto tile = cg::tiled_partition<TILE_SIZE>(this_thread_block());
    int tile_result = tile_reduce<TILE_SIZE>(tile, myval);
    if (tile.thread_rank() == 0) {
        atomicAdd(reduce_result, tile_result);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <L>" << std::endl;
        exit(1);
    }

    int L = atoi(argv[1]);

    if (L % BLOCK_SIZE != 0) {
        std::cerr << "The size of the array must be multiple of BLOCK_SIZE" << std::endl;
        exit(1);
    }

    cudaSetDevice(0); // set the working device
    int *h_array = (int *) malloc(L * sizeof(int));
    for (int i=0; i<L; i++) {
        h_array[i] = i+1;
    }
    int *h_reduce_result = (int *) malloc(sizeof(int));
    *h_reduce_result = 0;

    int *d_array;
    int *d_reduce_result;
    gpuErrchk(cudaMalloc((void**) &d_array, L * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_reduce_result, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_array, h_array, L * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_reduce_result, h_reduce_result, sizeof(int), cudaMemcpyHostToDevice));

    unsigned int numBlocks = L/BLOCK_SIZE;
    call_reduce<<<numBlocks, BLOCK_SIZE>>>(d_array, d_reduce_result); // launch the kernel
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize(); // wait for the kernel completion

    gpuErrchk(cudaMemcpy(h_reduce_result, d_reduce_result, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Risultato reduce: %d\n", *h_reduce_result);

    return 0;
}
