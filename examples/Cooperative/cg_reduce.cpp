/* Per-block and per-warp reduce using cooperative groups */

#include<iostream>
#include<cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define BLOCK_SIZE 256

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__device__ int reduce(cg::thread_group g, int *smem, int val)
{
    int id = g.thread_rank();
    for (int i=g.size()/2; i>0; i /=2) {
        smem[id] = val;
        g.sync();
        if (id < i) {
            val += smem[id + i];
        }
        g.sync();
    }
    return val;
}

__global__ void call_reduce(int *array, int *reduce_result)
{
    __shared__ int array_smem[BLOCK_SIZE];
    int myval = array[(blockIdx.x * blockDim.x) + threadIdx.x];
    array[(blockIdx.x * blockDim.x) + threadIdx.x] = 0;
    cg::thread_block g = cg::this_thread_block();
#if defined (BLOCK_BASIS)
    int result = reduce(g, array_smem, myval);
    if (g.thread_rank() == 0) { // g.thread_rank() == threadIdx.x
        atomicAdd(reduce_result, result);
        ;
    }
#endif
#if defined (WARP_BASIS)
    int tileIdx = g.thread_rank() / 32; // g.thread_rank() == threadIdx.x
    int *ptr = array_smem + (tileIdx * 32);
    auto tile = cg::tiled_partition(this_thread_block(), 32);
    int tile_result = reduce(tile, ptr, myval);
    if (tile.thread_rank() == 0) {
        atomicAdd(reduce_result, tile_result);
    }
#endif
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
