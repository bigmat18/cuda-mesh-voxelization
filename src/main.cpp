#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <string>
#include <sys/types.h>
#include <vector>
#include <random>

#include "cuda_utils.cuh"
#include "mesh_io.h"
#include "bounding_box.h"
#include "voxels_grid.h"

#include <cuda_runtime.h>
#include <vector_types.h>

bool random_bool() {
    static std::mt19937 engine(std::random_device{}());
    static std::bernoulli_distribution distribution(0.5);
    return distribution(engine);
}

__global__ void test(int n, VoxelsGrid8bit grid)
{

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n)
        LOG_INFO("%d", index);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Need a mesh parameter\n");
        exit(0);
    }

    std::vector<uint32_t> faces;
    std::vector<Vertex> vertices;

    if (!ImportMesh(argv[1], faces, vertices)) {
        LOG_ERROR("Error in mesh import");
        return -1;
    }

    std::pair<float, float> X, Y, Z;
    const float sideLength = CalculateBoundingBox(vertices, X, Y, Z);
    const size_t side_size = 512;

    int device = 0;
    cudaSetDevice(device);

    int maxBlockDimX, maxBlockDimY, maxBlockDimZ;
    cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, device);
    cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, device);
    cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, device);
   
    DeviceVoxelsGrid8bit grid(side_size, sideLength / side_size);

    int n = faces.size();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    test<<< gridSize, blockSize >>>(n, grid.View());

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize(); 
   
    //if(!ExportMesh("assets/test.obj", faces, vertices)) {
        //LOG_ERROR("Error in mesh export");
        //return -1;
    //}


    return 0;
}
