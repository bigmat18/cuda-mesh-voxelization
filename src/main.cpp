#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
#include <iostream>
#include <ostream>
#include <span>
#include <string>
#include <sys/types.h>
#include <vector>
#include <random>

#include "cuda_utils.cuh"
#include "mesh_io.h"
#include "bounding_box.h"
#include "voxels_grid.h"
#include "voxels_to_mesh.h"

#include <cuda_runtime.h>
#include <vector_types.h>

bool random_bool() {
    static std::mt19937 engine(std::random_device{}());
    static std::bernoulli_distribution distribution(0.5);
    return distribution(engine);
}

__global__ void test(int n, uint32_t* faces, Vertex* vertices, VoxelsGrid32bit grid)
{

    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n) {
        Vertex V0 = vertices[faces[(index * 3)]];
        Vertex V1 = vertices[faces[(index * 3) + 1]];
        Vertex V2 = vertices[faces[(index * 3) + 2]];
        Vertex facesVertices[3] = {V0, V1, V2};
        std::pair<float, float> BB_X, BB_Y, BB_Z;
        CalculateBoundingBox(std::span<Vertex>(&facesVertices[0], 3), BB_X, BB_Y, BB_Z);

        for(float y = BB_Y.first, i = 0; y < BB_Y.second; y += grid.VoxelSize(), ++i)
        {
            for(float z = BB_Z.first, j = 0; z < BB_Z.second; z += grid.VoxelSize(), ++j)
            {
                int voxelY = static_cast<int>((y - grid.OriginY()) / grid.VoxelSize());
                int voxelZ = static_cast<int>((z - grid.OriginZ()) / grid.VoxelSize());
                //LOG_INFO("\nstartY: %f, endY: %f\n startZ: %f, endZ: %f\n(%f - %f) / %f = %d), ((%f - %f) / %f = %d)\n", BB_Y.first, BB_Y.second, BB_Z.first, BB_Z.second, y, grid.OriginY(), grid.VoxelSize(), voxelY, z, grid.OriginZ(), grid.VoxelSize(), voxelZ);
                grid(0, voxelY, voxelZ) = true;
            }
        }
    }
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
    const float sideLength = CalculateBoundingBox(std::span<Vertex>(&vertices[0], vertices.size()), X, Y, Z);

    int device = 0;
    cudaSetDevice(device);

    int maxBlockDimX, maxBlockDimY, maxBlockDimZ;
    cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, device);
    cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, device);
    cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, device);
   
    uint32_t* devFaces;
    gpuAssert(cudaMalloc((void**) &devFaces, faces.size() * sizeof(uint32_t)));
    gpuAssert(cudaMemcpy(devFaces, &faces[0], faces.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    Vertex* devVertices;
    gpuAssert(cudaMalloc((void**) &devVertices, vertices.size() * sizeof(Vertex)));
    gpuAssert(cudaMemcpy(devVertices, &vertices[0], vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice));

    const size_t voxelsPerSide = 64;
    DeviceVoxelsGrid32bit devGrid(voxelsPerSide, sideLength);
    devGrid.View().SetOrigin(X.first, Y.first, Z.first);

    int n = faces.size() / 3;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    test<<< gridSize, blockSize >>>(n, devFaces, devVertices, devGrid.View());

    gpuAssert(cudaPeekAtLastError());
    cudaDeviceSynchronize(); 

    HostVoxelsGrid32bit hostGrid(devGrid);

    #ifdef DEBUG
    for (int i = 0; i < hostGrid.View().VoxelsPerSide(); ++i) {
        for (int j = 0; j < hostGrid.View().VoxelsPerSide(); ++j) {
            for (int k = 0; k < hostGrid.View().VoxelsPerSide(); ++k) {
                std::cout << hostGrid.View()(k, j, i) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    #endif // DEBUG
    
    VoxelsGridToMesh(hostGrid.View(), faces, vertices);
    if(!ExportMesh("assets/test.obj", faces, vertices)) {
        LOG_ERROR("Error in mesh export");
        return -1;
    }


    return 0;
}
