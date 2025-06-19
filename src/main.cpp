#include <mesh/mesh_io.h>
#include <mesh/voxels_to_mesh.h>
#include <voxels_grid.h>

/*__device__ std::tuple<float, float, float> CalculateEdgeTerms(Vertex& V0, Vertex& V1)*/
/*{*/

    /*float A = V0.Z - V1.Z;*/
    /*float B = V1.Y - V0.Y;*/
    /*float C = -A * V0.Y - B * V0.Z;    */

    /*return {A, B, C};*/
/*}*/

/*__device__ inline float CalculateEdgeFunction(float A, float B, float C, float y, float z)*/
/*{*/
    /*return (A * y) + (B * z) + C;*/
/*}*/

/*__global__ void test(int n, uint32_t* facesCoords, Vertex* coords, VoxelsGrid32bitDev grid)*/
/*{*/

    /*int index = (blockIdx.x * blockDim.x) + threadIdx.x;*/
    /*if (index >= n)*/
        /*return;*/

    /*Vertex V0 = coords[facesCoords[(index * 3)]];*/
    /*Vertex V1 = coords[facesCoords[(index * 3) + 1]];*/
    /*Vertex V2 = coords[facesCoords[(index * 3) + 2]];*/
        
    /*auto [A0, B0, C0] = CalculateEdgeTerms(V0, V1);*/
    /*auto [A1, B1, C1] = CalculateEdgeTerms(V1, V2);*/
    /*auto [A2, B2, C2] = CalculateEdgeTerms(V2, V0);*/

    /*Vertex facesVertices[3] = {V0, V1, V2};*/
    /*std::pair<float, float> BB_X, BB_Y, BB_Z;*/
    /*CalculateBoundingBox(std::span<Vertex>(&facesVertices[0], 3), BB_X, BB_Y, BB_Z);*/

    /*int startY = static_cast<int>(std::floorf((BB_Y.first - grid.OriginY()) / grid.VoxelSize()));*/
    /*int endY   = static_cast<int>(std::ceilf((BB_Y.second - grid.OriginY()) / grid.VoxelSize()));*/
    /*int startZ = static_cast<int>(std::floorf((BB_Z.first - grid.OriginZ()) / grid.VoxelSize()));*/
    /*int endZ   = static_cast<int>(std::ceilf((BB_Z.second - grid.OriginZ()) / grid.VoxelSize()));*/

    /*Vertex edge0 = V1 - V0;*/
    /*Vertex edge1 = V2 - V0;*/
    /*auto [A, B, C] = Vertex::Cross(edge0, edge1);*/
    /*float D = Vertex::Dot({A, B, C}, V0);*/

    /*for(int y = startY; y < endY; ++y)*/
    /*{*/
        /*for(int z = startZ; z < endZ; ++z)*/
        /*{*/
            /*float centerY = grid.OriginY() + ((y * grid.VoxelSize()) + (grid.VoxelSize() / 2));*/
            /*float centerZ = grid.OriginZ() + ((z * grid.VoxelSize()) + (grid.VoxelSize() / 2));*/

            /*float E0 = CalculateEdgeFunction(A0, B0, C0, centerY, centerZ);*/
            /*float E1 = CalculateEdgeFunction(A1, B1, C1, centerY, centerZ);*/
            /*float E2 = CalculateEdgeFunction(A2, B2, C2, centerY, centerZ);*/

            /*bool ccw_test = (E0 >= 0 && E1 >= 0 && E2 >= 0);*/
            /*bool cw_test = (E0 <= 0 && E1 <= 0 && E2 <= 0);*/
            
            /*if (ccw_test || cw_test) {*/
                /*float intersection = (D - (B * centerY) - (C * centerZ)) / A;*/

                /*int startX = static_cast<int>((intersection - grid.OriginX()) / grid.VoxelSize());*/
                /*int endX = grid.VoxelsPerSide();*/
                /*for(int x = startX; x < endX; ++x)*/
                    /*grid(x, y, z) ^= true;*/
            /*}*/
        /*}*/
    /*}*/
/*}*/

int main(int argc, char **argv) {
    cpuAssert((argc >= 3), "Need two mesh parameter [input file] [output file]\n");

    Mesh mesh;
    cpuAssert(ImportMesh(argv[1], mesh), "Error in mesh import");

    HostVoxelsGrid32bit grid(8, 10);
    grid.View()(0,0,0) = true;

    VoxelsGridToMesh(grid.View(), mesh);

/*    std::pair<float, float> bbX, bbY, bbZ;*/
    /*const float sideLength = CalculateBoundingBox(*/
        /*std::span<Vertex>(&coords[0], coords.size()), */
        /*bbX, bbY, bbZ*/
    /*);*/

    /*int device = 0;*/
    /*cudaSetDevice(device);*/

    /*int maxBlockDimX, maxBlockDimY, maxBlockDimZ;*/
    /*cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, device);*/
    /*cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, device);*/
    /*cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, device);*/
   
    /*uint32_t* devFaces;*/
    /*gpuAssert(cudaMalloc((void**) &devFaces, facesCoords.size() * sizeof(uint32_t)));*/
    /*gpuAssert(cudaMemcpy(devFaces, &facesCoords[0], facesCoords.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));*/

    /*Vertex* devVertices;*/
    /*gpuAssert(cudaMalloc((void**) &devVertices, coords.size() * sizeof(Vertex)));*/
    /*gpuAssert(cudaMemcpy(devVertices, &coords[0], coords.size() * sizeof(Vertex), cudaMemcpyHostToDevice));*/

    /*const size_t voxelsPerSide = 128;*/
    /*DeviceVoxelsGrid32bit devGrid(voxelsPerSide, sideLength);*/
    /*devGrid.View().SetOrigin(bbX.first, bbY.first, bbZ.first);*/

    /*int n = facesCoords.size() / 3;*/
    /*int blockSize = 256;*/
    /*int gridSize = (n + blockSize - 1) / blockSize;*/
    
    /*test<<< gridSize, blockSize >>>(n, devFaces, devVertices, devGrid.View());*/

    /*gpuAssert(cudaPeekAtLastError());*/
    /*cudaDeviceSynchronize(); */

    /*HostVoxelsGrid32bit hostGrid(devGrid);*/

    /*#ifdef DEBUG*/
    /*for (int i = 0; i < hostGrid.View().VoxelsPerSide(); ++i) {*/
        /*for (int j = 0; j < hostGrid.View().VoxelsPerSide(); ++j) {*/
            /*for (int k = 0; k < hostGrid.View().VoxelsPerSide(); ++k) {*/
                /*std::cout << hostGrid.View()(k, j, i) << " ";*/
            /*}*/
            /*std::cout << std::endl;*/
        /*}*/
        /*std::cout << std::endl;*/
    /*}*/
    /*#endif // DEBUG*/
   

    /*VoxelsGridToMesh(hostGrid.View(), facesCoords, facesNormals, coords, normals, colors);*/

    if(!ExportMesh(argv[2], mesh)) {
        LOG_ERROR("Error in mesh export");
        return -1;
    }


    return 0;
}
