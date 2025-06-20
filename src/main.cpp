#include <cstdint>
#include <mesh/mesh_io.h>
#include <mesh/voxels_to_mesh.h>
#include <bounding_box.h>
#include <voxels_grid.h>
#include <voxelization/voxelization.h>


int main(int argc, char **argv) {
    cpuAssert((argc >= 3), "Need two mesh parameter [input file] [output file]\n");

    Mesh mesh;
    cpuAssert(ImportMesh(argv[1], mesh), "Error in mesh import");

    std::pair<float, float> bbX, bbY, bbZ;
    const float sideLength = CalculateBoundingBox(
        std::span<Position>(&mesh.Coords[0], mesh.Coords.size()), 
        bbX, bbY, bbZ 
    );

    const size_t voxelsPerSide = 128;
    DeviceVoxelsGrid32bit devGrid(voxelsPerSide, sideLength);
    devGrid.View().SetOrigin(bbX.first, bbY.first, bbZ.first);

    int device = 0;
    cudaSetDevice(device);

    Voxelization::Compute<Voxelization::Types::NAIVE, uint32_t>(
        devGrid, mesh, device, 256
    );

    HostVoxelsGrid32bit hostGrid(devGrid);

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
   
    VoxelsGridToMesh(hostGrid.View(), mesh);

    if(!ExportMesh(argv[2], mesh)) {
        LOG_ERROR("Error in mesh export");
        return -1;
    }


    return 0;
}
