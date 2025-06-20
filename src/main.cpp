#include <cstdint>
#include <mesh/mesh_io.h>
#include <mesh/voxels_to_mesh.h>
#include <bounding_box.h>
#include <voxels_grid.h>
#include <voxelization/voxelization.h>
#include <profiling.h>


int main(int argc, char **argv) {
    int device = 0;
    cudaSetDevice(device);

    cpuAssert((argc >= 3), "Need two mesh parameter [input file] [output file]\n");

    Mesh mesh;
    cpuAssert(ImportMesh(argv[1], mesh), "Error in mesh import");

    std::pair<float, float> bbX, bbY, bbZ;
    float sideLength = 0.0f;

    sideLength = CalculateBoundingBox(
        std::span<Position>(&mesh.Coords[0], mesh.Coords.size()), 
        bbX, bbY, bbZ 
    );
    

    const size_t voxelsPerSide = 256;
    DeviceVoxelsGrid32bit devGrid(voxelsPerSide, sideLength);
    devGrid.View().SetOrigin(bbX.first, bbY.first, bbZ.first);

    HostVoxelsGrid32bit hostGrid(devGrid);
    Voxelization::Compute<Voxelization::Types::SEQUENTIAL>(
        hostGrid, mesh
    );

    Voxelization::Compute<Voxelization::Types::NAIVE, uint32_t>(
        devGrid, mesh, device, 256
    );

    //HostVoxelsGrid32bit hostGrid(devGrid);   
    //VoxelsGridToMesh(hostGrid.View(), mesh);
    //if(!ExportMesh(argv[2], mesh)) {
        //LOG_ERROR("Error in mesh export");
        //return -1;
    //}


    return 0;
}
