#include <cstdint>
#include <mesh/mesh_io.h>
#include <mesh/voxels_to_mesh.h>
#include <bounding_box.h>
#include <string>
#include <voxels_grid.h>
#include <voxelization/voxelization.h>
#include <profiling.h>

#define EXPORT 1
#define ACTIVE_SEQUENTIAL 1
#define ACTIVE_NAIVE 1
#define ACTIVE_TILED 1

int main(int argc, char **argv) {
    int device = 0; cudaSetDevice(device);

    cpuAssert((argc >= 4), "Need two mesh parameter [input file] [output file] [voxel per side]\n");

    Mesh mesh; 
    cpuAssert(ImportMesh(argv[1], mesh), "Error in mesh import");

    std::pair<float, float> bbX, bbY, bbZ;
    float sideLength = 0.0f;
    sideLength = CalculateBoundingBox(
        std::span<Position>(&mesh.Coords[0], mesh.Coords.size()), 
        bbX, bbY, bbZ 
    );

    const size_t voxelsPerSide = atoi(argv[3]);

    #if ACTIVE_SEQUENTIAL 
    {
        HostVoxelsGrid32bit hostGrid(voxelsPerSide, sideLength);
        hostGrid.View().SetOrigin(bbX.first, bbY.first, bbZ.first);

        Voxelization::Compute<Voxelization::Types::SEQUENTIAL>(
            hostGrid, mesh
        );   

        #if EXPORT
        Mesh outMesh;
        VoxelsGridToMesh(hostGrid.View(), outMesh);
        if(!ExportMesh("out/sequential_" + std::string(argv[2]), outMesh)) {
            LOG_ERROR("Error in sequential mesh export");
            return -1;
        }
        #endif
    } 
    #endif 

    #if ACTIVE_NAIVE
    {
        DeviceVoxelsGrid32bit devGrid(voxelsPerSide, sideLength);
        devGrid.View().SetOrigin(bbX.first, bbY.first, bbZ.first);

        Voxelization::Compute<Voxelization::Types::NAIVE, uint32_t>(
            devGrid, mesh, device 
        );

        #if EXPORT
        HostVoxelsGrid32bit hostGrid(devGrid);   
        Mesh outMesh;
        VoxelsGridToMesh(hostGrid.View(), outMesh);
        if(!ExportMesh("out/naive_" + std::string(argv[2]), outMesh)) {
            LOG_ERROR("Error in naive mesh export");
            return -1;
        }
        #endif
    }
    #endif

    #if ACTIVE_TILED
    {
        DeviceVoxelsGrid32bit devGrid(voxelsPerSide, sideLength);
        devGrid.View().SetOrigin(bbX.first, bbY.first, bbZ.first);

        Voxelization::Compute<Voxelization::Types::TILED, uint32_t>(
            devGrid, mesh, device
        );

        #if EXPORT
        Mesh outMesh;
        HostVoxelsGrid32bit hostGrid(devGrid);   
        VoxelsGridToMesh(hostGrid.View(), outMesh);
        if(!ExportMesh("out/tiled_" + std::string(argv[2]), outMesh)) {
            LOG_ERROR("Error in tiled mesh export");
            return -1;
        }
        #endif
    }
    #endif

    return 0;
}
