#include <cstddef>
#include <cstdint>
#include <mesh/mesh_io.h>
#include <mesh/grid_to_mesh.h>
#include <bounding_box.h>
#include <string>
#include <vector>
#include <voxels_grid.h>
#include <voxelization/voxelization.cuh>
#include <profiling.h>

#define EXPORT 0
#define ACTIVE_SEQUENTIAL 1
#define ACTIVE_NAIVE 1
#define ACTIVE_TILED 1

std::string getFilename(const std::string path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) return path.substr(pos + 1);
    else                          return path;
}

int main(int argc, char **argv) {
    int device = 0; cudaSetDevice(device);

    cpuAssert((argc >= 1), "Need [input file]\n");

    size_t voxelsPerSide = 32;
    std::string outFileName = "out";
    std::string opType = "null";  
    std::vector<Mesh> meshes;


    for(int i = 1; i < argc; ++i) { 
        std::string s(argv[i]); 
        if(s == "-o") {  
            cpuAssert(i+1 < argc, "Missing -o [output file]"); 
            outFileName = argv[i+1];
            i++;
        } else if(s == "-op") { 
            cpuAssert(i+1 < argc, "Missing -o [output file]"); 
            opType = argv[i+1]; 
            cpuAssert(opType != "union" || opType != "inter" || opType != "diff" || opType != "null",
                      "Wrong -op param. It can be: 'union', 'inter', 'diff', 'null'");
            i++;
        } else if(s == "-n") {
            cpuAssert(i+1 < argc, "Missing -n [voxel per side]"); 
            voxelsPerSide = atoi(argv[i+1]);
            i++;
        } else {
            meshes.push_back(Mesh(getFilename(argv[i])));
            cpuAssert(ImportMesh(argv[i], meshes[meshes.size() - 1]), 
                      "Error in mesh import named %s"); 
        }
    }

    for(auto& mesh : meshes) {

        printf("============== %s =============\n", mesh.Name.c_str());
        std::pair<float, float> bbX, bbY, bbZ;
        float sideLength = 0.0f;
        sideLength = CalculateBoundingBox(
            std::span<Position>(&mesh.Coords[0], mesh.Coords.size()), 
            bbX, bbY, bbZ 
        );

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
            if(!ExportMesh("out/sequential_" + mesh.Name, outMesh)) {
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
            if(!ExportMesh("out/naive_" + mesh.Name, outMesh)) {
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
            if(!ExportMesh("out/tiled_" + mesh.Name, outMesh)) {
                LOG_ERROR("Error in tiled mesh export");
                return -1;
            }
            #endif
        }
        #endif
    }

    return 0;
}
