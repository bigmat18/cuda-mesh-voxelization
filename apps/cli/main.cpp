#include <string>
#include <cstddef>

#include <mesh/grid_to_mesh.h>
#include <mesh/mesh_io.h>
#include <mesh/mesh.h>

#include <voxelization/voxelization.cuh>
#include <csg.cuh>
#include <jfa.cuh>

#include <bounding_box.h>
#include <profiling.h>
#include <voxels_grid.h>

#define EXPORT_STEPS 1
#define ACTIVE_SEQUENTIAL 1
#define ACTIVE_NAIVE 1
#define ACTIVE_TILED 1

std::string getFilename(const std::string path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos)
        return path.substr(pos + 1);
    else
        return path;
}

int main(int argc, char **argv) {
    int device = 0;
    cudaSetDevice(device);

    cpuAssert((argc >= 1), "Need [input file]\n");

    size_t voxelsPerSide = 32;
    std::string outFileName = "out.obj";
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

    std::vector<std::unique_ptr<DeviceVoxelsGrid32bit>> grids;

    size_t size = 0;
    std::vector<Position> allCoords;
    for(const auto& mesh : meshes)
        size += mesh.Coords.size();

    allCoords.reserve(size);

    for(const auto& mesh : meshes)
        allCoords.insert(allCoords.end(), mesh.Coords.begin(), mesh.Coords.end());

    std::pair<float, float> bbX, bbY, bbZ;
    float sideLength = CalculateBoundingBox(
        std::span<Position>(&allCoords[0], allCoords.size()),
        bbX, bbY, bbZ
    );

    for(int i = 0; i < meshes.size(); i++) {
        auto& mesh = meshes[i];

        printf("============== %s =============\n", mesh.Name.c_str());

        if(opType != "null") {
            grids.emplace_back(std::make_unique<DeviceVoxelsGrid32bit>(voxelsPerSide, sideLength)); 
            grids[i]->View().SetOrigin(bbX.first, bbY.first, bbZ.first);

            Voxelization::Compute<Voxelization::Types::TILED, uint32_t>(*grids[i], mesh, device);
        }

    #if ACTIVE_SEQUENTIAL
    {
        HostVoxelsGrid32bit hostGrid(voxelsPerSide, sideLength);
        hostGrid.View().SetOrigin(bbX.first, bbY.first, bbZ.first);

        Voxelization::Compute<Voxelization::Types::SEQUENTIAL>(
            hostGrid, mesh
        );

        #if EXPORT_STEPS
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

        #if EXPORT_STEPS
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

        #if EXPORT_STEPS
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
    // int size = 32;
    // HostVoxelsGrid32bit test(size, 1);
    // test.View().Voxel(2, 2, 0) =true;
    // test.View().Voxel(3, 2, 0) = true;
    // test.View().Voxel(4, 2, 0) = true;
    // test.View().Voxel(2, 3, 0) = true;
    // test.View().Voxel(3, 3, 0) = true;
    // test.View().Voxel(4, 3, 0) = true;
    // test.View().Voxel(2, 4, 0) = true;
    // test.View().Voxel(3, 4, 0) = true;
    // test.View().Voxel(4, 4, 0) = true;

    // test.View().Voxel(2, 2, 1) = true;
    // test.View().Voxel(3, 2, 1) = true;
    // test.View().Voxel(4, 2, 1) = true;
    // test.View().Voxel(2, 3, 1) = true;
    // test.View().Voxel(3, 3, 1) = true;
    // test.View().Voxel(4, 3, 1) = true;
    // test.View().Voxel(2, 4, 1) = true;
    // test.View().Voxel(3, 4, 1) = true;
    // test.View().Voxel(4, 4, 1) = true;

    // test.View().Voxel(2, 2, 2) = true;
    // test.View().Voxel(3, 2, 2) = true;
    // test.View().Voxel(4, 2, 2) = true;
    // test.View().Voxel(2, 3, 2) = true;
    // test.View().Voxel(3, 3, 2) = true;
    // test.View().Voxel(4, 3, 2) = true;
    // test.View().Voxel(2, 4, 2) = true;
    // test.View().Voxel(3, 4, 2) = true;
    // test.View().Voxel(4, 4, 2) = true;

    // //test.View().Print();

    // DeviceVoxelsGrid32bit devTest(test);

    // std::vector<JFA::SDF> sdfValues(size * size * size);
    // JFA::Compute<JFA::Types::NAIVE, uint32_t>(devTest, sdfValues);

    // HostVoxelsGrid32bit out(devTest);

    // for (int z = 0; z < out.View().VoxelsPerSide(); ++z) {
    //   for (int y = 0; y < out.View().VoxelsPerSide(); ++y) {
    //     for (int x = 0; x < out.View().VoxelsPerSide(); ++x) {
    //       printf("%.1f ", sdfValues[(z * size * size) + (y * size) + x].distance);
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    // }

    if (opType != "null") {
        printf("============== Start CSG =============\n");

        for (int i = 1; i < grids.size(); ++i) {
            if (opType == "union")
                CSG::Compute<uint32_t>(*grids[0], *grids[i], CSG::Union<uint32_t>());

            if (opType == "inter")
                CSG::Compute<uint32_t>(*grids[0], *grids[i], CSG::Intersection<uint32_t>());

            if (opType == "diff")   
                CSG::Compute<uint32_t>(*grids[0], *grids[i], CSG::Difference<uint32_t>());
        }

        Mesh outMesh;
        HostVoxelsGrid32bit hostGrid(*grids[0]);
        VoxelsGridToMesh(hostGrid.View(), outMesh);
        if(!ExportMesh("out/" + outFileName, outMesh)) {
            LOG_ERROR("Error in tiled mesh export");
            return -1;
        }
    }

    return 0;
}
