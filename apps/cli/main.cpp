#include "proc_utils.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mesh/mesh.h>
#include <mesh/mesh_io.h>
#include <mesh/grid_to_mesh.h>
#include <debug_utils.h>
#include <csg/csg.h>
#include <vox/vox.h>
#include <jfa/jfa.h>
#include <grid/voxels_grid.h>
#include <bounding_box.h>

#include <cxxopts.hpp>

int main(int argc, char **argv) {
    const int device = 0;
    cudaSetDevice(device);
    cpuAssert((argc >= 2), "Need [input file]\n");

    cxxopts::Options options("cli", "CLI apps to test csg voxelization");

    options.add_options()
        ("i,filenames", "Input filenames list", cxxopts::value<std::vector<std::string>>())
        ("n,num-voxels", "Number of voxel per side", cxxopts::value<unsigned int>()->default_value("32"))
        ("t,type", "Type of processing (0 = sequential, 1 = naive, 2 = tiled)", cxxopts::value<int>()->default_value("2"))
        ("o,output", "Output filename", cxxopts::value<std::string>()->default_value("out.obj"))
        ("p,operation", "CSG Operations (1 = union, 2 = inter, 3 = diff)", cxxopts::value<int>()->default_value("0"))
        ("e,export", "Exports the phases", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");

    options.parse_positional({"filenames"});
    auto result = options.parse(argc, argv);

    if(result.count("help")) {
        printf("%s", options.help().c_str()); 
        return 0;
    }

    cpuAssert(result.count("filenames") >= 1, "Need [input filename]");

    const std::vector<std::string>  FILENAMES    = result["filenames"].as<std::vector<std::string>>();
    const std::string               OUT_FILENAME = result["output"].as<std::string>();
    const CSG::Op                   OPERATION    = static_cast<CSG::Op>(result["operation"].as<int>());
    const Types                     TYPE         = static_cast<Types>(result["type"].as<int>());
    const unsigned int              NUM_VOXELS   = result["num-voxels"].as<unsigned int>();
    const bool                      EXPORT       = result["export"].as<bool>();

    std::vector<Mesh> meshes(result.count("filenames"));
    std::vector<HostVoxelsGrid32bit> grids(result.count("filenames"));
    float originX, originY, originZ, voxelSize;
    {
        std::vector<Position> coords;
        int fullSize = 0;
        for(int i = 0; i < meshes.size(); i++) {
            cpuAssert(ImportMesh(FILENAMES[i], meshes[i]), "Error in " + FILENAMES[i] + " import");
            fullSize += meshes[i].Coords.size();
        }
        coords.reserve(fullSize);
        for(const auto& mesh : meshes)
            coords.insert(coords.end(), mesh.Coords.begin(), mesh.Coords.end());

        std::pair<float, float> bbX, bbY, bbZ;
        float sideLength = CalculateBoundingBox(
            std::span<Position>(&coords[0], coords.size()),
            bbX, bbY, bbZ
        );
        
        originX = bbX.first;
        originY = bbY.first;
        originZ = bbZ.first;
        voxelSize = sideLength / NUM_VOXELS;
    }

    for(int i = 0; i < meshes.size(); i++) {
        auto& mesh = meshes[i];
        auto& grid = grids[i];


        if (TYPE == Types::SEQUENTIAL) {
            grid = HostVoxelsGrid32bit(NUM_VOXELS, voxelSize);
            grid.View().SetOrigin(originX, originY, originZ);
            VOX::Compute<Types::SEQUENTIAL>(grid, mesh);
        }
        
        else if (TYPE == Types::NAIVE) {
            DeviceVoxelsGrid32bit devGrid(NUM_VOXELS, voxelSize);
            devGrid.View().SetOrigin(originX, originY, originZ);
            VOX::Compute<Types::NAIVE>(devGrid, mesh);
            grid = HostVoxelsGrid32bit(devGrid);
        }

        
        else if (TYPE == Types::TILED) {
            DeviceVoxelsGrid32bit devGrid(NUM_VOXELS, voxelSize);
            devGrid.View().SetOrigin(originX, originY, originZ);
            VOX::Compute<Types::TILED>(devGrid, mesh);
            grid = HostVoxelsGrid32bit(devGrid);
        }

        if (EXPORT) {
            Mesh outMesh;
            VoxelsGridToMesh(grid.View(), outMesh);
            cpuAssert(ExportMesh("out/" + GetTypesString(TYPE) + "_" + GetFilename(FILENAMES[i]), outMesh), 
                      "Error in " + GetTypesString(TYPE) + " " + FILENAMES[i] + " export");
        }


        if (i > 0 && OPERATION != CSG::Op::VOID) {

            if (TYPE == Types::SEQUENTIAL) {

            }

            else if (TYPE == Types::NAIVE || TYPE == Types::TILED) {
                DeviceVoxelsGrid32bit devGridFirst(grids[0]);
                DeviceVoxelsGrid32bit devGridSecond(grid);

                switch (OPERATION) {
                    case CSG::Op::UNION: 
                        CSG::Compute<Types::NAIVE>(devGridFirst, devGridSecond, CSG::Union<uint32_t>());   
                        break;

                    case CSG::Op::DIFFERENCE: 
                        CSG::Compute<Types::NAIVE>(devGridFirst, devGridSecond, CSG::Difference<uint32_t>());   
                        break;

                    case CSG::Op::INTERSECTION: 
                        CSG::Compute<Types::NAIVE>(devGridFirst, devGridSecond, CSG::Intersection<uint32_t>());   
                        break;

                    case CSG::Op::VOID: 
                        break;
                }

                grids[0] = HostVoxelsGrid32bit(devGridFirst);
            }
        }
    }


    if (EXPORT && OPERATION != CSG::Op::VOID)  {
        Mesh outMesh;
        VoxelsGridToMesh(grids[0].View(), outMesh);
        cpuAssert(ExportMesh("out/csg_vox_" + GetTypesString(TYPE) + "_" + OUT_FILENAME, outMesh), 
                  "Error in " + OUT_FILENAME + " export (csg)");
    }

    HostGrid<float> sdf(grids[0].View().VoxelsPerSide(), -std::numeric_limits<float>::infinity());

    if (TYPE == Types::SEQUENTIAL) {

    }

    else if (TYPE == Types::NAIVE) {
        DeviceVoxelsGrid32bit devGrid(grids[0]);
        DeviceGrid<float> devSDF(sdf);
        DeviceGrid<Position> devPositions(grids[0].View().VoxelsPerSide());

        JFA::Compute<Types::NAIVE>(devGrid, devSDF, devPositions);
        sdf = HostGrid(devSDF);
    }

    else if (TYPE == Types::TILED) {
        DeviceVoxelsGrid32bit devGrid(grids[0]);
        DeviceGrid<float> devSDF(sdf);
        DeviceGrid<Position> devPositions(grids[0].View().VoxelsPerSide());

        JFA::Compute<Types::TILED>(devGrid, devSDF, devPositions);
        sdf = HostGrid(devSDF);
    }

    if (EXPORT) {
        Mesh outMesh;
        VoxelsGridToMeshSDFColor(grids[0].View(), sdf.View(), outMesh);
        cpuAssert(ExportMesh("out/sdf_" + GetTypesString(TYPE) + "_" + OUT_FILENAME, outMesh), 
                  "Error in " + OUT_FILENAME + " export (sdf)");
    }

    return 0;
}
