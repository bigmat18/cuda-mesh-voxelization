#include <cstdint>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include "cuda_utils.cuh"
#include "voxels_grid.h"
#include "mesh_io.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Need a mesh parameter\n");
        exit(0);
    }

    std::vector<uint32_t> faces;
    std::vector<Vertex> vertices;
    if(!ImportMesh(std::string(argv[1]), faces, vertices)) {
        std::cout << "Error in mesh loading" << std::endl;
        return -1;
    }

    if(!ExportMesh("assets/my_torus.obj", faces, vertices)) {
        std::cout << "Error in mesh saving" << std::endl;
        return -1;
    }

    VoxelsGrid8bit v(2);
    v(4,1,0) = true;

    for(int i = 0 ; i < v.SideSize() ; ++i) {
        for(int j = 0 ; j < v.SideSize() ; ++j) {
            for(int k = 0 ; k < v.SideSize() ; ++k) {
                LOG_INFO("%d", static_cast<bool>(v(i, j, k)));
            }
        }
    }
    return 0;
}
