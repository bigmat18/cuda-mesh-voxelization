#include <cstdint>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include "cuda_utils.cuh"
#include "voxels_grid.h"
#include "mesh_io.h"

#include <random>

bool random_bool() {
    static std::mt19937 engine(std::random_device{}());
    static std::bernoulli_distribution distribution(0.5);
    return distribution(engine);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Need a mesh parameter\n");
        exit(0);
    }

    std::vector<uint32_t> faces;
    std::vector<Vertex> vertices;
    /*if(!ImportMesh(std::string(argv[1]), faces, vertices)) {*/
        /*std::cout << "Error in mesh loading" << std::endl;*/
        /*return -1;*/
    /*}*/

    VoxelsGrid8bit grid(8);

    for(int i = 0 ; i < grid.SideSize() ; ++i) {
        for(int j = 0 ; j < grid.SideSize() ; ++j) {
            for(int k = 0 ; k < grid.SideSize() ; ++k) {
                grid(k,j,i) = random_bool();
            }
        }
    }

    VoxelsGridToMesh<uint8_t>(grid, faces, vertices);

    if(!ExportMesh("assets/test.obj", faces, vertices)) {
        std::cout << "Error in mesh saving" << std::endl;
        return -1;
    }


    return 0;
}
