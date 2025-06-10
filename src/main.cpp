#include <cstdint>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "cuda_utils.cuh"
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

    return 0;
}
