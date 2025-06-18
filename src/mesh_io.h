#ifndef MESH_IO
#define MESH_IO

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <ios>
#include <ostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cmath>

#include "cuda_utils.cuh"
#include "vertex.h"

bool ImportMesh(const std::string filename, 
                std::vector<uint32_t>& facesCoords,
                std::vector<uint32_t>& facesNormals,
                std::vector<Vertex>& coordinates,
                std::vector<Normal>& normals,
                std::vector<Color>& colors) 
{
    std::string ext = std::filesystem::path(filename).extension().string();
    if (ext != ".obj" && ext != ".OBJ") {
        LOG_ERROR("%s is a wrong file extension. It must be .obj or .OBJ", ext.c_str());
        return false;
    }
    

    std::ifstream file(filename);
    if(!file.is_open()) {
        LOG_ERROR("Error to open file %s", filename.c_str());
        return false;
    }

    LOG_INFO("%s loaded sucessfully", filename.c_str());
    std::string line;
    std::string X, Y, Z, nX, nY, nZ, r, g, b;

    facesCoords.clear();
    facesNormals.clear();
    coordinates.clear();
    normals.clear();
    colors.clear();

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ss.imbue(std::locale::classic()); 

        std::string prefix;
        ss >> prefix;

        if(prefix == "#") {
            int count;      
            if (sscanf(line.c_str(), "# Vertices: %d", &count) == 1) {
                coordinates.reserve(count);
                normals.reserve(count);
                colors.reserve(count);
            } else if (sscanf(line.c_str(), "# Faces: %d", &count) == 1) {
                facesCoords.reserve(count * 3);
                facesNormals.reserve(count * 3);
            }
        } else if (prefix == "vn") {
            ss >> nX >> nY >> nZ;
            normals.emplace_back(std::stof(nX), std::stof(nY), std::stof(nZ));
        } else if (prefix == "v") {
            ss >> X >> Y >> Z;
            coordinates.emplace_back(std::stof(X), std::stof(Y), std::stof(Z));

            bool hasRGB = bool(ss >> r >> g >> b);
            if(hasRGB)
                colors.emplace_back(std::stof(r), std::stof(g), std::stof(g), 1.0f);
        } else if (prefix == "f") {
            uint32_t pos_inx, norm_idx;
            std::string vertex_str;

            for (uint32_t i = 0; i < 3; ++i) {
                ss >> vertex_str;
                sscanf(vertex_str.c_str(), " %d//%d", &pos_inx, &norm_idx);

                if(pos_inx != norm_idx)
                    LOG_WARN("Face coords and face norms are not equals");

                facesCoords.push_back(pos_inx - 1);
                facesNormals.push_back(norm_idx - 1);
            }
        }
    }
    
    LOG_INFO("Mesh sucessfully imported");
    facesCoords.shrink_to_fit();
    facesNormals.shrink_to_fit();
    coordinates.shrink_to_fit();
    normals.shrink_to_fit();
    colors.shrink_to_fit();
    return true;
}


bool ExportMesh(const std::string filename, 
                std::vector<uint32_t>& facesCoords,
                std::vector<uint32_t>& facesNormals,
                std::vector<Vertex>& coordinates,
                std::vector<Normal>& normals,
                std::vector<Color>& colors)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG_ERROR("Error to create or open %s file", filename.c_str());
        return false;
    }

    file << std::fixed << std::setprecision(6);
    file << "# OBJ file exporter by Matteo Giuntoni custom exporter\n";
    file << "# Vertices: " << coordinates.size() << "\n";
    file << "# Faces: " << facesCoords.size() / 3 << "\n";


    for (size_t i = 0; i < coordinates.size(); ++i) {
        float r = static_cast<float>(colors[i].R()) / 255.0f;
        float g = static_cast<float>(colors[i].G()) / 255.0f;
        float b = static_cast<float>(colors[i].B()) / 255.0f;
        
        file << "v " << coordinates[i].X << " " << coordinates[i].Y << " " << coordinates[i].Z << " " << r << " " << g << " " << b << "\n";
    }
    LOG_INFO("Coordinates are loaded");
    file << "\n";

    for (size_t i = 0; i < normals.size(); ++i) {
        file << "vn " << normals[i].X << " " << normals[i].Y << " " << normals[i].Z;
    }

    LOG_INFO("Normals are loades");
    file << "\n";

    for (size_t i = 0; i < facesCoords.size(); i += 3) {
        uint32_t i1 = facesCoords[i] + 1;
        uint32_t i2 = facesCoords[i + 1] + 1;
        uint32_t i3 = facesCoords[i + 2] + 1;

        uint32_t in1 = facesNormals[i] + 1;
        uint32_t in2 = facesNormals[i + 1] + 1;
        uint32_t in3 = facesNormals[i + 2] + 1;
 
        file << "f " << i1 << "//" << in1 << " " << i2 << "//" << in2 << " " << i3 << "//" << in3 << "\n";
    }

    LOG_INFO("Faces are loaded. Mesh is exported");
    return true;
}
#endif // MESH_IO
