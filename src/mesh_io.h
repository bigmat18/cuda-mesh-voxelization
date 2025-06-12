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

struct Vertex {
  float X, Y, Z;
  float nX, nY, nZ;
  uint32_t color;

  /* Set color from float to uint32_t format */
  inline void SetColor(float r, float g, float b, float a) {
    color = static_cast<uint32_t>(
                    (static_cast<uint8_t>(std::round(r * 255)) << 24) |
                    (static_cast<uint8_t>(std::round(g * 255)) << 16) |
                    (static_cast<uint8_t>(std::round(b * 255)) << 8)  | 
                    (static_cast<uint8_t>(std::round(a * 255)))
            );
  }

  inline uint8_t R() const { return (color >> 24) & 0xFF; }

  inline uint8_t G() const { return (color >> 16) & 0xFF; }

  inline uint8_t B() const { return (color >> 8) & 0xFF; }

  inline uint8_t A() const { return color & 0xFF; }

};


struct Faces {
    std::vector<uint32_t> faces;
    std::vector<Vertex> vertices;

    inline size_t Size() const 
    {
        return faces.size() / 3;
    }
};

bool ImportMesh(const std::string filename, std::vector<uint32_t> &faces, std::vector<Vertex> &vertices) 
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

    faces.clear();
    vertices.clear();

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ss.imbue(std::locale::classic()); 

        std::string prefix;
        ss >> prefix;

        if(prefix == "#") {
            int count;      
            if (sscanf(line.c_str(), "# Vertices: %d", &count) == 1) {
                vertices.reserve(count);
            } else if (sscanf(line.c_str(), "# Faces: %d", &count) == 1) {
                faces.reserve(count * 3);
            }
        } else if (prefix == "vn") {
            ss >> nX >> nY >> nZ;
        } else if (prefix == "v") {
            ss >> X >> Y >> Z >> r >> g >> b;

            Vertex v(std::stof(X), std::stof(Y), std::stof(Z), std::stof(nX), std::stof(nY), std::stof(nZ), 0);
            v.SetColor(std::stof(r), std::stof(g), std::stof(b), 1.0);
            vertices.push_back(v);
        } else if (prefix == "f") {
            uint32_t pos_inx, norm_idx;
            std::string vertex_str;

            for (uint32_t i = 0; i < 3; ++i) {
                ss >> vertex_str;
                sscanf(vertex_str.c_str(), " %d//%d", &pos_inx, &norm_idx);

                if(pos_inx != norm_idx) {
                    return false;
                }
                faces.push_back(pos_inx - 1);
            }
        }
    }
    
    LOG_INFO("Mesh sucessfully imported");
    faces.shrink_to_fit();
    vertices.shrink_to_fit();
    return true;
}


bool ExportMesh(const std::string filename, std::vector<uint32_t>& faces, std::vector<Vertex> &vertices)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG_ERROR("Error to create or open %s file", filename.c_str());
        return false;
    }

    file << std::fixed << std::setprecision(6);
    file << "# OBJ file exporter by Matteo Giuntoni custom exporter\n";
    file << "# Vertices: " << vertices.size() << "\n";
    file << "# Faces: " << faces.size() / 3 << "\n";


    for (const auto &vertex : vertices) {
        float r = static_cast<float>(vertex.R()) / 255.0f;
        float g = static_cast<float>(vertex.G()) / 255.0f;
        float b = static_cast<float>(vertex.B()) / 255.0f;
        
        file << "vn " << vertex.nX << " " << vertex.nY << " " << vertex.nZ << "\n";
        file << "v " << vertex.X << " " << vertex.Y << " " << vertex.Z << " " << r << " " << g << " " << b << "\n";
    }
    LOG_INFO("Vertices are loaded");
    file << "\n";

    for (size_t i = 0; i < faces.size(); i += 3) {
        uint32_t i1 = faces[i] + 1;
        uint32_t i2 = faces[i + 1] + 1;
        uint32_t i3 = faces[i + 2] + 1;

        file << "f " << i1 << "//" << i1 << " " << i2 << "//" << i2 << " " << i3 << "//" << i3 << "\n";
    }

    LOG_INFO("Faces are loaded. Mesh is exported");
    return true;
}


#endif // MESH_IO
