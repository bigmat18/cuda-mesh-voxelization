#include <mesh/mesh_io.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <ios>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <cmath>

bool ImportMesh(const std::string filename, Mesh& mesh) 
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

    //LOG_INFO("%s loaded sucessfully", filename.c_str());
    std::string line;
    std::string X, Y, Z, nX, nY, nZ, r, g, b;

    mesh.Clear();

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ss.imbue(std::locale::classic()); 

        std::string prefix;
        ss >> prefix;

        if(prefix == "#") {
            int count;      
            if (sscanf(line.c_str(), "# Vertices: %d", &count) == 1) {
                mesh.VerticesReserve(count);
            } else if (sscanf(line.c_str(), "# Faces: %d", &count) == 1) {
                mesh.FacesReserve(count);
            }
        } else if (prefix == "vn") {
            ss >> nX >> nY >> nZ;
            mesh.Normals.emplace_back(std::stof(nX), std::stof(nY), std::stof(nZ));
        } else if (prefix == "v") {
            ss >> X >> Y >> Z;
            mesh.Coords.emplace_back(std::stof(X), std::stof(Y), std::stof(Z));

            bool hasRGB = bool(ss >> r >> g >> b);
            if(hasRGB)
                mesh.Colors.emplace_back(std::stof(r), std::stof(g), std::stof(g), 1.0f);
        } else if (prefix == "f") {
            uint32_t pos_inx, norm_idx;
            std::string vertex_str;

            for (uint32_t i = 0; i < 3; ++i) {
                ss >> vertex_str;
                sscanf(vertex_str.c_str(), " %d//%d", &pos_inx, &norm_idx);

                if(pos_inx != norm_idx)
                    LOG_WARN("Face coords and face norms are not equals");

                mesh.FacesCoords.push_back(pos_inx - 1);
                mesh.FacesNormals.push_back(norm_idx - 1);
            }
        }
    }
   
    mesh.Name = filename;
    mesh.ShrinkToFit();
    //LOG_INFO("Mesh %s sucessfully imported", filename.c_str());
    return true;
}


bool ExportMesh(const std::string filename, const Mesh& mesh)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        LOG_ERROR("Error to create or open %s file", filename.c_str());
        return false;
    }

    file << std::fixed << std::setprecision(6);
    file << "# OBJ file exporter by Matteo Giuntoni custom exporter\n";
    file << "# Vertices: " << mesh.VerticesSize() << "\n";
    file << "# Faces: " << mesh.FacesSize() << "\n";


    for (size_t i = 0; i < mesh.VerticesSize(); ++i) {
        float r = static_cast<float>(mesh.Colors[i].R()) / 255.0f;
        float g = static_cast<float>(mesh.Colors[i].G()) / 255.0f;
        float b = static_cast<float>(mesh.Colors[i].B()) / 255.0f;
        
        file << "v " << mesh.Coords[i].X << " " << mesh.Coords[i].Y << " " << mesh.Coords[i].Z << " " << r << " " << g << " " << b << "\n";
    }
    //LOG_INFO("Coords are loaded");
    file << "\n";

    for (size_t i = 0; i < mesh.NormalsSize(); ++i) {
        file << "vn " << mesh.Normals[i].X << " " << mesh.Normals[i].Y << " " << mesh.Normals[i].Z << "\n";
    }

    //LOG_INFO("Normals are loades");
    file << "\n";

    for (size_t i = 0; i < mesh.FacesSize() * 6; i += 3) {
        uint32_t i1 = mesh.FacesCoords[i] + 1;
        uint32_t i2 = mesh.FacesCoords[i + 1] + 1;
        uint32_t i3 = mesh.FacesCoords[i + 2] + 1;
 
        uint32_t in1 = mesh.FacesNormals[i] + 1;
        uint32_t in2 = mesh.FacesNormals[i + 1] + 1;
        uint32_t in3 = mesh.FacesNormals[i + 2] + 1;
 
        file << "f " << i1 << "//" << in1 << " " << i2 << "//" << in2 << " " << i3 << "//" << in3 << "\n";
    }

    //LOG_INFO("Faces are loaded.");

    LOG_INFO("Mesh %s sucessfully exported", filename.c_str());
    return true;
}

