#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "vertex.h"
#include "voxels_grid.h"

#ifndef VOXEL_TO_MESH_H

using uint = unsigned int;

template <typename T, bool X = false,  bool Y = false, bool Z = false, bool front = false>
void AddFacesVertex(float voxelX, float voxelY, float voxelZ,
                    const VoxelsGrid<T>& grid,
                    std::vector<uint32_t>& facesCoord, 
                    std::vector<uint32_t>& facesNormals,
                    std::vector<Vertex>& coords,
                    std::array<std::vector<bool>, 3> &faces_marked, 
                    std::unordered_map<uint, std::pair<uint, Vertex>> &vertices_marked) 
{
    constexpr uint plane_index = (!Y * 2) + !X;
    const uint VERTEX_SIZE = grid.VoxelsPerSide() + 1;
    
    uint XX = X ? voxelX : voxelZ;
    uint YY = Y ? voxelY : voxelZ;
    uint ZZ = X ? (Y ? (voxelZ + front) : (voxelY + front)) : voxelX + front;
    
    uint face_index = (ZZ * grid.VoxelsPerSide() * grid.VoxelsPerSide()) + (YY * grid.VoxelsPerSide()) + XX;

    if(faces_marked[plane_index][face_index])
        return;

    faces_marked[plane_index][face_index] = true;


    std::array<uint, 4> final_vertex_index;
    for (uint v = 0; v < 2; ++v) {
        for (uint u = 0; u < 2; ++u) {
            uint Vx = voxelX + (!X * front) + (X * u);
            uint Vy = voxelY + (!Y * front) + (Y * v);
            uint Vz = voxelZ + (!Z * front) + (Z * ((X * v) + (Y * u)));

            uint vertex_index = (Vz * VERTEX_SIZE * VERTEX_SIZE) + (Vy * VERTEX_SIZE) + Vx;
            auto [vertex, is_new] = vertices_marked.try_emplace(vertex_index, 
                std::piecewise_construct, 
                std::forward_as_tuple(coords.size()), 
                std::forward_as_tuple(grid.OriginX() + (Vx * grid.VoxelSize()), 
                                      grid.OriginY() + (Vy * grid.VoxelSize()), 
                                      grid.OriginZ() + (Vz * grid.VoxelSize()))
            ); 
            if(is_new) 
                coords.push_back(vertex->second.second);
 
            final_vertex_index[u + (v * 2)] = vertex->second.first;
        } 
    } 

    if constexpr (front) {
        if constexpr (plane_index != 0) {
            facesCoord.insert(facesCoord.end(), {final_vertex_index[0], final_vertex_index[2], final_vertex_index[1]});
            facesCoord.insert(facesCoord.end(), {final_vertex_index[1], final_vertex_index[2], final_vertex_index[3]});
        } else {
            facesCoord.insert(facesCoord.end(), {final_vertex_index[0], final_vertex_index[1], final_vertex_index[2]});
            facesCoord.insert(facesCoord.end(), {final_vertex_index[1], final_vertex_index[3], final_vertex_index[2]});
        }
    } else {
        if constexpr (plane_index != 0) {
            facesCoord.insert(facesCoord.end(), {final_vertex_index[0], final_vertex_index[1], final_vertex_index[2]});
            facesCoord.insert(facesCoord.end(), {final_vertex_index[1], final_vertex_index[3], final_vertex_index[2]});
        } else {
            facesCoord.insert(facesCoord.end(), {final_vertex_index[0], final_vertex_index[2], final_vertex_index[1]});
            facesCoord.insert(facesCoord.end(), {final_vertex_index[1], final_vertex_index[2], final_vertex_index[3]});
        }
    }

    facesNormals.insert(facesNormals.end(), 6, (front * 3) + plane_index);
}

template <typename T, bool front = false>
inline void AddFacesVertexXY(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T> &grid, 
                             std::vector<uint32_t>& facesCoord, 
                             std::vector<uint32_t>& facesNormals,
                             std::vector<Vertex>& coords, 
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Vertex>> &vertices_marked) 
{
    AddFacesVertex<T, true, true, false, front>(
        voxelX, voxelY, voxelZ, grid, 
        facesCoord, facesNormals, coords, 
        faces_marked, vertices_marked
    );
}

template <typename T, bool front = false>
inline void AddFacesVertexXZ(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T> &grid,
                             std::vector<uint32_t>& facesCoord, 
                             std::vector<uint32_t>& facesNormals,
                             std::vector<Vertex>& coords, 
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Vertex>> &vertices_marked) 
{
    AddFacesVertex<T, true, false, true, front>(
        voxelX, voxelY, voxelZ, grid, 
        facesCoord, facesNormals, coords, 
        faces_marked, vertices_marked
    );
}

template <typename T, bool front = false>
inline void AddFacesVertexYZ(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T>& grid, 
                             std::vector<uint32_t>& facesCoord, 
                             std::vector<uint32_t>& facesNormals,
                             std::vector<Vertex>& coords, 
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Vertex>> &vertices_marked) 
{
    AddFacesVertex<T, false, true, true, front>(
        voxelX, voxelY, voxelZ, grid, 
        facesCoord, facesNormals, coords, 
        faces_marked, vertices_marked
    );
}

template<typename T>
bool VoxelsGridToMesh(const VoxelsGrid<T>& grid, 
                      std::vector<uint32_t>& facesCoord, 
                      std::vector<uint32_t>& facesNormals,
                      std::vector<Vertex>& coords,
                      std::vector<Normal>& normals,
                      std::vector<Color>& colors) 
{
    facesCoord.clear(); facesNormals.clear(); 
    coords.clear(), normals.clear(); colors.clear();

    uint max_vertices_num = std::pow(grid.VoxelsPerSide() + 1, 3);
    uint max_faces_num = std::pow(grid.VoxelsPerSide(), 2) * (grid.VoxelsPerSide() + 1);

    std::unordered_map<uint, std::pair<uint, Vertex>> vertices_marked(max_vertices_num);
    std::array<std::vector<bool>, 3> faces_marked = {
        std::vector<bool>(max_faces_num, false),
        std::vector<bool>(max_faces_num, false),
        std::vector<bool>(max_faces_num, false)
    };

    normals.emplace_back(0,0,1);
    normals.emplace_back(0,1,0);
    normals.emplace_back(1,0,0);
    
    normals.emplace_back(0,0,-1);
    normals.emplace_back(0,-1,0);
    normals.emplace_back(-1,0,0);

    for (uint z = 0; z < grid.VoxelsPerSide(); ++z) {
        for (uint y = 0; y < grid.VoxelsPerSide(); ++y) {
            for (uint x = 0; x < grid.VoxelsPerSide(); ++x) {
                if(!grid(x, y, z))
                    continue;
                
                AddFacesVertexXY<T, false>(x, y, z, grid, facesCoord, facesNormals, coords, faces_marked, vertices_marked);
                AddFacesVertexXY<T, true>(x, y, z, grid, facesCoord, facesNormals, coords, faces_marked, vertices_marked);

                AddFacesVertexXZ<T, false>(x, y, z, grid, facesCoord, facesNormals, coords, faces_marked, vertices_marked);
                AddFacesVertexXZ<T, true>(x, y, z, grid, facesCoord, facesNormals, coords, faces_marked, vertices_marked);

                AddFacesVertexYZ<T, false>(x, y, z, grid, facesCoord, facesNormals, coords, faces_marked, vertices_marked);
                AddFacesVertexYZ<T, true>(x, y, z, grid, facesCoord, facesNormals, coords, faces_marked, vertices_marked);
            } 
        }
    }

    colors.reserve(coords.size());
    colors.assign(coords.size(), Color(1.0f, 1.0f, 1.0f, 1.0f));

    #ifdef DEBUG
    for(uint i = 0; i < vertices.size(); ++i) {
        std::cout << i << ": " << vertices[i].X << " " << vertices[i].Y << " " << vertices[i].Z << std::endl;
    }

    for (uint i=0; i< faces.size(); i+= 3) {
        std::cout << faces[i] << " " << faces[i + 1] << " " << faces[i + 2] << std::endl;
    }
    #endif // DEBUG

    return true;
}

#endif // !VOXEL_TO_MESH_H
