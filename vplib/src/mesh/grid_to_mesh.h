#ifndef GRID_TO_MESH_H
#define GRID_TO_MESH_H

#include <array>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mesh/mesh.h>
#include <grid/voxels_grid.h>

using uint = unsigned int;

inline float Clamp(float x, float a, float b) { return std::max(a, std::min(x, b)); }

inline std::tuple<float, float, float> SDFToRGB(float d, float dmin, float dmax) {
    float t = (Clamp(d, dmin, dmax) - dmin) / (dmax - dmin);

    float r, g, b;
    if (t < 0.5f) {
        float s = t / 0.5f;
        r = s; g = s; b = 1.0f;
    } else {
        float s = (t - 0.5f) / 0.5f;
        r = 1.0f; g = 1.0f - s; b = 1.0f - s;
    }
    return {r, g, b};
}

template <typename T, bool X = false,  bool Y = false, bool Z = false, bool front = false>
void AddFacesVertex(float voxelX, float voxelY, float voxelZ,
                    const VoxelsGrid<T>& grid, Mesh& mesh,
                    std::array<std::vector<bool>, 3> &faces_marked, 
                    std::unordered_map<uint, std::pair<uint, Position>> &vertices_marked) 
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
                std::forward_as_tuple(mesh.Coords.size()), 
                std::forward_as_tuple(grid.OriginX() + (Vx * grid.VoxelSize()), 
                                      grid.OriginY() + (Vy * grid.VoxelSize()), 
                                      grid.OriginZ() + (Vz * grid.VoxelSize()))
            ); 
            if(is_new) 
                mesh.Coords.push_back(vertex->second.second);
 
            final_vertex_index[u + (v * 2)] = vertex->second.first;
        } 
    } 

    if constexpr (front) {
        if constexpr (plane_index != 0) {
            mesh.FacesCoords.insert(mesh.FacesCoords.end(), {final_vertex_index[0], final_vertex_index[2], final_vertex_index[1]});
            mesh.FacesCoords.insert(mesh.FacesCoords.end(), {final_vertex_index[1], final_vertex_index[2], final_vertex_index[3]});
        } else {
            mesh.FacesCoords.insert(mesh.FacesCoords.end(), {final_vertex_index[0], final_vertex_index[1], final_vertex_index[2]});
            mesh.FacesCoords.insert(mesh.FacesCoords.end(), {final_vertex_index[1], final_vertex_index[3], final_vertex_index[2]});
        }
    } else {
        if constexpr (plane_index != 0) {
            mesh.FacesCoords.insert(mesh.FacesCoords.end(), {final_vertex_index[0], final_vertex_index[1], final_vertex_index[2]});
            mesh.FacesCoords.insert(mesh.FacesCoords.end(), {final_vertex_index[1], final_vertex_index[3], final_vertex_index[2]});
        } else {
            mesh.FacesCoords.insert(mesh.FacesCoords.end(), {final_vertex_index[0], final_vertex_index[2], final_vertex_index[1]});
            mesh.FacesCoords.insert(mesh.FacesCoords.end(), {final_vertex_index[1], final_vertex_index[2], final_vertex_index[3]});
        }
    }

    mesh.FacesNormals.insert(mesh.FacesNormals.end(), 6, (front * 3) + plane_index);
}

template <typename T, bool front = false>
inline void AddFacesVertexXY(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T> &grid, Mesh& mesh,
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Position>> &vertices_marked) 
{
    AddFacesVertex<T, true, true, false, front>(
        voxelX, voxelY, voxelZ, grid, mesh,        
        faces_marked, vertices_marked
    );
}

template <typename T, bool front = false>
inline void AddFacesVertexXZ(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T> &grid, Mesh& mesh,
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Position>> &vertices_marked) 
{
    AddFacesVertex<T, true, false, true, front>(
        voxelX, voxelY, voxelZ, grid, mesh,
        faces_marked, vertices_marked
    );
}

template <typename T, bool front = false>
inline void AddFacesVertexYZ(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T>& grid, Mesh& mesh,
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Position>> &vertices_marked) 
{
    AddFacesVertex<T, false, true, true, front>(
        voxelX, voxelY, voxelZ, grid, mesh,
        faces_marked, vertices_marked
    );
}

template <typename T>
bool VoxelsGridToMesh(const VoxelsGrid<T>& grid, Mesh& mesh); 


template <typename T>
bool VoxelsGridToMeshSDFColor(const VoxelsGrid<T>& grid, const Grid<float>& colors, Mesh& mesh); 

#endif // !GRID_TO_MESH_H
