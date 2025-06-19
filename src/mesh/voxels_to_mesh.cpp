#include "voxels_grid.h"
#include <mesh/voxels_to_mesh.h>

template <typename T>
bool VoxelsGridToMesh(const VoxelsGrid<T>& grid, Mesh& mesh) 
{
    mesh.Clear();

    uint max_vertices_num = std::pow(grid.VoxelsPerSide() + 1, 3);
    uint max_faces_num = std::pow(grid.VoxelsPerSide(), 2) * (grid.VoxelsPerSide() + 1);

    std::unordered_map<uint, std::pair<uint, Position>> vertices_marked(max_vertices_num);
    std::array<std::vector<bool>, 3> faces_marked = {
        std::vector<bool>(max_faces_num, false),
        std::vector<bool>(max_faces_num, false),
        std::vector<bool>(max_faces_num, false)
    };

    mesh.Normals.emplace_back(0,0,1);
    mesh.Normals.emplace_back(0,1,0);
    mesh.Normals.emplace_back(1,0,0);
    
    mesh.Normals.emplace_back(0,0,-1);
    mesh.Normals.emplace_back(0,-1,0);
    mesh.Normals.emplace_back(-1,0,0);

    for (uint z = 0; z < grid.VoxelsPerSide(); ++z) {
        for (uint y = 0; y < grid.VoxelsPerSide(); ++y) {
            for (uint x = 0; x < grid.VoxelsPerSide(); ++x) {
                if(!grid(x, y, z))
                    continue;
                
                AddFacesVertexXY<T, false>(x, y, z, grid, mesh, faces_marked, vertices_marked);
                AddFacesVertexXY<T, true>(x, y, z, grid, mesh, faces_marked, vertices_marked);

                AddFacesVertexXZ<T, false>(x, y, z, grid, mesh, faces_marked, vertices_marked);
                AddFacesVertexXZ<T, true>(x, y, z, grid, mesh, faces_marked, vertices_marked);

                AddFacesVertexYZ<T, false>(x, y, z, grid, mesh, faces_marked, vertices_marked);
                AddFacesVertexYZ<T, true>(x, y, z, grid, mesh, faces_marked, vertices_marked);
            } 
        }
    }

    mesh.Colors.reserve(mesh.VerticesSize());
    mesh.Colors.assign(mesh.VerticesSize(), Color(1.0f, 1.0f, 1.0f, 1.0f));

    #ifdef DEBUG
    for(uint i = 0; i < mesh.Coords.size(); ++i) {
        LOG_INFO("%d: %f %f %f", i, mesh.Coords[i].X, mesh.Coords[i].Y, mesh.Coords[i].Z);
    }

    for (uint i=0; i< mesh.FacesCoords.size(); i+= 3) {
        LOG_INFO("%d %d %d", mesh.FacesCoords[i], mesh.FacesCoords[i+1], mesh.FacesCoords[i+2]);
    }
    #endif // DEBUG

    return true;
}

template bool VoxelsGridToMesh<uint8_t>
    (const VoxelsGrid<uint8_t> &grid, Mesh &mesh);

template bool VoxelsGridToMesh<uint16_t>
    (const VoxelsGrid<uint16_t> &grid, Mesh &mesh);

template bool VoxelsGridToMesh<uint32_t>
    (const VoxelsGrid<uint32_t> &grid, Mesh &mesh);

template bool VoxelsGridToMesh<uint64_t>
    (const VoxelsGrid<uint64_t> &grid, Mesh &mesh);
