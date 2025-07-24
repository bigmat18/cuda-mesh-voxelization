#include "grid/voxels_grid.h"
#include "jfa/jfa.h"
#include "mesh/mesh.h"
#include <cmath>
#include <cstdint>
#include <mesh/grid_to_mesh.h>

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
                if(!grid.Voxel(x, y, z))
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

    #if 0
    for(uint i = 0; i < mesh.Coords.size(); ++i)
        LOG_INFO("%d: %f %f %f", i, mesh.Coords[i].X, mesh.Coords[i].Y, mesh.Coords[i].Z);

    for (uint i=0; i< mesh.FacesCoords.size(); i+= 3)
        LOG_INFO("%d %d %d", mesh.FacesCoords[i], mesh.FacesCoords[i+1], mesh.FacesCoords[i+2]);
    #endif

    return true;
}


template <typename T>
bool VoxelsGridToMeshSDFColor(const VoxelsGrid<T>& grid, const Grid<float>& colors, Mesh& mesh) 
{
    mesh.Clear();

    uint max_vertices_num = grid.Size() * 8;
    uint max_faces_num = grid.Size() * 6;

    mesh.VerticesReserve(max_vertices_num);
    mesh.FacesReserve(max_faces_num);

    mesh.Normals.emplace_back(0,0,1);
    mesh.Normals.emplace_back(0,1,0);
    mesh.Normals.emplace_back(1,0,0);
    
    mesh.Normals.emplace_back(0,0,-1);
    mesh.Normals.emplace_back(0,-1,0);
    mesh.Normals.emplace_back(-1,0,0);

    float max = std::sqrt(std::pow(grid.VoxelsPerSide() * grid.VoxelSize(), 2) * 3);
    unsigned int numberVoxelInsert = 0;
    for (uint z = 0; z < grid.VoxelsPerSide(); ++z) {
        for (uint y = 0; y < grid.VoxelsPerSide(); ++y) {
            for (uint x = 0; x < grid.VoxelsPerSide(); ++x) {
                if(!grid.Voxel(x, y, z))
                    continue;
               
                for(int dz = 0; dz <= 1; ++dz) {
                    for (int dy = 0; dy <= 1; ++dy) {
                        for (int dx = 0; dx <= 1; ++dx) {
                            mesh.Coords.emplace_back(
                                grid.OriginX() + (x * grid.VoxelSize()) + (grid.VoxelSize() * dx),
                                grid.OriginY() + (y * grid.VoxelSize()) + (grid.VoxelSize() * dy),
                                grid.OriginZ() + (z * grid.VoxelSize()) + (grid.VoxelSize() * dz)
                            );
                            auto rgb = SDFToRGB(std::sqrt(colors(x, y, z)), max);

                            mesh.Colors.emplace_back(std::get<0>(rgb), std::get<1>(rgb), std::get<2>(rgb), 1.0f);
                        }
                    }
                }

                // ===== BACK =====
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 0, 
                                                                 (numberVoxelInsert * 8) + 2, 
                                                                 (numberVoxelInsert * 8) + 1});
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 1, 
                                                                 (numberVoxelInsert * 8) + 2, 
                                                                 (numberVoxelInsert * 8) + 3});
                mesh.FacesNormals.insert(mesh.FacesNormals.end(), 6, 0);

                // ===== FRONT =====
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 4, 
                                                                 (numberVoxelInsert * 8) + 5, 
                                                                 (numberVoxelInsert * 8) + 6});
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 5, 
                                                                 (numberVoxelInsert * 8) + 7, 
                                                                 (numberVoxelInsert * 8) + 6});
                mesh.FacesNormals.insert(mesh.FacesNormals.end(), 6, 3);


                // ===== TOP =====
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 6, 
                                                                 (numberVoxelInsert * 8) + 3, 
                                                                 (numberVoxelInsert * 8) + 2});
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 3, 
                                                                 (numberVoxelInsert * 8) + 6, 
                                                                 (numberVoxelInsert * 8) + 7});
                mesh.FacesNormals.insert(mesh.FacesNormals.end(), 6, 1);


                // ===== BOTTOM =====
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 0, 
                                                                 (numberVoxelInsert * 8) + 1, 
                                                                 (numberVoxelInsert * 8) + 4});
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 1, 
                                                                 (numberVoxelInsert * 8) + 5, 
                                                                 (numberVoxelInsert * 8) + 4});
                mesh.FacesNormals.insert(mesh.FacesNormals.end(), 6, 4);


                // ===== RIGHT =====
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 1, 
                                                                 (numberVoxelInsert * 8) + 3, 
                                                                 (numberVoxelInsert * 8) + 5});
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 3, 
                                                                 (numberVoxelInsert * 8) + 7, 
                                                                 (numberVoxelInsert * 8) + 5});
                mesh.FacesNormals.insert(mesh.FacesNormals.end(), 6, 2);


                // ===== LEFT =====
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 0, 
                                                                 (numberVoxelInsert * 8) + 4, 
                                                                 (numberVoxelInsert * 8) + 2});
                mesh.FacesCoords.insert(mesh.FacesCoords.end(), {(numberVoxelInsert * 8) + 2, 
                                                                 (numberVoxelInsert * 8) + 4, 
                                                                 (numberVoxelInsert * 8) + 6});
                mesh.FacesNormals.insert(mesh.FacesNormals.end(), 6, 5);

                numberVoxelInsert++;
            } 
        }
    }
    mesh.ShrinkToFit();

    return true;
}


template bool VoxelsGridToMeshSDFColor<uint32_t>
 (const VoxelsGrid<uint32_t>&, const Grid<float>&, Mesh&);

template bool VoxelsGridToMeshSDFColor<uint64_t>
 (const VoxelsGrid<uint64_t>&, const Grid<float>&, Mesh&);

template bool VoxelsGridToMesh<uint32_t>
    (const VoxelsGrid<uint32_t> &grid, Mesh &mesh);

template bool VoxelsGridToMesh<uint64_t>
    (const VoxelsGrid<uint64_t> &grid, Mesh &mesh);
