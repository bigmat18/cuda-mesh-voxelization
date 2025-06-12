#ifndef VOXELS_GRID
#define VOXELS_GRID

#include "mesh_io.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

template<typename T, typename... Types>
inline constexpr bool is_one_of = ( std::is_same_v<T, Types> || ... );

template <
    typename T = uint8_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class VoxelsGrid {
    std::span<T> mVoxels;
    std::unique_ptr<T[]> mStorage;
    size_t mSideSize, mVoxelSize;

    class Bit {
        T& mWord;
        T mMask;

    public:
        Bit(T& word, T mask) :
            mWord(word), mMask(mask) {}

        Bit& operator= (bool value) 
        {
            if(value) mWord |= mMask;
            else      mWord &= ~mMask;
            return *this;
        }

        operator bool() const { return (mWord & mMask) != 0; }
    };

public:

    VoxelsGrid(const size_t side_size, const float voxel_size = 1.0f) :
        mSideSize(side_size), mVoxelSize(voxel_size)
    {
        const size_t storage_size = (side_size * side_size * side_size + 7) / 8;
        mStorage = std::make_unique<T[]>(storage_size);
        mVoxels = std::span<T>(mStorage.get(), storage_size);

        std::fill(mVoxels.begin(), mVoxels.end(), 0);
    }

    VoxelsGrid(T* data, const size_t side_size, const float voxel_size = 1.0f) :
        mVoxels(data, side_size * side_size * side_size), mVoxelSize(voxel_size) {}


    ~VoxelsGrid() = default;

    Bit operator()(size_t x, size_t y, size_t z) {
        assert(x < mSideSize);
        assert(y < mSideSize);
        assert(z < mSideSize);

        size_t index = Index(x, y, z);
        return Bit(mVoxels[index / WordSize()], (1 << (index % WordSize())));
    }

    bool operator()(size_t x, size_t y, size_t z) const {
        assert(x < mSideSize);
        assert(y < mSideSize);
        assert(z < mSideSize);

        size_t index = Index(x, y, z);
        return (mVoxels[index / WordSize()] & (1 << (index % WordSize()))) != 0;
    }

    inline size_t Index(size_t x, size_t y, size_t z) const { return (z * mSideSize * mSideSize) + (y * mSideSize) + x; }

    inline size_t WordSize() const { return sizeof(T) * 8; }

    inline size_t Size() const { return mSideSize * mSideSize * mSideSize; }

    inline size_t SideSize() const { return mSideSize; }

    inline size_t VoxelSize() const { return mVoxelSize; }
};

using VoxelsGrid8bit = VoxelsGrid<uint8_t>;

using VoxelsGrid16bit = VoxelsGrid<uint16_t>;

using VoxelsGrid32bit = VoxelsGrid<uint32_t>;

using VoxelsGrid64bit = VoxelsGrid<uint64_t>;

using uint = unsigned int;

template <typename T, bool X = false,  bool Y = false, bool Z = false, bool front = false>
void AddFacesVertex(float voxelX, float voxelY, float voxelZ,
                    const VoxelsGrid<T> &grid,
                    std::vector<uint32_t> &faces, std::vector<Vertex> &vertices,
                    std::array<std::vector<bool>, 3> &faces_marked, 
                    std::unordered_map<uint, std::pair<uint, Vertex>> &vertices_marked) 
{
    constexpr uint plane_index = (!Y * 2) + !X;
    const uint VERTEX_SIZE = grid.SideSize() + 1;
    
    uint XX = X ? voxelX : voxelZ;
    uint YY = Y ? voxelY : voxelZ;
    uint ZZ = X ? (Y ? (voxelZ + front) : (voxelY + front)) : voxelX + front;
    
    uint face_index = (ZZ * grid.SideSize() * grid.SideSize()) + (YY * grid.SideSize()) + XX;

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
                std::forward_as_tuple(vertices.size()), 
                std::forward_as_tuple(Vx * grid.VoxelSize(), Vy * grid.VoxelSize(), Vz * grid.VoxelSize(), 0.f, 0.f, 0.f, 0xFFFFFFFF)
            );
            if(is_new)
                vertices.push_back(vertex->second.second);

            final_vertex_index[u + (v * 2)] = vertex->second.first;
        }
    }

    faces.insert(faces.end(), {final_vertex_index[0], final_vertex_index[2], final_vertex_index[1]});
    faces.insert(faces.end(), {final_vertex_index[1], final_vertex_index[2], final_vertex_index[3]});
}

template <typename T, bool front = false>
inline void AddFacesVertexXY(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T> &grid, 
                             std::vector<uint32_t> &faces, std::vector<Vertex> &vertices, 
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Vertex>> &vertices_marked) 
{
    AddFacesVertex<T, true, true, false, front>(voxelX, voxelY, voxelZ, grid, faces, vertices, faces_marked, vertices_marked);
}

template <typename T, bool front = false>
inline void AddFacesVertexXZ(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T> &grid, 
                             std::vector<uint32_t> &faces, std::vector<Vertex> &vertices, 
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Vertex>> &vertices_marked) 
{
    AddFacesVertex<T, true, false, true, front>(voxelX, voxelY, voxelZ, grid, faces, vertices, faces_marked, vertices_marked);
}

template <typename T, bool front = false>
inline void AddFacesVertexYZ(float voxelX, float voxelY, float voxelZ, 
                             const VoxelsGrid<T> &grid, 
                             std::vector<uint32_t> &faces, std::vector<Vertex> &vertices, 
                             std::array<std::vector<bool>, 3> &faces_marked, 
                             std::unordered_map<uint, std::pair<uint, Vertex>> &vertices_marked) 
{
    AddFacesVertex<T, false, true, true, front>(voxelX, voxelY, voxelZ, grid, faces, vertices, faces_marked, vertices_marked);
}

template<typename T>
bool VoxelsGridToMesh(const VoxelsGrid<T> &grid, std::vector<uint32_t> &faces, std::vector<Vertex> &vertices) 
{

    uint max_vertices_num = std::pow(grid.SideSize() + 1, 3);
    uint max_faces_num = std::pow(grid.SideSize(), 2) * (grid.SideSize() + 1);

    std::unordered_map<uint, std::pair<uint, Vertex>> vertices_marked(max_vertices_num);
    std::array<std::vector<bool>, 3> faces_marked = {
        std::vector<bool>(max_faces_num, false),
        std::vector<bool>(max_faces_num, false),
        std::vector<bool>(max_faces_num, false)
    };


    for (uint z = 0; z < grid.SideSize(); ++z) {
        for (uint y = 0; y < grid.SideSize(); ++y) {
            for (uint x = 0; x < grid.SideSize(); ++x) {
                if(!grid(x, y, z))
                    continue;
                
                AddFacesVertexXY<T, false>(x, y, z, grid, faces, vertices, faces_marked, vertices_marked);
                AddFacesVertexXY<T, true>(x, y, z, grid, faces, vertices, faces_marked, vertices_marked);

                AddFacesVertexXZ<T, false>(x, y, z, grid, faces, vertices, faces_marked, vertices_marked);
                AddFacesVertexXZ<T, true>(x, y, z, grid, faces, vertices, faces_marked, vertices_marked);

                AddFacesVertexYZ<T, false>(x, y, z, grid, faces, vertices, faces_marked, vertices_marked);
                AddFacesVertexYZ<T, true>(x, y, z, grid, faces, vertices, faces_marked, vertices_marked);
            } 
        }
    }

    int count = 0;

    for (int i = 0; i < max_faces_num; i++) {
       count += faces_marked[0][i];
    }
    for (int i = 0; i < max_faces_num; i++) {
       count += faces_marked[1][i];
    }
    for (int i = 0; i < max_faces_num; i++) {
       count += faces_marked[2][i];
    }

    LOG_INFO("Faces counted: %d", count);

    LOG_INFO("Vertices counted: %ld", vertices_marked.size());

    for(uint i = 0; i < vertices.size(); ++i) {
        std::cout << i << ": " << vertices[i].X << " " << vertices[i].Y << " " << vertices[i].Z << std::endl;
    }

    for (uint i=0; i< faces.size(); i+= 3) {
        std::cout << faces[i] << " " << faces[i + 1] << " " << faces[i + 2] << std::endl;
    }

    return true;
}

#endif // !VOXELS_GRID
