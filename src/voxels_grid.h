#ifndef VOXELS_GRID
#define VOXELS_GRID

#include "mesh_io.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <sys/types.h>
#include <type_traits>
#include <unordered_map>
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

    inline size_t VoxelSide() const { return mVoxelSize; }
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
    
    uint faceX = X ? voxelX : voxelZ;
    uint faceY = Y ? voxelY : voxelZ;
    uint faceZ = X ? (Y ? (voxelZ + front) : (voxelY + front)) : voxelX + front;
    
    uint face_index = (faceZ * grid.SideSize() * grid.SideSize()) + (faceY * grid.SideSize()) + faceX;

    if(faces_marked[plane_index][face_index])
        return;

    faces_marked[plane_index][face_index] = true;

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

    LOG_INFO("%d", count);

    return true;
}

#endif // !VOXELS_GRID
