#ifndef VOXELS_GRID
#define VOXELS_GRID

#include "mesh_io.h"
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

    VoxelsGrid(const size_t side_size, const float vexel_size = 1.0f) :
        mStorage(std::make_unique<T[]>((side_size * side_size * side_size + 7) / 8)),
        mVoxels(mStorage.get(), (side_size * side_size * side_size + 7) / 8),
        mSideSize(side_size), mVoxelSize(voxel_side)
    {
        std::fill(mVoxels.begin(), mVoxels.end(), 0);
    }

    VoxelsGrid(T* data, const size_t side_size, const float vexel_size = 1.0f) :
        mVoxels(data, side_size * side_size * side_size), mVoxelSize(voxel_side) {}


    Bit operator()(size_t x, size_t y, size_t z) {
        assert(x < mSideSize);
        assert(y < mSideSize);
        assert(z < mSideSize);

        size_t index = (x * mSideSize * mSideSize) + (y * mSideSize) + z;
        return Bit(mVoxels[index / WordSize()], (1 << (index % WordSize())));
    }

    bool operator()(size_t x, size_t y, size_t z) const {
        assert(x < mSideSize);
        assert(y < mSideSize);
        assert(z < mSideSize);

        size_t index = (x * mSideSize * mSideSize) + (y * mSideSize) + z;
        return (mVoxels[index / WordSize()] & (1 << (index % WordSize()))) != 0;
    }

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
                    const std::vector<uint32_t> &faces, const std::vector<Vertex> &vertices,
                    const std::vector<bool>& faces_marked, const std::unordered_map<uint, std::pair<uint, Vertex>> vertices_marked) 
{
    constexpr uint face_index = (Y * 4) + (X * 2);
    uint voxel_index = (voxelX * grid.SideSize() * grid.SideSize()) + (voxelY * grid.SideSize()) + voxelZ;
    uint index = face_index;

    if (faces_marked[index])
        return;

    
}


template<typename T>
bool VoxelsGridToMesh(const VoxelsGrid<T> &grid, const std::vector<uint32_t> &faces, const std::vector<Vertex> &vertices) 
{

    uint max_vertices_num = std::pow(grid.SideSize() + 1, 3);
    uint max_faces_num = (3 * std::pow(grid.SideSize(), 3)) + (3 * std::pow(grid.SideSize(), 3));

    std::unordered_map<uint, std::pair<uint, Vertex>> vertices_marked(max_vertices_num);
    std::vector<bool> faces_marked(max_faces_num, false);


    for (uint x = 0; x < grid.SideSize(); ++x) {
        for (uint y = 0; y < grid.SideSize(); ++y) {
            for (uint z = 0; z < grid.SideSize(); ++z) {
                if(!grid(x, y, z))
                    continue;
                    
                float worldX = x * grid.VoxelSide();
                float worldY = y * grid.VoxelSide();
                float worldZ = z * grid.VoxelSide();

                uint general_index = (x * grid.SideSize() * grid.SideSize()) + (y * grid.SideSize()) + z;

                if(faces_marked[general_index])
                    continue;

            } 
        }
    }

    return true;
}

#endif // !VOXELS_GRID
