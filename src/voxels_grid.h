#ifndef VOXELS_GRID
#define VOXELS_GRID

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <sys/types.h>
#include <type_traits>

template<typename T, typename... Types>
__host__ __device__ static inline constexpr bool is_one_of = 
    ( std::is_same_v<T, Types> || ... );

template <
    typename T = uint8_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class HostVoxelsGrid;

template <
    typename T = uint8_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class DeviceVoxelsGrid;


template <
    typename T = uint8_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class VoxelsGrid 
{
    std::span<T> mVoxels;
    size_t mSideSize, mVoxelSize = 1;
    float mOriginX, mOriginY, mOriginZ = 0;

    class Bit {
        T& mWord;
        T mMask;

    public:

        __host__ __device__
        Bit(T& word, T mask) :
            mWord(word), mMask(mask) {}

        __host__ __device__
        Bit& operator= (bool value) 
        {
            if(value) mWord |= mMask;
            else      mWord &= ~mMask;
            return *this;
        }

        __host__ __device__
        operator bool() const { return (mWord & mMask) != 0; }
    };

public:

    VoxelsGrid() = default;

    __host__ __device__
    VoxelsGrid(T* data, const size_t side_size, const float voxel_size = 1.0f) :
        mVoxels(data, side_size * side_size * side_size), mVoxelSize(voxel_size) {}

    __host__ __device__
    Bit operator()(size_t x, size_t y, size_t z) {
        assert(x < mSideSize);
        assert(y < mSideSize);
        assert(z < mSideSize);

        size_t index = Index(x, y, z);
        return Bit(mVoxels[index / WordSize()], (1 << (index % WordSize())));
    }

    __host__ __device__
    bool operator()(size_t x, size_t y, size_t z) const {
        assert(x < mSideSize);
        assert(y < mSideSize);
        assert(z < mSideSize);

        size_t index = Index(x, y, z);
        return (mVoxels[index / WordSize()] & (1 << (index % WordSize()))) != 0;
    }

    __host__ __device__
    void SetOrigin(float x, float y, float z) { mOriginX = x; mOriginY = y; mOriginZ = z; }

    __host__ __device__
    inline size_t Index(size_t x, size_t y, size_t z) const { return (z * mSideSize * mSideSize) + (y * mSideSize) + x; }

    // The size of a single word, the voxels space are stored in block of uint (8, 16, 32 ... bit)
    __host__ __device__
    inline size_t WordSize() const { return sizeof(T) * 8; }

    // The size of alla voxels space (SideSize*SideSize*SideSize)
    __host__ __device__
    inline size_t SpaceSize() const { return mSideSize * mSideSize * mSideSize; }

    // The size of a side of voxels grid.
    __host__ __device__
    inline size_t SideSize() const { return mSideSize; }

    // The size of single voxel. Understood as the side of square
    __host__ __device__
    inline size_t VoxelSize() const { return mVoxelSize; }

    // How much word we need to store with a specific size
    static inline size_t StorageSize(const size_t side_size) { return (side_size * side_size * side_size + (sizeof(T) - 1)) / sizeof(T);}

    friend class HostVoxelsGrid<T>;
    friend class DeviceVoxelsGrid<T>;
};


template <typename T, typename>
class HostVoxelsGrid 
{
    std::unique_ptr<T> mData;
    VoxelsGrid<T> mView;

public:
    HostVoxelsGrid(const size_t side_size, const float voxel_size = 1.0f)
    {
        mData = std::make_unique<T[]>(VoxelsGrid<T>::StorageSize(side_size));
        mView = VoxelsGrid<T>(mData, side_size, voxel_size);
        std::fill(mView.mVoxels.begin(), mView.mVoxels.end(), 0);
    }

    HostVoxelsGrid(const HostVoxelsGrid&) = delete;
    HostVoxelsGrid& operator=(const HostVoxelsGrid&) = delete;

    inline VoxelsGrid<T> View() { return mView; }

    inline const VoxelsGrid<T> View() const { return mView; }
};


template <typename T, typename>
class DeviceVoxelsGrid 
{
    T* mData = nullptr;
    VoxelsGrid<T> mView;

public:
    DeviceVoxelsGrid(const size_t side_size, const float voxel_size = 1.0f)
    {
        gpuAssert(cudaMalloc((void**) &mData, VoxelsGrid<T>::StorageSize(side_size)));
        gpuAssert(cudaMemset(mData, 0, VoxelsGrid<T>::StorageSize(side_size)));
        mView = VoxelsGrid<T>(mData, side_size, voxel_size);
    }

    ~DeviceVoxelsGrid()
    {
        if(mData) 
            gpuAssert(cudaFree(mData));
    }   

    DeviceVoxelsGrid(const DeviceVoxelsGrid&) = delete;
    DeviceVoxelsGrid& operator=(const DeviceVoxelsGrid&) = delete;

    inline VoxelsGrid<T> View() { return mView; }

    inline const VoxelsGrid<T> View() const { return mView; }
};



using VoxelsGrid8bit  = VoxelsGrid<uint8_t>;
using VoxelsGrid16bit = VoxelsGrid<uint16_t>;
using VoxelsGrid32bit = VoxelsGrid<uint32_t>;
using VoxelsGrid64bit = VoxelsGrid<uint64_t>;

using HostVoxelsGrid8bit  = HostVoxelsGrid<uint8_t>;
using HostVoxelsGrid16bit = HostVoxelsGrid<uint16_t>;
using HostVoxelsGrid32bit = HostVoxelsGrid<uint32_t>;
using HostVoxelsGrid64bit = HostVoxelsGrid<uint64_t>;


using DeviceVoxelsGrid8bit  = DeviceVoxelsGrid<uint8_t>;
using DeviceVoxelsGrid16bit = DeviceVoxelsGrid<uint16_t>;
using DeviceVoxelsGrid32bit = DeviceVoxelsGrid<uint32_t>;
using DeviceVoxelsGrid64bit = DeviceVoxelsGrid<uint64_t>;

#endif // !VOXELS_GRID
