#ifndef VOXELS_GRID_H
#define VOXELS_GRID_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <sys/types.h>
#include <type_traits>
#include <cassert>
#include <cuda_runtime.h>

#include <debug_utils.h>
#include <grid/grid.h>

template <typename T, typename... Ts> __device__
constexpr bool is_one_of_v = (std::is_same_v<T, Ts> || ...);

template <typename T>
concept VGType = is_one_of_v<T, uint32_t, uint64_t>;

template <VGType T>
class HostVoxelsGrid;

template <VGType T>
class DeviceVoxelsGrid;


template <VGType T, bool device = false>
class VoxelsGrid : protected Grid<T>
{
    using Grid<T>::mSizeX;
    using Grid<T>::mSizeY;
    using Grid<T>::mSizeZ;
    using Grid<T>::mGrid;

    float mVoxelSize = 1; 

    float mOriginX = 0;
    float mOriginY = 0;
    float mOriginZ = 0;

    class Bit {
        T* mWord;
        T mMask;

    public:
 
        __host__ __device__
        Bit(T* word, T mask) :
            mWord(word), mMask(mask) {}

        __host__ __device__
        Bit& operator=(bool value) 
        {
            if constexpr (device) {
                static_assert(sizeof(T) >= 4);
                if(value) atomicOr(mWord, mMask);
                else      atomicAnd(mWord, ~mMask);
            } else {
                if(value) (*mWord) |= mMask;
                else      (*mWord) &= ~mMask;
            }
            return *this;
        }

        __host__ __device__
        Bit& operator^=(bool value)
        {
            if constexpr (device) {
               static_assert(sizeof(T) >= 4);
               if(value) atomicXor(mWord, mMask);
            } else {
                if(value) (*mWord) ^= mMask;
            }
            return *this;
        }

        __host__ __device__
        operator bool() const { return ((*mWord) & mMask) != 0; }
    };

public:
    using Grid<T>::Size;
    using Grid<T>::SizeX;
    using Grid<T>::SizeY;
    using Grid<T>::SizeZ;
    using Grid<T>::Index;

    VoxelsGrid() = default;

    __host__ __device__
    VoxelsGrid(T* data, 
               const size_t voxelsPerSideX, 
               const size_t voxelsPerSideY, 
               const size_t voxelsPerSideZ,
               const float voxelSize = 1.0f) :
        mVoxelSize(voxelSize)
    {
        mSizeX = voxelsPerSideX; mSizeY = voxelsPerSideY; mSizeZ = voxelsPerSideZ;
        mGrid = std::span<T>(data, CalculateStorageSize(mSizeX, mSizeY, mSizeZ));
    }

    __host__ __device__
    VoxelsGrid(T* data, const size_t voxelsPerSide, const float voxelSize = 1) :
        mVoxelSize(voxelSize) 
    {
        mSizeX = voxelsPerSide; mSizeY = voxelsPerSide; mSizeZ = voxelsPerSide;
        mGrid = std::span<T>(data, CalculateStorageSize(mSizeX));
    }

    // ======= Method to acces ad voxels data ========
    __host__ __device__
    Bit Voxel(const uint x, const uint y, const uint z) 
    {
        assert(x < mSizeX); assert(y < mSizeY); assert(z < mSizeZ);
        const uint index = Index(x, y, z);
        return Bit(&mGrid[index / WordSize()], (T(1) << (index % WordSize())));
    }

    __host__ __device__
    bool Voxel(const uint x, const uint y, const uint z) const 
    {
        assert(x < mSizeX); assert(y < mSizeY); assert(z < mSizeZ); 
        const uint index = Index(x, y, z);
        return (mGrid[index / WordSize()] & (T(1) << (index % WordSize()))) != 0;
    }

    __host__ __device__
    T& Word(const uint x, const uint y, const uint z) 
    { 
        assert(x < mSizeX); assert(y < mSizeY); assert(z < mSizeZ);
        return mGrid[Index(x, y, z) / WordSize()]; 
    }

    __host__ __device__
    T Word(const uint x, const uint y, const uint z) const 
    { 
        assert(x < mSizeX); assert(y < mSizeY); assert(z < mSizeZ);
        return mGrid[Index(x, y, z) / WordSize()]; 
    }
    // ======= Method to acces ad voxels data =======

    __host__ __device__
    inline size_t VoxelsPerSide() const 
    { 
        assert(mSizeX == mSizeY && mSizeY == mSizeZ);
        return mSizeX; 
    }

    __host__ __device__
    inline float VoxelSize() const 
    { 
        return mVoxelSize; 
    }

    __host__ __device__
    void SetOrigin(float x, float y, float z) { mOriginX = x; mOriginY = y; mOriginZ = z; }

    __host__ __device__
    inline float OriginX() const { return mOriginX; }

    __host__ __device__
    inline float OriginY() const { return mOriginY; }
 
    __host__ __device__
    inline float OriginZ() const { return mOriginZ; }

    __host__ __device__
    inline void Print() const 
    {
        for(int z = 0; z < mSizeZ; ++z) {
            for (int y = 0; y < mSizeY; ++y) {
                for (int x = 0; x < mSizeX; ++x) {
                    printf("%d ", (*this).Voxel(x, y, z));    
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    __host__ __device__
    static inline size_t WordSize() { return sizeof(T) * 8; }

    __host__ __device__
    static inline size_t CalculateStorageSize(const size_t size) 
    {  
        return ((size * size * size) + (WordSize() - 1)) / WordSize();
    }

    __host__ __device__
    static inline size_t CalculateStorageSize(const size_t sizeX, 
                                              const size_t sizeY, 
                                              const size_t sizeZ) 
    { 
        return ((sizeX * sizeY * sizeZ) + (WordSize() - 1)) / WordSize();
    }

    friend class HostVoxelsGrid<T>;
    friend class DeviceVoxelsGrid<T>;
};


template <VGType T>
class HostVoxelsGrid 
{
    std::unique_ptr<T[]> mData;
    VoxelsGrid<T, false> mView;

public:
    HostVoxelsGrid() = default;

    HostVoxelsGrid(const size_t voxelsPerSideX, 
                   const size_t voxelsPerSideY, 
                   const size_t voxelsPerSideZ,
                   const float voxelSize = 1.0f);

    HostVoxelsGrid(const size_t voxelsPerSide, const float voxelSize = 1.0);

    HostVoxelsGrid(const DeviceVoxelsGrid<T>& device);

    HostVoxelsGrid(const HostVoxelsGrid<T>& other);

    HostVoxelsGrid(HostVoxelsGrid<T>&& other) { swap(other); }

    HostVoxelsGrid& operator=(HostVoxelsGrid<T> other) { swap(other); return *this; }

    void swap(HostVoxelsGrid<T>& other);

    friend void swap(HostVoxelsGrid<T>& first, HostVoxelsGrid<T>& second) { first.swap(second); }

    inline VoxelsGrid<T, false>& View() { return mView; }

    inline const VoxelsGrid<T, false>& View() const { return mView; }

    friend class DeviceVoxelsGrid<T>;
};


template <VGType T> 
class DeviceVoxelsGrid 
{
    T* mData = nullptr;
    VoxelsGrid<T, true> mView;

public:
    DeviceVoxelsGrid() = default;

    DeviceVoxelsGrid(const size_t voxelsPerSideX, 
                     const size_t voxelsPerSideY, 
                     const size_t voxelsPerSideZ,
                     const float voxelSize = 1.0f);

    DeviceVoxelsGrid(const size_t voxelsPerSide, const float voxelSize = 1.0f);

    DeviceVoxelsGrid(const HostVoxelsGrid<T>& host);

    DeviceVoxelsGrid(const DeviceVoxelsGrid<T>& other);

    DeviceVoxelsGrid(DeviceVoxelsGrid<T>&& other) { swap(other); }

    ~DeviceVoxelsGrid()
    {
        if(mData)   
            gpuAssert(cudaFree(mData));
    }   

    DeviceVoxelsGrid& operator=(DeviceVoxelsGrid<T> other) { swap(other); return *this; }

    void swap(DeviceVoxelsGrid<T>& other);

    friend void swap(DeviceVoxelsGrid<T>& first, DeviceVoxelsGrid<T>& second) { first.swap(second); }

    inline VoxelsGrid<T, true>& View() { return mView; }

    inline const VoxelsGrid<T, true>& View() const { return mView; }

    friend class HostVoxelsGrid<T>;
};


using HostVoxelsGrid32bit = HostVoxelsGrid<uint32_t>;
using HostVoxelsGrid64bit = HostVoxelsGrid<uint64_t>;

using DeviceVoxelsGrid32bit = DeviceVoxelsGrid<uint32_t>;
using DeviceVoxelsGrid64bit = DeviceVoxelsGrid<uint64_t>;

#endif // !VOXELS_GRID_H
