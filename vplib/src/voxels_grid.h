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
#include <grid.h>

template<typename T, typename... Types>
__device__ constexpr bool is_one_of = 
    ( std::is_same_v<T, Types> || ... );

template <
    typename T = uint8_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class HostVoxelsGrid;

template <
    typename T = uint32_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class DeviceVoxelsGrid;


template <
    typename T = uint32_t,
    bool device = false,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class VoxelsGrid : protected Grid<T>
{
    using Grid<T>::mSizeX;
    using Grid<T>::mSizeY;
    using Grid<T>::mSizeZ;
    using Grid<T>::mGrid;
    using Grid<T>::Index;

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

        size_t index = Index(x, y, z);
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
    T& Word(const uint x, const uint y, const uint z) { return Grid<T>::operator()(x, y, z); }

    __host__ __device__
    T Word(const uint x, const uint y, const uint z) const { return Grid<T>::operator()(x, y, z); }
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


template <typename T, typename>
class HostVoxelsGrid 
{
    std::unique_ptr<T[]> mData;
    VoxelsGrid<T, false> mView;

public:
    HostVoxelsGrid(const size_t voxelsPerSideX, 
                   const size_t voxelsPerSideY, 
                   const size_t voxelsPerSideZ,
                   const float voxelSize = 1.0f)
    {
        mData = std::make_unique<T[]>(VoxelsGrid<T>::CalculateStorageSize(voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ));
        mView = VoxelsGrid<T, false>(mData.get(), voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ, voxelSize);
        std::fill(mView.mGrid.begin(), mView.mGrid.end(), 0);
    }

    HostVoxelsGrid(const size_t voxelsPerSide, const float voxelSize = 1.0)
    {
        mData = std::make_unique<T[]>(VoxelsGrid<T>::CalculateStorageSize(voxelsPerSide));
        mView = VoxelsGrid<T, false>(mData.get(), voxelsPerSide, voxelSize);
        std::fill(mView.mGrid.begin(), mView.mGrid.end(), 0);
    }

    HostVoxelsGrid(const DeviceVoxelsGrid<T>& device) 
    {
        const VoxelsGrid v = device.View();
        const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(v.mSizeX, v.mSizeY, v.mSizeZ);

        mData = std::make_unique<T[]>(storageSize);
        mView = VoxelsGrid<T>(mData.get(), v.mSizeX, v.mVoxelSize); 
        mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
        gpuAssert(cudaMemcpy(mData.get(), device.mData, storageSize * sizeof(T), cudaMemcpyDeviceToHost));
    }

    HostVoxelsGrid(const HostVoxelsGrid& other) 
    {
        const VoxelsGrid v = other.View();
        const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(v.mSizeX, v.mSizeY, v.mSizeZ);

        mData = std::make_unique<T[]>(storageSize);
        mView = VoxelsGrid<T>(mData.get(), v.mSizeX, v.mVoxelSize);
        mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
        std::copy(mData.get(), mData.get() + mView.Size(), other.mData.get());
    }

    HostVoxelsGrid(HostVoxelsGrid&& other) { swap(other); }

    HostVoxelsGrid& operator=(const HostVoxelsGrid& other) { swap(other); return *this; }

    void swap(HostVoxelsGrid& other)
    {
        using std::swap;
        swap(mData, other.mData);
        swap(mView, other.mView);
    }

    friend void swap(HostVoxelsGrid& first, HostVoxelsGrid& second) { first.swap(second); }

    inline VoxelsGrid<T, false>& View() { return mView; }

    inline const VoxelsGrid<T, false>& View() const { return mView; }

    friend class DeviceVoxelsGrid<T>;
};


template <typename T, typename>
class DeviceVoxelsGrid 
{
    T* mData = nullptr;
    VoxelsGrid<T, true> mView;

public:
    DeviceVoxelsGrid(const size_t voxelsPerSideX, 
                     const size_t voxelsPerSideY, 
                     const size_t voxelsPerSideZ,
                     const float voxelSize = 1.0f)
    {
        const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(
            voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ) * sizeof(T);

        gpuAssert(cudaMalloc((void**) &mData, storageSize));   
        mView = VoxelsGrid<T, true>(mData, voxelsPerSideX, voxelsPerSideY, voxelsPerSideZ, voxelSize);
        gpuAssert(cudaMemset(mData, 0, storageSize));
    }

    DeviceVoxelsGrid(const size_t voxelsPerSide, const float voxelSize = 1.0f)
    {
        const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(voxelsPerSide) * sizeof(T);

        gpuAssert(cudaMalloc((void**) &mData, storageSize));   
        mView = VoxelsGrid<T, true>(mData, voxelsPerSide, voxelSize);
        gpuAssert(cudaMemset(mData, 0, storageSize));
    }

    DeviceVoxelsGrid(const HostVoxelsGrid<T>& host) 
    {
        const VoxelsGrid v = host.View();
        const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(
            v.mSizeX, v.mSizeY, v.mSizeZ) * sizeof(T);

        gpuAssert(cudaMalloc((void**) &mData, storageSize));
        mView = VoxelsGrid<T, true>(mData, v.mSizeX, v.mSizeY, v.mSizeZ, v.mVoxelSize);
        mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
        gpuAssert(cudaMemcpy(mData, host.mData.get(), storageSize, cudaMemcpyHostToDevice));
    }

    DeviceVoxelsGrid(const DeviceVoxelsGrid& other)
    {
        const VoxelsGrid v = other.View();
        const size_t storageSize = VoxelsGrid<T>::CalculateStorageSize(
            v.mSizeX, v.mSizeY, v.mSizeZ) * sizeof(T);

        gpuAssert(cudaMalloc((void**) &mData, storageSize));
        mView = VoxelsGrid<T, true>(mData, v.mSizeX, v.mSizeY, v.mSizeZ, v.mSideLength);
        mView.SetOrigin(v.OriginX(), v.OriginY(), v.OriginZ());
        gpuAssert(cudaMemcpy(mData, other.mData, storageSize, cudaMemcpyDeviceToDevice));
    }

    DeviceVoxelsGrid(DeviceVoxelsGrid&& other) { swap(other); }

    ~DeviceVoxelsGrid()
    {
        if(mData)   
            gpuAssert(cudaFree(mData));
    }   

    DeviceVoxelsGrid& operator=(const DeviceVoxelsGrid& other) { swap(other); return *this; }

    void swap(DeviceVoxelsGrid& other)
    {
        using std::swap;
        swap(mData, other.mData);
        swap(mView, other.mView);
    }

    friend void swap(DeviceVoxelsGrid& first, DeviceVoxelsGrid& second) { first.swap(second); }

    inline VoxelsGrid<T, true>& View() { return mView; }

    inline const VoxelsGrid<T, true>& View() const { return mView; }

    friend class HostVoxelsGrid<T>;
};


using VoxelsGrid8bit  = VoxelsGrid<uint8_t>;
using VoxelsGrid16bit = VoxelsGrid<uint16_t>;
using VoxelsGrid32bit = VoxelsGrid<uint32_t>;
using VoxelsGrid64bit = VoxelsGrid<uint64_t>;


using VoxelsGrid8bitHost  = VoxelsGrid<uint8_t, false>;
using VoxelsGrid16bitHost = VoxelsGrid<uint16_t, false>;
using VoxelsGrid32bitHost = VoxelsGrid<uint32_t, false>;
using VoxelsGrid64bitHost = VoxelsGrid<uint64_t, false>;

using HostVoxelsGrid8bit  = HostVoxelsGrid<uint8_t>;
using HostVoxelsGrid16bit = HostVoxelsGrid<uint16_t>;
using HostVoxelsGrid32bit = HostVoxelsGrid<uint32_t>;
using HostVoxelsGrid64bit = HostVoxelsGrid<uint64_t>;


using VoxelsGrid32bitDev = VoxelsGrid<uint32_t, true>;
using VoxelsGrid64bitDev = VoxelsGrid<uint64_t, true>;    

using DeviceVoxelsGrid32bit = DeviceVoxelsGrid<uint32_t>;
using DeviceVoxelsGrid64bit = DeviceVoxelsGrid<uint64_t>;

#endif // !VOXELS_GRID_H
