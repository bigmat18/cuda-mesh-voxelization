#ifndef VOXELS_GRID
#define VOXELS_GRID

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

template<typename T, typename... Types>
inline constexpr bool is_one_of = ( std::is_same_v<T, Types> || ... );

template <
    typename T = uint8_t,
    typename = std::enable_if_t<is_one_of<T, uint8_t, uint16_t, uint32_t, uint64_t>>>
class VoxelsGrid {
    std::span<T> mVoxels;
    std::unique_ptr<T[]> mStorage;
    size_t mSideSize;

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

    VoxelsGrid(const size_t side_size) :
        mStorage(std::make_unique<T[]>((side_size * side_size * side_size + 7) / 8)),
        mVoxels(mStorage.get(), (side_size * side_size * side_size + 7) / 8),
        mSideSize(side_size)
    {
        std::fill(mVoxels.begin(), mVoxels.end(), 0);
    }

    VoxelsGrid(T* data, const size_t side_size) :
        mVoxels(data, side_size * side_size * side_size) {}


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
};

using VoxelsGrid8bit = VoxelsGrid<uint8_t>;

using VoxelsGrid16bit = VoxelsGrid<uint16_t>;

using VoxelsGrid32bit = VoxelsGrid<uint32_t>;

using VoxelsGrid64bit = VoxelsGrid<uint64_t>;

#endif // !VOXELS_GRID
