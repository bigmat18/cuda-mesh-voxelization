#ifndef VERTEX_H
#define VERTEX_H

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

struct Color {
    __host__ __device__ Color() = default;
    
    __host__ __device__ Color(float r, float g, float b, float a) {
        SetColor(r, g, b, a);
    }

    __host__ __device__ Color(uint32_t color) : mColor(color) {}

    __host__ __device__ inline void SetColor(float r, float g, float b, float a) {
      mColor = (static_cast<uint32_t>(std::round(r * 255)) << 24) |
              (static_cast<uint32_t>(std::round(g * 255)) << 16) |
              (static_cast<uint32_t>(std::round(b * 255)) << 8) |
              (static_cast<uint32_t>(std::round(a * 255)));
    }

    __host__ __device__ inline uint8_t R() const { return (mColor >> 24) & 0xFF; }
    __host__ __device__ void R(float r) { SetColor(r, G(), B(), A()); }

    __host__ __device__ inline uint8_t G() const { return (mColor >> 16) & 0xFF; }
    __host__ __device__ void G(float g) { SetColor(R(), g, B(), A()); }

    __host__ __device__ inline uint8_t B() const { return (mColor >> 8) & 0xFF; }
    __host__ __device__ void B(float b) { SetColor(R(), G(), b, A()); }

    __host__ __device__ inline uint8_t A() const { return mColor & 0xFF; }
    __host__ __device__ void A(float a) { SetColor(R(), G(), B(), a); }

private:
    uint32_t mColor = 0xFFFFFFFF;
};


template <typename T = float> 
struct Vec3 {
    T X = 0; 
    T Y = 0;
    T Z = 0;

    __host__ __device__ Vec3() = default;

    __host__ __device__ Vec3(T x, T y, T z) : X(x), Y(y), Z(z) {}

    __host__ __device__ Vec3(const Vec3<T> &v) : X(v.X), Y(v.Y), Z(v.Z) {}

    __host__ __device__ Vec3<T> operator+(const Vec3<T> &v) const {
      return Vec3<T>(X + v.X, Y + v.Y, Z + v.Z);
    }

    __host__ __device__ Vec3<T> operator-(const Vec3<T> &v) const {
      return Vec3<T>(X - v.X, Y - v.Y, Z - v.Z);
    }

    __host__ __device__ Vec3<T> operator+(const T scalar) const {
      return Vec3<T>(X + scalar, Y + scalar, Z + scalar);
    }

    __host__ __device__ Vec3<T> operator-(const T scalar) const {
      return Vec3<T>(X - scalar, Y - scalar, Z - scalar);
    }

    __host__ __device__ Vec3<T> operator*(const T scalar) const {
      return Vec3<T>(X * scalar, Y * scalar, Z * scalar);
    }

    __host__ __device__ Vec3<T> operator/(const T scalar) const {
      return Vec3<T>(X / scalar, Y / scalar, Z / scalar);
    }

    __host__ __device__ Vec3<T> operator-() const { return Vec3<T>(-X, -Y, -Z); }

    __host__ __device__ Vec3<T> &operator=(const Vec3<T> &v) {
      if (this != &v) { X = v.X; Y = v.Y; Z = v.Z; }
      return *this;
    }

    __host__ __device__ Vec3<T> &operator+=(const Vec3<T> &v) {
      X += v.X; Y += v.Y; Z += v.Z;
      return *this;
    }

    __host__ __device__ Vec3<T> &operator-=(const Vec3<T> &v) {
      X -= v.X; Y -= v.Y; Z -= v.Z;
      return *this;
     }

    __host__ __device__ Vec3<T> &operator*=(T scalar) {
      X *= scalar; Y *= scalar; Z *= scalar;
      return *this;
    }

    __host__ __device__ Vec3<T> &operator/=(T scalar) {
      X /= scalar; Y /= scalar; Z /= scalar;
      return *this;
    }

    __host__ __device__ bool operator==(const Vec3<T> &v) const {
      return X == v.X && Y == v.Y && Z == v.Z;
    }

    __host__ __device__ bool operator!=(const Vec3<T> &other) const {
      return !(*this == other);
    }

    __host__ __device__ static inline T Dot(const Vec3<T> &v0,
                                            const Vec3<T> &v1) {
      return (v0.X * v1.X + v0.Y * v1.Y + v0.Z * v1.Z);
    }

    __host__ __device__ static inline Vec3<T>
    Cross(const Vec3<T> &v0, const Vec3<T> &v1) {
      float resX_pos = (v0.Y * v1.Z) - (v0.Z * v1.Y);
      float resY_pos = (v0.Z * v1.X) - (v0.X * v1.Z);
      float resZ_pos = (v0.X * v1.Y) - (v0.Y * v1.X);

      return {resX_pos, resY_pos, resZ_pos};
    }
};

using Position = Vec3<float>;
using Normal = Vec3<float>;


struct Mesh 
{
    std::vector<uint32_t> FacesCoords;
    std::vector<uint32_t> FacesNormals;

    std::vector<Position> Coords;
    std::vector<Normal> Normals;
    std::vector<Color> Colors;

    void VerticesReserve(const size_t size) { 
        Coords.reserve(size); Normals.reserve(size); Colors.reserve(size);
    }

    void FacesReserve(const size_t size) {
        FacesCoords.reserve(size * 6); FacesNormals.reserve(size * 6);
    }

    void ShrinkToFit() {
        FacesCoords.shrink_to_fit(); FacesNormals.shrink_to_fit();
        Coords.shrink_to_fit(); Normals.shrink_to_fit(); Colors.shrink_to_fit();
    }

    void Clear() {
        FacesCoords.clear(); FacesNormals.clear();
        Coords.clear(); Normals.clear(); Colors.clear();
    }

    inline size_t VerticesSize() const { return Coords.size(); }

    inline size_t NormalsSize() const { return Normals.size(); }
    
    inline size_t FacesSize() const { return FacesCoords.size() / 6; }
}; 

#endif // !VERTEX_H
