#ifndef PROC_UTILS_H
#define PROC_UTILS_H

#include <string>

enum class Types {
    SEQUENTIAL, NAIVE, TILED
};

__host__ inline int NextPow2(const int n, const int max) {
    int pow2 = 1;
    while (pow2 < n && pow2 < max)
        pow2 <<= 1;
    return pow2;
}

__host__ inline std::string GetFilename(const std::string path) {
  size_t pos = path.find_last_of('/');
  if (pos != std::string::npos)
    return path.substr(pos + 1);
  else
    return path;
}

__host__ inline std::string GetTypesString(const Types type) {
    switch (type) {
        case Types::SEQUENTIAL: return "sequential";
        case Types::NAIVE:      return "naive";
        case Types::TILED:      return "tiled";
        default:                return "Unknown";
    }
}


#endif // !PROC_UTILS_H
