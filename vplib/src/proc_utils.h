#ifndef PROC_UTILS_H
#define PROC_UTILS_H

__host__ inline int NextPow2(const int n, const int max) {
    int pow2 = 1;
    while (pow2 < n && pow2 < max)
        pow2 <<= 1;
    return pow2;
}

enum class Types {
    SEQUENTIAL, NAIVE, TILED
};

#endif // !PROC_UTILS_H
