#include <cstdio>
#include <string>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#ifndef CUDA_UTILS
#define CUDA_UTILS

inline std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
    return ss.str();
}

#define LOG_INTERNAL(level_str, format, ...)                               \
{                                                                          \
    fprintf(stderr, "[%s] [%s] [%s:%d] " format "\n",                      \
            getCurrentTimestamp().c_str(), level_str, __FILE__, __LINE__,  \
            ##__VA_ARGS__);                                                \
}

#define LOG_ERROR(format, ...) LOG_INTERNAL("ERROR", format, ##__VA_ARGS__)

#define LOG_WARN(format, ...) LOG_INTERNAL("WARN", format, ##__VA_ARGS__)

#define LOG_INFO(format, ...) LOG_INTERNAL("INFO", format, ##__VA_ARGS__)

#define LOG_DEBUG(format, ...) LOG_INTERNAL("DEBUG", format, ##__VA_ARGS__)


inline void gpuAssert(cudaError_t code)
{
    if (code != cudaSuccess) {
        LOG_ERROR("CUDA Assert Code: %s", cudaGetErrorString(code));
        exit(code);
    }
}


#endif
