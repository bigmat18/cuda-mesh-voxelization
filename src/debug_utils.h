#include <cstdio>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#ifndef CUDA_UTILS
#define CUDA_UTILS

__host__ __device__ inline const char* getCurrentTimestamp() {
    #ifndef __CUDA_ARCH__
        static char timestamp_buffer[64];

        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::strftime(timestamp_buffer, sizeof(timestamp_buffer),
                "%Y-%m-%d %X", std::localtime(&in_time_t));
        return timestamp_buffer;
    #else
        return "DEVICE";
    #endif
}

#if LOGGING 
#define LOG_INTERNAL(level_str, format, ...)                               \
{                                                                          \
    printf("[%s] [%s] [%s:%d] " format "\n",                               \
           getCurrentTimestamp(), level_str, __FILE__, __LINE__,  \
           ##__VA_ARGS__);                                                \
}
#else 
#define LOG_INTERNAL(level_str, format, ...)
#endif // LOGGING 

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

inline void cpuAssert(bool condition, const std::string msg = "No error msg")
{
    if (!condition) { 
        LOG_ERROR("CPU Assert: %s", msg.c_str());
        exit(-1);
    }
}


#endif
