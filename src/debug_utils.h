#include <cstdio>
#include <cuda_runtime_api.h>
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

__device__ __host__
constexpr std::string_view file_name(std::string_view path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string_view::npos) ? path : path.substr(pos + 1);
}

#define __FILENAME__ file_name(__FILE__).data()

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

inline void gpuAssertBase(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        printf("[%s:%d] CUDA Assert: %s", file, line, cudaGetErrorString(code));
        exit(code);
    }
}

inline void cpuAssertBase(bool condition, const std::string msg,
                          const char* file, int line)
{
    if (!condition) { 
        printf("[%s:%d] CPU Assert: %s", file, line, msg.c_str());
        exit(-1);
    }
}

#define gpuAssert(ans) gpuAssertBase((ans), __FILE__, __LINE__);

#define cpuAssert(ans, msg) cpuAssertBase((ans), msg, __FILE__, __LINE__);

#endif
