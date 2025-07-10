#ifndef PROFILING_H
#define PROFILING_H

#include <chrono>
#include <cstdio>
#include <string>

#if PROFILING

class Profiling 
{
    std::string mMsg;
    std::chrono::high_resolution_clock::time_point mStart;

public:
    Profiling(const std::string msg = "")
        : mMsg(msg), mStart(std::chrono::high_resolution_clock::now()) {}

    ~Profiling()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration<double, std::milli>(end - mStart).count();
        if(!mMsg.empty())
            printf("[PROFILING] %s: %f ms\n", mMsg.c_str(), delta);
        else
            printf("[PROFILING] Delta Time: %f ms\n", delta);
    }
};


#else
#warning "Profiling is not active"
class Profiling {};
#endif

#define PROFILING_SCOPE(msg) Profiling timer##__LINE__(msg)
#endif // !PROFILING_H
