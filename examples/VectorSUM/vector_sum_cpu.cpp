/* Serial CPU application computing the sum of two integer arrays */

#include<iostream>
#include<stdlib.h>

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl; 
        exit(1);
    }
    int L = atoi(argv[1]);

    // initialization of the host arrays host_A and host_B
    int *host_A = (int *) malloc(sizeof(int) * L);
    int *host_B = (int *) malloc(sizeof(int) * L);
    int *host_C = (int *) malloc(sizeof(int) * L);

    for (int i=0; i<L; i++) {
        host_A[i] = rand() % 100;
        host_B[i] = rand() % 100;
    }

    uint64_t initial_time = current_time_nsecs();

    for (int i=0; i<L; i++) {
        host_C[i] = host_A[i] + host_B[i];
    }

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // Deallocate host memory
    free(host_A);
    free(host_B);
    free(host_C);
    return 0;
}
