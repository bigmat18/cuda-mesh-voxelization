/* Serial CPU application computing the horizontal flip of a bitmap image */

#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include "utils.h"

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

void FlipImage(pel *img_src, pel *img_dst, img_Descr *img_d) {
    uint h = img_d->height;
    uint w = img_d->width;
    for (int iy=0; iy<h; iy++) { // index of the row
        for (int ix=0; ix<w; ix++) { // index of the column
            unsigned int offset_src = (iy * img_d->rowByte) + ix * 3;
            unsigned int ixx = w - ix - 1;
            unsigned int offset_dst = (iy * img_d->rowByte) + ixx * 3;
            img_dst[offset_dst] = img_src[offset_src];
            img_dst[offset_dst + 1] = img_src[offset_src + 1];
            img_dst[offset_dst + 2] = img_src[offset_src + 2];          
        }
    }
}

int main(int argc, char **argv)
{
    pel *imgSrc, *imgDst;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <InputFilename> <OutputFilename>" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create host memory to store the input and output images
    struct img_Descr img_d_source;
    imgSrc = read_bmpFile(argv[1], &img_d_source);
    if (imgSrc == NULL) {
        std::cout << "Error reading input image" << std::endl;
        exit(EXIT_FAILURE);
    }
    imgDst = (pel *) malloc(img_d_source.rowByte * img_d_source.height);
    if (imgDst == NULL) {
        free(imgSrc);
        std::cout << "Error allocation destination image" << std::endl;
        exit(EXIT_FAILURE);
    }

    uint64_t initial_time = current_time_nsecs();

    // host computation
    FlipImage(imgSrc, imgDst, &img_d_source);

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // Write the flipped image back to disk
    write_bmpFile(imgDst, argv[2], &img_d_source);

    // Deallocate memory
    free(imgSrc);
    free(imgDst);   
    return 0;
}
