/* Serial CPU application computing the blur filtering of a bitmap image */

#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include "utils.h"

#define BLUR_SIZE 10 // blur filter neighbors in each dimension

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

void FlipImage(pel *img_src, pel *img_dst, img_Descr *img_d) {
    uint h = img_d->height;
    uint w = img_d->width;
    uint wbytes = img_d->rowByte;
    for (int iy=0; iy<h; iy++) { // index of the row
        for (int ix=0; ix<w; ix++) { // index of the column
            unsigned int avg_r = 0;
            unsigned int avg_g = 0;
            unsigned int avg_b = 0;
            for (int iyy = iy - BLUR_SIZE; iyy < iy + BLUR_SIZE + 1; iyy++) {
                for(int ixx = ix - BLUR_SIZE; ixx < ix + BLUR_SIZE + 1; ixx++) {
                    if ((iyy >= 0 && iyy < h) && (ixx >= 0 && ixx < w)) {
                        unsigned int offset_neighbors = (iyy * wbytes) + ixx * 3;
                        avg_r += img_src[offset_neighbors];
                        avg_g += img_src[offset_neighbors + 1];
                        avg_b += img_src[offset_neighbors + 2];                     
                    }
                }
            }
            unsigned int offset = (iy * wbytes) + ix * 3;
            unsigned int n_neighbors = (2*BLUR_SIZE + 1)*(2*BLUR_SIZE + 1);
            img_dst[offset] = (unsigned char)(avg_r/n_neighbors);
            img_dst[offset + 1] = (unsigned char)(avg_g/n_neighbors);
            img_dst[offset + 2] = (unsigned char)(avg_b/n_neighbors);        
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
