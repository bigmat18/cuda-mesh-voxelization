/* Utility structs and functions for the ImageBLUR application */

#ifndef _BMP_UTILITY_H_
#define _BMP_UTILITY_H_

struct img_Descr
{
    int width; // in pixels
    int height; // in pixels
    unsigned char headInfo[54]; //header
    unsigned long int rowByte; // in bytes
};

#define WIDTHB img.rowByte
#define WIDTH img.width
#define HEIGHT img.height
#define IMAGESIZE (WIDTHB * HEIGHT)

struct pixel
{
    unsigned char R;
    unsigned char G;
    unsigned char B;
};

typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned char pel; // pixel element

// read a 24-bit/pixel BMP file into a 1D linear array
pel *read_bmpFile(char *filename, struct img_Descr *img_d)
{
    pel *img;
    FILE *f = fopen(filename, "rb");
    if (f == NULL) {
        printf("File not found\n");
        exit(EXIT_FAILURE);
    }
    size_t nByte = fread(img_d->headInfo, sizeof(pel), 54, f); // read the 54-byte header
    // extract image height and width from header
    int width = *((int *) &img_d->headInfo[18]);
    img_d->width = width;
    int height = *((int *) &img_d->headInfo[22]);
    img_d->height = height;
    int s = (width * 3 + 3) & (~3);  // row is multiple of 4 pixel
    img_d->rowByte = s;
    printf("Input file name: %s  (%d x %d), file size=%lu\n", filename, img_d->width, img_d->height, img_d->rowByte * img_d->height);
    img = (pel *) malloc(img_d->rowByte * img_d->height); // allocate memory to store the main image (1 Dimensional array)
    if (img == NULL) {
        return img; // cannot allocate memory
    }
    size_t out = fread(img, sizeof(pel), img_d->rowByte * img_d->height, f); // read the image from disk
    fclose(f);
    return img;
}

// write the 1D linear-memory stored image into file
void write_bmpFile(pel *img, char *filename, struct img_Descr *img_d)
{
    FILE *f = fopen(filename, "wb");
    if (f == NULL) {
        printf("File cannot be created\n");
        exit(1);
    }
    fwrite(img_d->headInfo, sizeof(pel), 54, f); // write header
    fwrite(img, sizeof(pel), img_d->rowByte * img_d->height, f); // write data
    printf("Output file name: %s  (%d x %d), file Size=%lu\n", filename, img_d->width, img_d->height, img_d->rowByte * img_d->height);
    fclose(f);
}

#endif
