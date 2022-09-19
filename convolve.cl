#ifndef PATCH_SIZE
#define PATCH_SIZE 77
#endif

#ifndef IMAGE_TYPE
#define IMAGE_TYPE uchar
#endif

#ifndef KERNEL_TYPE
#define KERNEL_TYPE float16
#endif

typedef KERNEL_TYPE convolution_kernel_t;
typedef IMAGE_TYPE input_t;

__kernel void convolve(__global input_t *img,
                       __global convolution_kernel_t *res,
                       __constant convolution_kernel_t *kernel_array, int size0,
                       int size1, int kernel_array_size) {
    int gid0 = get_global_id(0), gid1 = get_global_id(1);

    convolution_kernel_t tmp, sum;

    int hfs = kernel_array_size / 2;
    int address0, address1;

    __local input_t cached[PATCH_SIZE * PATCH_SIZE];

    int image_index = get_group_id(2);
    for (int image_address0 = 0; image_address0 < PATCH_SIZE; image_address0++) {
        for (int image_address1 = 0; image_address1 < PATCH_SIZE;
            image_address1++) {
                address0 = gid0 * (PATCH_SIZE - kernel_array_size) - hfs + image_address0;
                if (address0 < 0)
                    address0 += size0;
                else if (address0 > size0)
                    address0 -= size0;
                address1 = gid1 * (PATCH_SIZE - kernel_array_size) - hfs + image_address1;
                if (address1 < 0)
                    address1 += size1;
                else if (address1 > size1)
                    address1 -= size1;
                cached[image_address0 * PATCH_SIZE + image_address1] =
                    img[image_index * size0 * size1 + address0 * size1 + address1];
        }
    }
    for (int image_address0 = hfs; image_address0 < PATCH_SIZE - hfs; image_address0++) {
        for (int image_address1 = hfs; image_address1 < PATCH_SIZE - hfs; image_address1++) {
            if ((image_address0 + gid0 * (PATCH_SIZE - kernel_array_size) - hfs) < size0 &&
                (image_address1 + gid1 * (PATCH_SIZE - kernel_array_size) - hfs) < size1) {
                
                sum = (convolution_kernel_t)(0);
                for (int i = -hfs; i < hfs; i++) {
                    for (int j = -hfs; j < hfs; j++) {
                        address0 = image_address0 + i;
                        address1 = image_address1 + j;
                        tmp = cached[address0 * PATCH_SIZE + address1] *
                            kernel_array[(i + hfs) * kernel_array_size + j + hfs];
                        sum = sum + tmp;
                    }
                }
                res[image_index * size0 * size1 +
                    (image_address0 + gid0 * (PATCH_SIZE - kernel_array_size) - hfs) * size1 +
                    (image_address1 + gid1 * (PATCH_SIZE - kernel_array_size) - hfs)] = sum;
            }
        }
    }
}