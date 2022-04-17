#ifndef PATCH_SIZE
#define PATCH_SIZE 77
#endif

#ifndef IMAGE_TYPE
#define IMAGE_TYPE uchar
#endif

#ifndef KERNEL_DATA_TYPE
#define KERNEL_DATA_TYPE float
#endif

#ifndef KERNEL_VEC_SIZE
#define KERNEL_VEC_SIZE
#endif

#define DATA_TYPE_(type, size) type##size
#define DATA_TYPE(type, size) DATA_TYPE_(type, size)
#define stringize(t) #t
#define print_macro(t) stringize(t)

#define KERNEL_TYPE DATA_TYPE(KERNEL_DATA_TYPE, KERNEL_VEC_SIZE)

typedef KERNEL_TYPE convolution_kernel_t;
typedef IMAGE_TYPE input_t;
__kernel void convolve(__global input_t *img, __global convolution_kernel_t *res, __constant convolution_kernel_t *krn, int sz0, int sz1, int krnsz, int numkrn, int numimg)
{
    //printf(""print_macro(IMAGE_TYPE)" ");
    int wid0=get_group_id(0), wid1=get_group_id(1);
    int lid0=get_local_id(0), lid1=get_local_id(1);
    int gid0=get_global_id(0), gid1=get_global_id(1);

    convolution_kernel_t tmp,sum;

    __local convolution_kernel_t locsum;
    int hfs=krnsz/2;
    int addr0, addr1;

    __local input_t cached[PATCH_SIZE*PATCH_SIZE];

    int iimg=get_group_id(2);
    for(int imaddr0=0;imaddr0<PATCH_SIZE;imaddr0++)
    {
        for(int imaddr1=0;imaddr1<PATCH_SIZE;imaddr1++)
        {
            addr0=gid0*(PATCH_SIZE-krnsz)-hfs+imaddr0;
            if(addr0<0)addr0+=sz0;
            else if(addr0>sz0)addr0-=sz0;
            addr1=gid1*(PATCH_SIZE-krnsz)-hfs+imaddr1;
            if(addr1<0)addr1+=sz1;
            else if(addr1>sz1)addr1-=sz1;
            cached[imaddr0*PATCH_SIZE+imaddr1]=img[iimg*sz0*sz1+addr0*sz1+addr1];

        }
    }
    for(int imaddr0=hfs;imaddr0<PATCH_SIZE-hfs;imaddr0++)
    {
        for(int imaddr1=hfs;imaddr1<PATCH_SIZE-hfs;imaddr1++)
        {
            if((imaddr0+gid0*(PATCH_SIZE-krnsz)-hfs)<sz0&&(imaddr1+gid1*(PATCH_SIZE-krnsz)-hfs)<sz1)
            {
                sum=(convolution_kernel_t)(0);
                for(int i=-hfs;i<hfs; i++)
                {
                    for(int j=-hfs;j<hfs; j++)
                    {
                        addr0=imaddr0+i;
                        addr1=imaddr1+j;
                        tmp=cached[addr0*PATCH_SIZE+addr1]*krn[(i+hfs)*krnsz+j+hfs];
                        sum=sum+tmp;
                    }
                }
                res[iimg*sz0*sz1+(imaddr0+gid0*(PATCH_SIZE-krnsz)-hfs)*sz1+(imaddr1+gid1*(PATCH_SIZE-krnsz)-hfs)]=sum;
            }
        }
    }
}