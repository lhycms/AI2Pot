#ifndef AI2POT_MTP_UTILITIES_CUH
#define AI2POT_MTP_UTILITIES_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>


#ifdef __INTELLISENSE__
#define KERNEL_ARG2(grid_size, block_size)
#define KERNEL_ARG3(grid_size, block_size, sh_mem)
#define KERNEL_ARG4(grid_size, block_size, sh_mem, stream)
#else
#define KERNEL_ARG2(grid_size, block_size) <<<grid_size, block_size>>>
#define KERNEL_ARG3(grid_size, block_size, sh_mem) <<<grid_size, block_size, sh_mem>>>
#define KERNEL_ARG4(grid_size, block_size, sh_mem, stream) <<<grid_size, block_size, sh_mem, stream>>>
#endif

#define CHECK(call)                                                     \
do {                                                                    \
    cudaError_t error_code = call;                                      \
    if (error_code != cudaSuccess)                                      \
    {                                                                   \
        printf("CUDA Error:\n");                                        \
        printf("1. File : %s\n", __FILE__);                             \
        printf("2. Function : %s\n", __FUNCTION__);                     \
        printf("3. Line : %d\n", __LINE__);                             \
        printf("4. Error code : %d\n", error_code);                     \
        printf("5. Error text : %s\n", cudaGetErrorString(error_code)); \
    }                                                                   \
} while(0);


const int MAX_CHEBYSHEV_SIZE = 20;
const int MAX_NUM_MTP_BASIS = 2445;
const int ALPHA_MOMENTS_COUNT = 17366;
#if MAX_MTP_LEVEL == 2
MAX_NUM_MTP_BASIS = 1;
ALPHA_MOMENTS_COUNT = 1;
#elif MAX_MTP_LEVEL == 4
MAX_NUM_MTP_BASIS = 2;
ALPHA_MOMENTS_CONUT = 2;
#elif MAX_MTP_LEVEL == 6
MAX_NUM_MTP_BASIS = 5;
ALPHA_MOMENTS_COUNT = 8;
#elif MAX_MTP_LEVEL == 8
MAX_NUM_MTP_BASIS = 9;
ALPHA_MOMENTS_COUNT = 18;
#elif MAX_MTP_LEVEL == 10
MAX_NUM_MTP_BASIS = 16;
ALPHA_MOMENTS_COUNT = 41;
#elif MAX_MTP_LEVEL == 12
MAX_NUM_MTP_BASIS = 29;
ALPHA_MOMENTS_COUNT = 84;
#elif MAX_MTP_LEVEL == 14
MAX_NUM_MTP_BASIS = 52;
ALPHA_MOMENTS_COUNT = 174;
#elif MAX_MTP_LEVEL == 16
MAX_NUM_MTP_BASIS = 92;
ALPHA_MOMENTS_COUNT = 130;
#elif MAX_MTP_LEVEL == 18
MAX_NUM_MTP_BASIS = 163;
ALPHA_MOMENTS_COUNT = 718;
#elif MAX_MTP_LEVEL == 20
MAX_NUM_MTP_BASIS = 288;
ALPHA_MOMENTS_COUNT = 1352;
#elif MAX_MTP_LEVEL == 22
MAX_NUM_MTP_BASIS = 500;
ALPHA_MOMENTS_COUNT = 2621;
#elif MAX_MTP_LEVEL == 24
MAX_NUM_MTP_BASIS = 864;
ALPHA_MOMENTS_COUNT = 4991;
#elif MAX_MTP_LEVEL == 26
MAX_NUM_MTP_BASIS = 1464;
ALPHA_MOMENTS_COUNT = 9396;
#else //#elif MAX_MTP_LEVEL == 28
MAX_NUM_MTP_BASIS = 2445;
ALPHA_MOMENTS_COUNT = 17366;
#endif

#endif