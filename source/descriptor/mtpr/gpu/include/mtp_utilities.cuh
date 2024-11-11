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
#if MAX_MTP_LEVEL == 2
    #define MAX_NUM_MTP_BASIS 1
    #define ALPHA_MOMENTS_COUNT 1
#elif MAX_MTP_LEVEL == 4
    #define MAX_NUM_MTP_BASIS 2
    #define ALPHA_MOMENTS_CONUT 2
#elif MAX_MTP_LEVEL == 6
    #define MAX_NUM_MTP_BASIS 5
    #define ALPHA_MOMENTS_COUNT 8
#elif MAX_MTP_LEVEL == 8
    #define MAX_NUM_MTP_BASIS 9
    #define ALPHA_MOMENTS_COUNT 18
#elif MAX_MTP_LEVEL == 10
    #define MAX_NUM_MTP_BASIS 16
    #define ALPHA_MOMENTS_COUNT 41
#elif MAX_MTP_LEVEL == 12
    #define MAX_NUM_MTP_BASIS 29
    #define ALPHA_MOMENTS_COUNT 84
#elif MAX_MTP_LEVEL == 14
    #define MAX_NUM_MTP_BASIS 52
    #define ALPHA_MOMENTS_COUNT 174
#elif MAX_MTP_LEVEL == 16
    #define MAX_NUM_MTP_BASIS 92
    #define ALPHA_MOMENTS_COUNT 130
#elif MAX_MTP_LEVEL == 18
    #define MAX_NUM_MTP_BASIS 163
    #define ALPHA_MOMENTS_COUNT 718
#elif MAX_MTP_LEVEL == 20
    #define MAX_NUM_MTP_BASIS 288
    #define ALPHA_MOMENTS_COUNT 1352
#elif MAX_MTP_LEVEL == 22
    #define MAX_NUM_MTP_BASIS 500
    #define ALPHA_MOMENTS_COUNT 2621
#elif MAX_MTP_LEVEL == 24
    #define MAX_NUM_MTP_BASIS 864
    #define ALPHA_MOMENTS_COUNT 4991
#elif MAX_MTP_LEVEL == 26
    #define MAX_NUM_MTP_BASIS 1464
    #define ALPHA_MOMENTS_COUNT 9396
#else //#elif MAX_MTP_LEVEL == 28
    #define MAX_NUM_MTP_BASIS 2445
    #define ALPHA_MOMENTS_COUNT 17366
#endif

#endif