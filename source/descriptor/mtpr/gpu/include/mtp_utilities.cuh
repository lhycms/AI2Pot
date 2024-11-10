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


const int MAX_NUM_MTP_BASIS = 2445;
#if MAX_MTP_LEVEL == 2
MAX_NUM_MTP_BASIS = 1;
#elif MAX_MTP_LEVEL == 4
MAX_NUM_MTP_BASIS = 2;
#elif MAX_MTP_LEVEL == 6
MAX_NUM_MTP_BASIS = 5;
#elif MAX_MTP_LEVEL == 8
MAX_NUM_MTP_BASIS = 9;
#elif MAX_MTP_LEVEL == 10
MAX_NUM_MTP_BASIS = 16;
#elif MAX_MTP_LEVEL == 12
MAX_NUM_MTP_BASIS = 29;
#elif MAX_MTP_LEVEL == 14
MAX_NUM_MTP_BASIS = 52;
#elif MAX_MTP_LEVEL == 16
MAX_NUM_MTP_BASIS = 92;
#elif MAX_MTP_LEVEL == 18
MAX_NUM_MTP_BASIS = 163;
#elif MAX_MTP_LEVEL == 20
MAX_NUM_MTP_BASIS = 288;
#elif MAX_MTP_LEVEL == 22
MAX_NUM_MTP_BASIS = 500;
#elif MAX_MTP_LEVEL == 24
MAX_NUM_MTP_BASIS = 864;
#elif MAX_MTP_LEVEL == 26
MAX_NUM_MTP_BASIS = 1464;
#else //#elif MAX_MTP_LEVEL == 28
MAX_NUM_MTP_BASIS = 2445;
#endif

#endif