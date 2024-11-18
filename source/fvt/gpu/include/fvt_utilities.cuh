#ifndef AI2POT_FVT_UTILITIES_CUH
#define AI2POT_FVT_UTILITIES_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
#define KERNEL_ARG2(grid_size, block_size)
#define KERNEL_ARG3(grid_size, block_size, sh_mem)
#define KERNEL_ARG4(grid_size, block_size, sh_mem, stream)
#else
#define KERNEL_ARG2(grid_size, block_size) <<<grid_size, block_size>>>
#define KERNEL_ARG3(grid_size, block_size, sh_mem) <<<grid_size, block_size, sh_mem>>>
#define KERNEL_ARG4(grid_size, block_size, sh_mem, stream) <<<grid_size, block_size, sh_mem, stream>>>
#endif


#define CHECK(call)                                                                 \
do                                                                                  \
{                                                                                   \
    cudaError_t error_code = call;                                                  \
    if (error_code != cudaSuccess)                                                  \
    {                                                                               \
        printf("CUDA Error:\n");                                                    \
        printf("\t1. File: %s\n", __FILE__);                                        \
        printf("\t2. Function: %s\n", __FUNCTION__);                                \
        printf("\t3. Line: %d\n", __LINE__);                                        \
        printf("\t4. Error code: %d\n", error_code);                                \
        printf("\t5. Error text: %s\n", cudaGetErrorString(error_code));            \
    }                                                                               \
} while(0);


const int MAX_NNEI = 128;
const int BLOCK_SIZE_X = 256;

#endif
