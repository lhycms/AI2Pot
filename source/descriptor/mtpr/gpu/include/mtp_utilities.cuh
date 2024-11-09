#ifndef AI2POT_MTP_UTILITIES
#define AI2POT_MTP_UTILITIES
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


template <typename CoordType>
__global__ void find_mtp_basis_val_der_cuda_kernel()
{

}

template <typename CoordType>
void find_mtp_basis_val_der_cuda_launcher()
{

}

#endif