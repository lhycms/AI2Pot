/*
    Copyright 2025 Hanyu Liu
    This file is part of AI2Pot.
    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.
*/

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


#define CHECK_CUDA_API(call)                                                \
do {                                                                        \
    cudaError_t error_code = call;                                          \
    if (error_code != cudaSuccess)                                          \
    {                                                                       \
        printf("CUDA Error:\n");                                            \
        printf("\t1. File : %s\n", __FILE__);                               \
        printf("\t2. Function : %s\n", __FUNCTION__);                       \
        printf("\t3. Line : %d\n", __LINE__);                               \
        printf("\t4. Error code : %d\n", error_code);                       \
        printf("\t5. Error text : %s\n", cudaGetErrorString(error_code));   \
    }                                                                       \
} while(0);


// *****
#define MAX_NUM_MUS 5
#define MAX_ALPHA_INDEX_BASIC 8
#define MAX_NUM_MTP_BASIS 163
#define MAX_ALPHA_MOMENTS_COUNT 718
#define MAX_CHEBYSHEV_SIZE 20
// *****

const int MAX_CHEBYSHEV_SIZE = 20;
const int MAX_NUM_TYPES = 10;
const int MAX_NNEI = 128;
#if MAX_MTP_LEVEL == 2
    #define MAX_NUM_MUS 1
    #define MAX_ALPHA_INDEX_BASIC 1
    #define MAX_NUM_MTP_BASIS 1
    #define MAX_ALPHA_MOMENTS_COUNT 1
#elif MAX_MTP_LEVEL == 4
    #define MAX_NUM_MUS 1
    #define MAX_ALPHA_INDEX_BASIC 1
    #define MAX_NUM_MTP_BASIS 2
    #define MAX_ALPHA_MOMENTS_COUNT 2
#elif MAX_MTP_LEVEL == 6
    #define MAX_NUM_MUS 2
    #define MAX_ALPHA_INDEX_BASIC 2
    #define MAX_NUM_MTP_BASIS 5
    #define MAX_ALPHA_MOMENTS_COUNT 8
#elif MAX_MTP_LEVEL == 8
    #define MAX_NUM_MUS 2
    #define MAX_ALPHA_INDEX_BASIC 3
    #define MAX_NUM_MTP_BASIS 9
    #define MAX_ALPHA_MOMENTS_COUNT 18
#elif MAX_MTP_LEVEL == 10
    #define MAX_NUM_MUS 3
    #define MAX_ALPHA_INDEX_BASIC 4
    #define MAX_NUM_MTP_BASIS 16
    #define MAX_ALPHA_MOMENTS_COUNT 41
#elif MAX_MTP_LEVEL == 12
    #define MAX_NUM_MUS 3
    #define MAX_ALPHA_INDEX_BASIC 5
    #define MAX_NUM_MTP_BASIS 29
    #define MAX_ALPHA_MOMENTS_COUNT 84
#elif MAX_MTP_LEVEL == 14
    #define MAX_NUM_MUS 4
    #define MAX_ALPHA_INDEX_BASIC 6
    #define MAX_NUM_MTP_BASIS 52
    #define MAX_ALPHA_MOMENTS_COUNT 174
#elif MAX_MTP_LEVEL == 16
    #define MAX_NUM_MUS 4
    #define MAX_ALPHA_INDEX_BASIC 7
    #define MAX_NUM_MTP_BASIS 92
    #define MAX_ALPHA_MOMENTS_COUNT 350
#elif MAX_MTP_LEVEL == 18
    #define MAX_NUM_MUS 5
    #define MAX_ALPHA_INDEX_BASIC 8
    #define MAX_NUM_MTP_BASIS 163
    #define MAX_ALPHA_MOMENTS_COUNT 718
#elif MAX_MTP_LEVEL == 20
    #define MAX_NUM_MUS 5
    #define MAX_ALPHA_INDEX_BASIC 9
    #define MAX_NUM_MTP_BASIS 288
    #define MAX_ALPHA_MOMENTS_COUNT 1352
#elif MAX_MTP_LEVEL == 22
    #define MAX_NUM_MUS 6
    #define MAX_ALPHA_INDEX_BASIC 10
    #define MAX_NUM_MTP_BASIS 500
    #define MAX_ALPHA_MOMENTS_COUNT 2621
#elif MAX_MTP_LEVEL == 24
    #define MAX_NUM_MUS 6
    #define MAX_ALPHA_INDEX_BASIC 11
    #define MAX_NUM_MTP_BASIS 864
    #define MAX_ALPHA_MOMENTS_COUNT 4991
#elif MAX_MTP_LEVEL == 26
    #define MAX_NUM_MUS 6
    #define MAX_ALPHA_INDEX_BASIC 11
    #define MAX_NUM_MTP_BASIS 1464
    #define MAX_ALPHA_MOMENTS_COUNT 9396
#elif MAX_MTP_LEVEL == 28
    #define MAX_NUM_MUS 6
    #define MAX_ALPHA_INDEX_BASIC 11
    #define MAX_NUM_MTP_BASIS 2445
    #define MAX_ALPHA_MOMENTS_COUNT 17366
#endif

#endif