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

#ifndef AI2POT_ZBL_UTILITIES_CUH
#define AI2POT_ZBL_UTILITIES_CUH

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


#define CHECK_CUDA_API(api_func)                                                            \
do {                                                                                        \
    cudaError_t error_code = api_func;                                                      \
    if (error_code != cudaSuccess)                                                          \
    {                                                                                       \
        printf("CUDA Error:\n");                                                            \
        printf("\t1. File : %s\n", __FILE__);                                               \
        printf("\t2. Function : %s\n", __FUNCTION__);                                       \
        printf("\t3. Line : %d\n", __LINE__);                                               \
        printf("\t4. Error code : %d\n", error_code);                                       \
        printf("\t5. Error text : %s\n", cudaGetErrorString(error_code));                   \
    }                                                                                       \
} while(0);


#endif