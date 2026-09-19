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
#include <stdio.h>


__constant__ double COVALENT_RADIUS[94] = {
  0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,    0.853333,
  0.893333, 1.86667,  1.66667, 1.50667, 1.38667, 1.46667, 1.36,     1.32,    1.28,
  2.34667,  2.05333,  1.77333, 1.62667, 1.61333, 1.46667, 1.42667,  1.38667, 1.33333,
  1.32,     1.34667,  1.45333, 1.49333, 1.45333, 1.53333, 1.46667,  1.52,    1.56,
  2.52,     2.22667,  1.96,    1.85333, 1.76,    1.65333, 1.53333,  1.50667, 1.50667,
  1.44,     1.53333,  1.64,    1.70667, 1.68,    1.68,    1.64,     1.76,    1.74667,
  2.78667,  2.34667,  2.16,    1.96,    2.10667, 2.09333, 2.08,     2.06667, 2.01333,
  2.02667,  2.01333,  2.0,     1.98667, 1.98667, 1.97333, 2.04,     1.94667, 1.82667,
  1.74667,  1.64,     1.57333, 1.54667, 1.48,    1.49333, 1.50667,  1.76,    1.73333,
  1.73333,  1.81333,  1.74667, 1.84,    1.89333, 2.68,    2.41333,  2.22667, 2.10667,
  2.02667,  2.04,     2.05333, 2.06667};


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


template <typename CoordType>
__device__
void find_revised_rmax_rmin(
    CoordType &zbl_rmax,
    CoordType &zbl_rmin,
    CoordType zbl_typewise_factor,
    int Zi,
    int Zj)
{
    CoordType old_zbl_rmax = zbl_rmax;
    CoordType typewise_zbl_rmax = zbl_typewise_factor
                                  * (COVALENT_RADIUS[Zi] + COVALENT_RADIUS[Zj]);
    if (typewise_zbl_rmax <= old_zbl_rmax) {
        zbl_rmax = typewise_zbl_rmax;
        zbl_rmin = 0.0;
    } else {
        zbl_rmax = old_zbl_rmax;
        zbl_rmin = old_zbl_rmax / 2.0;
    }
}


#endif
