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

#ifndef AI2POT_FITUTILS_DESCRIPTORS_STATISTICS_TORCH_LAUNCHER_CUH
#define AI2POT_FITUTILS_DESCRIPTORS_STATISTICS_TORCH_LAUNCHER_CUH
#include "./fitutils_utilities.cuh"
#include "./descriptorsStatistics_torch_launcher.h"
#include "./descriptorsStatistics.cuh"


namespace ai2pot {
namespace fitutils {

template <typename CoordType>
__host__
void find_all_type_descriptors_statistics_torch_launcher(
    int *d_natoms_in_batch_ptr,
    CoordType *d_descriptors_mean,
    CoordType *d_descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *d_binum,
    CoordType *d_bdescriptors)
{
    int grid_size_x = descriptor_dim;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    // 1. natoms_in_batch_ptr
    find_natoms_in_batch_kernel KERNEL_ARG2(grid_size, block_size) (
        d_natoms_in_batch_ptr,
        batch_size,
        d_binum);
    CHECK_CUDA_KERNEL;

    // 2. mean
    find_all_type_descriptors_mean_kernel KERNEL_ARG2(grid_size, block_size) (
        d_natoms_in_batch_ptr,
        d_descriptors_mean,
        batch_size,
        natoms_pad,
        descriptor_dim,
        d_binum,
        d_bdescriptors);
    CHECK_CUDA_KERNEL;

    // 3. M2
    find_all_type_descriptors_M2_kernel KERNEL_ARG2(grid_size, block_size) (
        d_natoms_in_batch_ptr,
        d_descriptors_mean,
        d_descriptors_M2,
        batch_size,
        natoms_pad,
        descriptor_dim,
        d_binum,
        d_bdescriptors);
    CHECK_CUDA_KERNEL;
}


};  // namespace : fitutils
};  // namespace : ai2pot
#endif