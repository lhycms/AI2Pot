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

#ifndef AI2POT_EFV_RMSE_TORCH_LAUNCHER_CUH
#define AI2POT_EFV_RMSE_TORCH_LAUNCHER_CUH

namespace ai2pot {
namespace fitutils {


// 1. find_num_real_atoms_in_batch
__host__ void find_num_real_atoms_in_batch_torch_launcher(
    int *d_num_real_atoms_in_batch_ptr,
    int batch_size,
    int *d_binum)
{
    int grid_size_x = 32;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_num_real_atoms_in_batch_kernel KERNEL_ARG2(grid_size, block_size) (
        d_num_real_atoms_in_batch_ptr,
        batch_size,
        d_binum);
    CHECK_CUDA_KERNEL;
}


// 2. find_e_se
template <typename CoordType>
void find_e_se_torch_launcher(
    CoordType *d_e_se_ptr,
    int batch_size,
    int *d_binum,
    CoordType *d_betot_ml,
    CoordType *d_betot_dft)
{
    int grid_size_x = 32;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_e_se_kernel KERNEL_ARG2(grid_size, block_size) (
        d_e_se_ptr,
        batch_size,
        d_binum,
        d_betot_ml,
        d_betot_dft);
    CHECK_CUDA_KERNEL;
}


// 3. find_f_se
template <typename CoordType>
void find_f_se_torch_launcher(
    CoordType *d_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    CoordType (*d_bforce_ml)[3],
    CoordType (*d_bforce_dft)[3])
{
    int grid_size_x = 32;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_f_se_kernel KERNEL_ARG2(grid_size, block_size) (
        d_f_se_ptr,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        d_bforce_ml,
        d_bforce_dft);
    CHECK_CUDA_KERNEL;
}


// 4. find_v_se
template <typename CoordType>
void find_v_se_torch_launcher(
    CoordType *d_v_se_ptr,
    int batch_size,
    int *d_binum,
    CoordType *d_bvirial_ml,
    CoordType *d_bvirial_dft)
{
    int grid_size_x = 32;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_v_se_kernel KERNEL_ARG2(grid_size, block_size) (
        d_v_se_ptr,
        batch_size,
        d_binum,
        d_bvirial_ml,
        d_bvirial_dft);
    CHECK_CUDA_KERNEL;
}


};  // namespace : fitutils
};  // namespace : ai2pot


#endif