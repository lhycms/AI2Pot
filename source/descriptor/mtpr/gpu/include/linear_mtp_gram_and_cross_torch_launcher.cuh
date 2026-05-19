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

#ifndef AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_TORCH_LAUNCHER_CUH
#define AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_TORCH_LAUNCHER_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "./linear_mtp_gram_and_cross_torch_launcher.h"
#include "./linear_mtp_gram_and_cross.cuh"


namespace ai2pot {
namespace mtpr {


template <typename CoordType>
void find_lin_matrix_lin_vector_torch_launcher(
    CoordType *d_lin_matrix,
    CoordType *d_lin_vector,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *d_betot_residual,
    CoordType (*d_bforce_residual)[3],
    CoordType *d_bvirial_residual,
    int chebyshev_size,
    CoordType *d_coeffs,
    CoordType *d_linear_coeffs,
    CoordType *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    CoordType (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *d_q_scaler)
{
    int block_size_x = 128;
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);
    
    // 1. 
    int num_parameters = alpha_scalar_moments + ntypes;
    CoordType *d_benergy_components;
    CoordType *d_bforce_components;
    CoordType *d_bvirial_components;
    CHECK_CUDA_API( cudaMalloc((void**)&d_benergy_components, sizeof(CoordType)*batch_size*num_parameters) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_components, sizeof(CoordType)*batch_size*natoms_pad*3*num_parameters) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_components, sizeof(CoordType)*batch_size*3*3*num_parameters) );
    CHECK_CUDA_API( cudaMemset(d_benergy_components, 0, sizeof(CoordType)*batch_size*num_parameters) );
    CHECK_CUDA_API( cudaMemset(d_bforce_components, 0, sizeof(CoordType)*batch_size*natoms_pad*3*num_parameters) );
    CHECK_CUDA_API( cudaMemset(d_bvirial_components, 0, sizeof(CoordType)*batch_size*3*3*num_parameters) );

    // 2. 
    find_efv_components_kernel KERNEL_ARG2(grid_size, block_size) (
        d_benergy_components,
        d_bforce_components,
        d_bvirial_components,
        chebyshev_size,
        d_coeffs,
        d_linear_coeffs,
        d_type_bias,
        alpha_moments_count,
        alpha_index_basic_count,
        d_alpha_index_basic,
        alpha_index_times_count,
        d_alpha_index_times,
        alpha_scalar_moments,
        d_alpha_moment_mapping,
        nmus,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        d_bnumneigh,
        d_bfirstneigh,
        d_brcs,
        d_btypes,
        ntypes,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        d_q_scaler);
    CHECK_CUDA_KERNEL;
    find_lin_matrix_lin_vector_kernel KERNEL_ARG2(grid_size, block_size) (
        d_lin_matrix,
        d_lin_vector,
        e_weight,
        f_weight,
        v_weight,
        d_betot_residual,
        d_bforce_residual,
        d_bvirial_residual,
        d_benergy_components,
        d_bforce_components,
        d_bvirial_components,
        alpha_scalar_moments,
        nmus,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        ntypes);
    CHECK_CUDA_KERNEL;
    int mirror_block_size_x = 16;
    int mirror_block_size_y = 16;
    dim3 mirror_block_size(mirror_block_size_x, mirror_block_size_y);
    int mirror_grid_size_x = (num_parameters - 1) / mirror_block_size_x + 1;
    int mirror_grid_size_y = (num_parameters - 1) / mirror_block_size_y + 1;
    dim3 mirror_grid_size(mirror_grid_size_x, mirror_grid_size_y);
    mirror_lin_matrix<CoordType> KERNEL_ARG2(mirror_grid_size, mirror_block_size) (d_lin_matrix, num_parameters);
    CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaFree(d_benergy_components) );
    CHECK_CUDA_API( cudaFree(d_bforce_components) );
    CHECK_CUDA_API( cudaFree(d_bvirial_components) );
}


};  // namespace : mtpr
};  // namespace : ai2pot

#endif
