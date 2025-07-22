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

#ifndef AI2POT_NNMTP_NNMTP_LOSS_TORCH_LAUNCHER_CUH
#define AI2POT_NNMTP_NNMTP_LOSS_TORCH_LAUNCHER_CUH

#include "./nnMtpLoss_torch_launcher.h"
#include "./nnMtpLoss.cuh"

namespace ai2pot {
namespace nnmtp {

// 1. find_loss_torch_launcher()
template <typename CoordType>
__host__
void find_loss_torch_launcher(
    CoordType *d_loss_ptr,
    int inum,
    int *d_ilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*d_force_ml)[3],
    CoordType (*d_force_dft)[3],
    CoordType *d_virial_ml,
    CoordType *d_virial_dft)
{
    int block_size_x = 64;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    
    find_loss_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_loss_ptr,
        inum,
        d_ilist,
        e_weight,
        f_weight,
        v_weight,
        etot_ml,
        etot_dft,
        d_force_ml,
        d_force_dft,
        d_virial_ml,
        d_virial_dft);
    

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}


// 2. find_ef_loss_torch_launcher()
template <typename CoordType>
__host__
void find_ef_loss_torch_launcher(
    CoordType *d_loss_ptr,
    int inum,
    int *d_ilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*d_force_ml)[3],
    CoordType (*d_force_dft)[3])
{
    int block_size_x = 64;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_ef_loss_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_loss_ptr,
        inum,
        d_ilist,
        e_weight,
        f_weight,
        etot_ml,
        etot_dft,
        d_force_ml,
        d_force_dft);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}


// 3. find_loss_backward_torch_launcher()
template <typename CoordType>
__host__
void find_loss_backward_torch_launcher(
    CoordType *d_loss_der2coeffs,
    CoordType *d_loss_der2w0,
    CoordType *d_loss_der2w1,
    CoordType *d_loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*d_force_ml)[3],
    CoordType (*d_force_dft)[3],
    CoordType *d_virial_ml,
    CoordType *d_virial_dft,
    int chebyshev_size,
    int num_neurons,
    CoordType *d_coeffs,
    CoordType *d_w0,
    CoordType *d_w1,
    CoordType *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    CoordType (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    int block_size_x = 64;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_loss_backward_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_loss_der2coeffs,
        d_loss_der2w0,
        d_loss_der2w1,
        d_loss_der2type_bias,
        e_weight,
        f_weight,
        v_weight,
        etot_ml,
        etot_dft,
        d_force_ml,
        d_force_dft,
        d_virial_ml,
        d_virial_dft,
        chebyshev_size,
        num_neurons,
        d_coeffs,
        d_w0,
        d_w1,
        d_type_bias,
        alpha_moments_count,
        alpha_index_basic_count,
        d_alpha_index_basic,
        alpha_index_times_count,
        d_alpha_index_times,
        alpha_scalar_moments,
        d_alpha_moment_mapping,
        nmus,
        inum,
        d_ilist,
        d_numneigh,
        d_firstneigh,
        d_rcs,
        d_types,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}


// 4. find_ef_loss_backward_torch_launcher()
template <typename CoordType>
__host__
void find_ef_loss_backward_torch_launcher(
    CoordType *d_loss_der2coeffs,
    CoordType *d_loss_der2w0,
    CoordType *d_loss_der2w1,
    CoordType *d_loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*d_force_ml)[3],
    CoordType (*d_force_dft)[3],
    int chebyshev_size,
    int num_neurons,
    CoordType *d_coeffs,
    CoordType *d_w0,
    CoordType *d_w1,
    CoordType *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    CoordType (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    int block_size_x = 64;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_ef_loss_backward_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_loss_der2coeffs,
        d_loss_der2w0,
        d_loss_der2w1,
        d_loss_der2type_bias,
        e_weight,
        f_weight,
        etot_ml,
        etot_dft,
        d_force_ml,
        d_force_dft,
        chebyshev_size,
        num_neurons,
        d_coeffs,
        d_w0,
        d_w1,
        d_type_bias,
        alpha_moments_count,
        alpha_index_basic_count,
        d_alpha_index_basic,
        alpha_index_times_count,
        d_alpha_index_times,
        alpha_scalar_moments,
        d_alpha_moment_mapping,
        nmus,
        inum,
        d_ilist,
        d_numneigh,
        d_firstneigh,
        d_rcs,
        d_types,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}

};  // namespace : nnmtp
};  // namespace : ai2pot

#endif