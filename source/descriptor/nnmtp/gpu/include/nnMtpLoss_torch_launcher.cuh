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
    CoordType *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *d_betot_ml,
    CoordType *d_betot_dft,
    CoordType (*d_bforce_ml)[3],
    CoordType (*d_bforce_dft)[3],
    CoordType *d_bvirial_ml,
    CoordType *d_bvirial_dft)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad- 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    
    find_loss_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_bloss_ptr,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        e_weight,
        f_weight,
        v_weight,
        d_betot_ml,
        d_betot_dft,
        d_bforce_ml,
        d_bforce_dft,
        d_bvirial_ml,
        d_bvirial_dft);
    

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}


// 2. find_ef_loss_torch_launcher()
template <typename CoordType>
__host__
void find_ef_loss_torch_launcher(
    CoordType *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *d_betot_ml,
    CoordType *d_betot_dft,
    CoordType (*d_bforce_ml)[3],
    CoordType (*d_bforce_dft)[3])
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_ef_loss_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_bloss_ptr,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        e_weight,
        f_weight,
        d_betot_ml,
        d_betot_dft,
        d_bforce_ml,
        d_bforce_dft);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}


// 3. find_loss_backward_torch_launcher()
template <typename CoordType>
__host__
void find_loss_backward_torch_launcher(
    CoordType *d_bloss_der2coeffs,
    CoordType *d_bloss_der2w0,
    CoordType *d_bloss_der2w1,
    CoordType *d_bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *d_betot_ml,
    CoordType *d_betot_dft,
    CoordType (*d_bforce_ml)[3],
    CoordType (*d_bforce_dft)[3],
    CoordType *d_bvirial_ml,
    CoordType *d_bvirial_dft,
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    CoordType (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *d_q_scaler)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_loss_backward_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_bloss_der2coeffs,
        d_bloss_der2w0,
        d_bloss_der2w1,
        d_bloss_der2type_bias,
        e_weight,
        f_weight,
        v_weight,
        d_betot_ml,
        d_betot_dft,
        d_bforce_ml,
        d_bforce_dft,
        d_bvirial_ml,
        d_bvirial_dft,
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
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        d_bnumneigh,
        d_bfirstneigh,
        d_brcs,
        d_btypes,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        d_q_scaler);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}


// 4. find_ef_loss_backward_torch_launcher()
template <typename CoordType>
__host__
void find_ef_loss_backward_torch_launcher(
    CoordType *d_bloss_der2coeffs,
    CoordType *d_bloss_der2w0,
    CoordType *d_bloss_der2w1,
    CoordType *d_bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *d_betot_ml,
    CoordType *d_betot_dft,
    CoordType (*d_bforce_ml)[3],
    CoordType (*d_bforce_dft)[3],
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    CoordType (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *d_q_scaler)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_ef_loss_backward_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_bloss_der2coeffs,
        d_bloss_der2w0,
        d_bloss_der2w1,
        d_bloss_der2type_bias,
        e_weight,
        f_weight,
        d_betot_ml,
        d_betot_dft,
        d_bforce_ml,
        d_bforce_dft,
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
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        d_bnumneigh,
        d_bfirstneigh,
        d_brcs,
        d_btypes,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        d_q_scaler);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}

};  // namespace : nnmtp
};  // namespace : ai2pot

#endif