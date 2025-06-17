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

#ifndef AI2POT_LINEAR_MTP_TORCH_LAUNCHER_CUH
#define AI2POT_LINEAR_MTP_TORCH_LAUNCHER_CUH
#include "./linearMtp_torch_launcher.h"
#include "./linearMtp.cuh"

namespace ai2pot {
namespace mtpr {


template <typename CoordType>
__host__
void find_efv_torch_launcher(
    CoordType *d_etot_ptr,
    CoordType (*d_force)[3],
    CoordType *d_virial,
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

    find_efv_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_etot_ptr,
        d_force,
        d_virial,
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


template <typename CoordType>
__host__
void find_ef_torch_launcher(
    CoordType *d_etot_ptr,
    CoordType (*d_force)[3],
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

    find_ef_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_etot_ptr,
        d_force,
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
        inum,
        d_ilist,
        d_numneigh,
        d_firstneigh,
        d_rcs,
        d_types,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms,nghost,
        rmax,
        rmin);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}


};  // namespace : mtpr
};  // namespace : ai2pot

#endif
