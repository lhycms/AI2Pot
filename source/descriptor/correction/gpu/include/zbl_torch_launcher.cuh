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

#ifndef AI2POT_ZBL_TORCH_LAUNCHER_CUH
#define AI2POT_ZBL_TORCH_LAUNCHER_CUH
#include "./zbl_torch_launcher.h"
#include "./zbl_utilities.cuh"
#include "./zbl.cuh"

namespace ai2pot {
namespace correction {

template <typename CoordType>
void correct_zbl_efv_torch_launcher(
    CoordType *d_etot_ptr,
    CoordType *d_force,
    CoordType *d_virial,
    CoordType rmax,
    CoordType rmin,
    CoordType *d_cks,
    CoordType *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    CoordType (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms)
{
    int block_size_x = 64;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    correct_zbl_efv_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_etot_ptr,
        d_force,
        d_virial,
        rmax,
        rmin,
        d_cks,
        d_dks,
        inum,
        d_ilist,
        d_numneigh,
        d_firstneigh,
        d_rcs,
        d_types,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}


template <typename CoordType>
void correct_zbl_ef_torch_launcher(
    CoordType *d_etot_ptr,
    CoordType *d_force,
    CoordType rmax,
    CoordType rmin,
    CoordType *d_cks,
    CoordType *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    CoordType (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms)
{
    int block_size_x = 64;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    correct_zbl_ef_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_etot_ptr,
        d_force,
        rmax,
        rmin,
        d_cks,
        d_dks,
        inum,
        d_ilist,
        d_numneigh,
        d_firstneigh,
        d_rcs,
        d_types,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
}

};  // namespace : correction
};  // namespace : ai2pot

#endif