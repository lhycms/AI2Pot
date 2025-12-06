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

#ifndef AI2POT_MTPR_SCHMIDT_ORTH_TORCH_LAUNCHER_CUH
#define AI2POT_MTPR_SCHMIDT_ORTH_TORCH_LAUNCHER_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "./schmidt_orth_torch_launcher.h"
#include "./schmidt_orth.cuh"


namespace ai2pot {
namespace mtpr {

template <typename CoordType>
__host__
void find_coeffs_orthogonal_basis_set_torch_launcher(
    CoordType *d_orth_coeffs,
    int ntypes,
    int nmus,
    int chebyshev_size,
    CoordType *d_coeffs)
{
    int grid_size_x = 1;
    int block_size_x = 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    find_coeffs_orthogonal_basis_set_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_orth_coeffs,
        ntypes,
        nmus,
        chebyshev_size,
        d_coeffs);
    CHECK_CUDA_KERNEL;
}

};  // namespace : mtpr
};  // namespace : ai2pot


#endif