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

#ifndef AI2POT_MTPR_SCHMIDT_ORTH_CUH
#define AI2POT_MTRP_SCHMIDT_ORTH_CUH

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

#include "./mtp_utilities.cuh"

namespace ai2pot {
namespace mtpr {


template <typename CoordType>
static __global__
void find_coeffs_orthogonal_basis_set_kernel(
    CoordType *orth_coeffs,
    int ntypes,
    int nmus,
    int chebyshev_size,
    CoordType *coeffs)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nx == 0) {
        for (int type_central=0; type_central<ntypes; type_central++) {
            for (int type_outer=0; type_outer<ntypes; type_outer++) {
                for (int mu=0; mu<nmus; mu++) {
                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                                  + mu*chebyshev_size + xi;
                        orth_coeffs[idx] = coeffs[idx];
                    }
                }
            }
        }

        for (int k1=0; k1<nmus; k1++) {
            // 1.
            for (int k2=0; k2<k1; k2++) {
                CoordType norm2 = 0.0;
                CoordType scal = 0.0;
                for (int type_central=0; type_central<ntypes; type_central++) {
                    for (int type_outer=0; type_outer<ntypes; type_outer++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            int idx1 = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                                       + k1*chebyshev_size + xi;
                            int idx2 = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                                       + k2*chebyshev_size + xi;
                            norm2 += orth_coeffs[idx2] * orth_coeffs[idx2];
                            scal += orth_coeffs[idx1] * orth_coeffs[idx2];
                        }
                    }
                }

                for (int type_central=0; type_central<ntypes; type_central++) {
                    for (int type_outer=0; type_outer<ntypes; type_outer++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            int idx1 = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                                       + k1*chebyshev_size + xi;
                            int idx2 = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                                       + k2*chebyshev_size + xi;
                            orth_coeffs[idx1] -= orth_coeffs[idx2] * scal / norm2;
                        }
                    }
                }
            }

            // 2.
            CoordType norm2 = 1e-10;
            for (int type_central=0; type_central<ntypes; type_central++) {
                for (int type_outer=0; type_outer<ntypes; type_outer++) {
                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                                  + k1*chebyshev_size + xi;
                        norm2 += orth_coeffs[idx] * orth_coeffs[idx];
                    }
                }
            }
            CoordType norm = std::sqrt(norm2);
            for (int type_central=0; type_central<ntypes; type_central++) {
                for (int type_outer=0; type_outer<ntypes; type_outer++) {
                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                                  + k1*chebyshev_size + xi;
                        orth_coeffs[idx] /= norm;
                    }
                }
            }
        }

    }
}


template <typename CoordType>
static __host__
void find_coeffs_orthogonal_basis_set_launcher(
    CoordType *h_orth_coeffs,
    int ntypes,
    int nmus,
    int chebyshev_size,
    CoordType *h_coeffs)
{
    int grid_size_x = 1;
    int block_size_x = 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    
    CoordType *d_orth_coeffs;
    CoordType *d_coeffs;
    CHECK_CUDA_API( cudaMalloc((void**)&d_orth_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*num_coeffs, cudaMemcpyHostToDevice) );

    
    find_coeffs_orthogonal_basis_set_kernel KERNEL_ARG2(grid_size, block_size) (
        d_orth_coeffs,
        ntypes,
        nmus,
        chebyshev_size,
        d_coeffs);
    
    CHECK_CUDA_KERNEL;
    CHECK_CUDA_API( cudaMemcpy(h_orth_coeffs, d_orth_coeffs, sizeof(CoordType)*num_coeffs, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_orth_coeffs) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
}

};  // namespace : mtpr
};  // namespace : ai2pot

#endif
