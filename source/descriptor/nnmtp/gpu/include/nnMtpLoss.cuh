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

#ifndef AI2POT_NNMTP_NNMTP_LOSS
#define AI2POT_NNMTP_NNMTP_LOSS

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>


namespace ai2pot {
namespace nnmtp {

template <typename CoordType>
static __global__ 
void find_loss_kernel(
    CoordType *loss_ptr,
    int inum,
    int *ilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    CoordType *virial_ml,
    CoordType *virial_dft);


template <typename CoordType>
static __host__
void find_loss_launcher(
    CoordType *h_loss_ptr,
    int inum,
    int *h_ilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*h_force_ml)[3],
    CoordType (*h_force_dft)[3],
    CoordType *h_virial_ml,
    CoordType *h_virial_dft);


template <typename CoordType>
static __global__
void find_ef_loss_kernel(CoordType &loss,
                         int inum,
                         int *ilist,
                         CoordType e_weight,
                         CoordType f_weight,
                         CoordType etot_ml,
                         CoordType etot_dft,
                         CoordType (*force_ml)[3],
                         CoordType (*force_dft)[3]);


template <typename CoordType>
static __host__
void find_ef_loss_launcher(CoordType *h_loss_ptr,
                           int inum,
                           int *h_ilist,
                           CoordType e_weight,
                           CoordType f_weight,
                           CoordType etot_ml,
                           CoordType etot_dft,
                           CoordType (*h_force_ml)[3],
                           CoordType (*h_force_dft)[3]);






template <typename CoordType>
__global__
void find_loss_kernel(
    CoordType *loss_ptr,
    int inum,
    int *ilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    CoordType *virial_ml,
    CoordType *virial_dft)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = nx;
    
    CoordType f_loss = 0.0;
    if (ii < inum) {
        int center_idx = ilist[ii];
        for (int aa=0; aa<3; aa++)
            f_loss += std::pow(force_ml[center_idx][aa] - force_dft[center_idx][aa], 2);
        f_loss = f_weight / (3*inum) * f_loss;
        atomicAdd(loss_ptr, f_loss);
    }

    if (nx == 0) {
        CoordType v_loss = 0.0;
        for (int aa=0; aa<3; aa++)
            for (int bb=0; bb<3; bb++)
                v_loss += std::pow(virial_ml[aa*3+bb] - virial_dft[aa*3+bb], 2);
        v_loss = v_weight / (9*inum) * v_loss;
        atomicAdd(loss_ptr, v_loss);

        CoordType e_loss = 0.0;
        e_loss = e_weight / inum * std::pow(etot_ml - etot_dft, 2);
        atomicAdd(loss_ptr, e_loss);
    }
}


template <typename CoordType>
void find_loss_launcher(CoordType *h_loss_ptr,
                        int inum,
                        int *h_ilist,
                        CoordType e_weight,
                        CoordType f_weight,
                        CoordType v_weight,
                        CoordType etot_ml,
                        CoordType etot_dft,
                        CoordType (*h_force_ml)[3],
                        CoordType (*h_force_dft)[3],
                        CoordType *h_virial_ml,
                        CoordType *h_virial_dft)
{
    int block_size_x = 128;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_loss_ptr;
    int *d_ilist;
    CoordType (*d_force_ml)[3];
    CoordType (*d_force_dft)[3];
    CoordType *d_virial_ml;
    CoordType *d_virial_dft;

    CHECK_CUDA_API( cudaMalloc((void**)&d_loss_ptr, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMemset(d_loss_ptr, 0.0, sizeof(CoordType)) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force_ml, sizeof(CoordType) * inum * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force_dft, sizeof(CoordType) * inum * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_virial_ml, sizeof(CoordType) * 9) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_virial_dft, sizeof(CoordType) * 9) );

    CHECK_CUDA_API( cudaMemcpy(d_ilist, h_ilist, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_force_ml, h_force_dft, sizeof(CoordType)*inum*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_force_dft, h_force_ml, sizeof(CoordType)*inum*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_virial_ml, h_virial_ml, sizeof(CoordType)*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_virial_dft, h_virial_dft, sizeof(CoordType)*9, cudaMemcpyHostToDevice) );

    // Launch kernel function
    find_loss_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (d_loss_ptr,
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

    CHECK_CUDA_API( cudaMemcpy(h_loss_ptr, d_loss_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_loss_ptr) );
    CHECK_CUDA_API( cudaFree(d_ilist) );
    CHECK_CUDA_API( cudaFree(d_force_ml) );
    CHECK_CUDA_API( cudaFree(d_force_dft) );
    CHECK_CUDA_API( cudaFree(d_virial_ml) );
    CHECK_CUDA_API( cudaFree(d_virial_dft) );
}


template <typename CoordType>
__global__
void find_ef_loss_kernel(CoordType *loss_ptr,
                         int inum,
                         int *ilist,
                         CoordType e_weight,
                         CoordType f_weight,
                         CoordType etot_ml,
                         CoordType etot_dft,
                         CoordType (*force_ml)[3],
                         CoordType (*force_dft)[3])
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = nx;

    CoordType f_loss = 0.0;
    if (ii < inum) {
        int center_idx = ilist[ii];
        for (int aa=0; aa<3; aa++)
            f_loss += std::pow(force_ml[center_idx][aa] - force_dft[center_idx][aa], 2);
        f_loss = f_weight / (3*inum) * f_loss;
        atomicAdd(loss_ptr, f_loss);
    }

    if (nx == 0) {
        CoordType e_loss = 0.0;
        e_loss = e_weight / inum * std::pow(etot_ml - etot_dft, 2);
        atomicAdd(loss_ptr, e_loss);
    }
}


template <typename CoordType>
void find_ef_loss_launcher(CoordType *h_loss_ptr,
                           int inum,
                           int *h_ilist,
                           CoordType e_weight,
                           CoordType f_weight,
                           CoordType etot_ml,
                           CoordType etot_dft,
                           CoordType (*h_force_ml)[3],
                           CoordType (*h_force_dft)[3])
{
    int block_size_x = 128;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_loss_ptr;
    int *d_ilist;
    CoordType (*d_force_ml)[3];
    CoordType (*d_force_dft)[3];
    

    CHECK_CUDA_API( cudaMalloc((void**)&d_loss_ptr, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMemset(d_loss_ptr, 0.0, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force_ml, sizeof(CoordType) * inum * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force_dft, sizeof(CoordType) * inum * 3) );
    CHECK_CUDA_API( cudaMemcpy(d_ilist, h_ilist, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_force_ml, h_force_ml, sizeof(CoordType) * inum * 3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_force_dft, h_force_dft, sizeof(CoordType) * inum * 3, cudaMemcpyHostToDevice) );

    // Launch kernel
    find_ef_loss_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (d_loss_ptr,
                                                                       inum,
                                                                       d_ilist,
                                                                       e_weight,
                                                                       f_weight,
                                                                       etot_ml,
                                                                       etot_dft,
                                                                       d_force_ml,
                                                                       d_force_dft);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError());

    CHECK_CUDA_API( cudaMemcpy(h_loss_ptr, d_loss_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );
    
    CHECK_CUDA_API( cudaFree(d_loss_ptr) );
    CHECK_CUDA_API( cudaFree(d_ilist) );
    CHECK_CUDA_API( cudaFree(d_force_ml) );
    CHECK_CUDA_API( cudaFree(d_force_dft) );
}



};  // namespace : nnmtp
};  // namespace : ai2pot


#endif