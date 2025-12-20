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

#ifndef AI2POT_NEP_NEP_LOSS_CUH
#define AI2POT_NEP_NEP_LOSS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include "./nep_utilities.cuh"


namespace ai2pot {
namespace nep {

template <typename CoordType>
static __global__ 
void find_loss_kernel(
    CoordType *bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
    CoordType *bvirial_dft);


template <typename CoordType>
static __host__
void find_loss_launcher(
    CoordType *h_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3],
    CoordType *h_bvirial_ml,
    CoordType *h_bvirial_dft);


template <typename CoordType>
static __global__
void find_ef_loss_kernel(
    CoordType &bloss,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3]);


template <typename CoordType>
static __host__
void find_ef_loss_launcher(
    CoordType *h_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_force_ml)[3],
    CoordType (*h_force_dft)[3]);



template <typename CoordType>
__global__
void find_loss_kernel(
    CoordType *bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
    CoordType *bvirial_dft)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int ii = nx % natoms_pad;
    
    CoordType *loss_ptr = &bloss_ptr[istruct];
    CoordType etot_ml = betot_ml[istruct];
    CoordType etot_dft = betot_dft[istruct];
    CoordType (*force_ml)[3] = &bforce_ml[istruct*natoms_pad + 0];
    CoordType (*force_dft)[3] = &bforce_dft[istruct*natoms_pad + 0];
    CoordType *virial_ml = &bvirial_ml[istruct*9 + 0];
    CoordType *virial_dft = &bvirial_dft[istruct*9 + 0];
    int inum = binum[istruct];
    int *ilist = &bilist[istruct*natoms_pad];

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
void find_loss_launcher(
    CoordType *h_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3],
    CoordType *h_bvirial_ml,
    CoordType *h_bvirial_dft)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_bloss_ptr;
    int *d_binum;
    int *d_bilist;
    CoordType *d_betot_ml;
    CoordType *d_betot_dft;
    CoordType (*d_bforce_ml)[3];
    CoordType (*d_bforce_dft)[3];
    CoordType *d_bvirial_ml;
    CoordType *d_bvirial_dft;

    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_ptr, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMemset(d_bloss_ptr, 0.0, sizeof(CoordType)*batch_size) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ml, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_dft, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_ml, sizeof(CoordType)*batch_size*9) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_dft, sizeof(CoordType)*batch_size*9) );

    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_ml, h_betot_ml, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_dft, h_betot_dft, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_ml, h_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_dft, h_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bvirial_ml, h_bvirial_ml, sizeof(CoordType)*batch_size*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bvirial_dft, h_bvirial_dft, sizeof(CoordType)*batch_size*9, cudaMemcpyHostToDevice) );

    // Launch kernel function
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

    CHECK_CUDA_API( cudaMemcpy(h_bloss_ptr, d_bloss_ptr, sizeof(CoordType)*batch_size, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_bloss_ptr) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_betot_ml) );
    CHECK_CUDA_API( cudaFree(d_betot_dft) );
    CHECK_CUDA_API( cudaFree(d_bforce_ml) );
    CHECK_CUDA_API( cudaFree(d_bforce_dft) );
    CHECK_CUDA_API( cudaFree(d_bvirial_ml) );
    CHECK_CUDA_API( cudaFree(d_bvirial_dft) );
}


template <typename CoordType>
__global__
void find_ef_loss_kernel(
    CoordType *bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3])
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int ii = nx % natoms_pad;

    CoordType *loss_ptr = &bloss_ptr[istruct];
    CoordType etot_ml = betot_ml[istruct];
    CoordType etot_dft = betot_dft[istruct];
    CoordType (*force_ml)[3] = &bforce_ml[istruct*natoms_pad + 0];
    CoordType (*force_dft)[3] = &bforce_dft[istruct*natoms_pad + 0];
    int inum = binum[istruct];
    int *ilist = &bilist[istruct*natoms_pad];

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
void find_ef_loss_launcher(
    CoordType *h_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3])
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_bloss_ptr;
    int *d_binum;
    int *d_bilist;
    CoordType *d_betot_ml;
    CoordType *d_betot_dft;
    CoordType (*d_bforce_ml)[3];
    CoordType (*d_bforce_dft)[3];
    

    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_ptr, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMemset(d_bloss_ptr, 0.0, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ml, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_dft, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3) );
    
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_ml, h_betot_ml, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_dft, h_betot_dft, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_ml, h_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_dft, h_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );

    // Launch kernel
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
    CHECK_CUDA_API( cudaGetLastError());

    CHECK_CUDA_API( cudaMemcpy(h_bloss_ptr, d_bloss_ptr, sizeof(CoordType)*batch_size, cudaMemcpyDeviceToHost) );
    
    CHECK_CUDA_API( cudaFree(d_bloss_ptr) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_betot_ml) );
    CHECK_CUDA_API( cudaFree(d_betot_dft) );
    CHECK_CUDA_API( cudaFree(d_bforce_ml) );
    CHECK_CUDA_API( cudaFree(d_bforce_dft) );
}


};  // namespace : nep
};  // namespace : ai2pot

#endif