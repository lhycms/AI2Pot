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

#ifndef AI2POT_FITUTILS_EFV_RMSE_CUH
#define AI2POT_FITUTILS_EFV_RMSE_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/fitutils_utilities.cuh"


namespace ai2pot {
namespace fitutils {


static __global__
void find_num_real_atoms_in_batch_kernel(
    int *num_real_atoms_in_batch_ptr,
    int batch_size,
    int *binum);

static __host__
void find_num_real_atoms_in_batch_launcher(
    int *h_num_real_atoms_in_batch_ptr,
    int batch_size,
    int *h_binum);

template <typename CoordType>
static __global__
void find_e_se_kernel(
    CoordType *e_se_ptr,
    int batch_size,
    int *binum,
    CoordType *betot_ml,
    CoordType *betot_dft);

template <typename CoordType>
static __host__
void find_e_se_launcher(
    CoordType *h_e_se_ptr,
    int batch_size,
    int *h_binum,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft);

template <typename CoordType>
static __global__
void find_f_se_kernel(
    CoordType *f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3]);

template <typename CoordType>
static __host__
void find_f_se_launcher(
    CoordType *h_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3]);

template <typename CoordType>
static __global__
void find_v_se_kernel(
    CoordType *v_se_ptr,
    int batch_size,
    int *binum,
    CoordType *bvirial_ml,
    CoordType *bvirial_dft);

template <typename CoordType>
static __host__
void find_v_se_launcher(
    CoordType *h_v_se_ptr,
    int batch_size,
    int *h_binum,
    CoordType *h_bvirial_ml,
    CoordType *h_bvirial_dft);




__global__ void find_num_real_atoms_in_batch_kernel(
    int *num_real_atoms_in_batch_ptr,
    int batch_size,
    int *binum)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int s_part_x[1024];

    int local_sum = 0;
    for (int ii=nx; ii<batch_size; ii+=gridDim.x*blockDim.x)
        local_sum += binum[ii];
    s_part_x[tid] = local_sum;
    __syncthreads();

    for (int offset=blockDim.x>>1; offset>=32; offset>>=1) {
        if (tid < offset)
            s_part_x[tid] += s_part_x[tid+offset];
        __syncthreads();
    }
    for (int offset=16; offset>0; offset>>=1) {
        if (tid < offset)
            s_part_x[tid] += s_part_x[tid+offset];
        __syncwarp();
    }

    if (tid == 0) {
        atomicAdd(num_real_atoms_in_batch_ptr, s_part_x[0]);
    }
}


__host__ void find_num_real_atoms_in_batch_launcher(
    int *h_num_real_atoms_in_batch_ptr,
    int batch_size,
    int *h_binum)
{
    int block_size_x = 1024;
    int grid_size_x = 32;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    int *d_num_real_atoms_in_batch_ptr;
    int *d_binum;
    
    CHECK_CUDA_API( cudaMalloc((void**)&d_num_real_atoms_in_batch_ptr, sizeof(int)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int) * batch_size) );

    CHECK_CUDA_API( cudaMemset(d_num_real_atoms_in_batch_ptr, 0, sizeof(int)) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );

    // Compute
    find_num_real_atoms_in_batch_kernel KERNEL_ARG2(grid_size, block_size) (
        d_num_real_atoms_in_batch_ptr,
        batch_size,
        d_binum);
    CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaMemcpy(h_num_real_atoms_in_batch_ptr, d_num_real_atoms_in_batch_ptr, sizeof(int), cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_num_real_atoms_in_batch_ptr) );
    CHECK_CUDA_API( cudaFree(d_binum) );
}


template <typename CoordType>
__global__ void find_e_se_kernel(
    CoordType *e_se_ptr,
    int batch_size,
    int *binum,
    CoordType *betot_ml,
    CoordType *betot_dft)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ CoordType s_part_x[1024];
    
    CoordType local_sum = 0.0;
    for (int ii=nx; ii<batch_size; ii+=gridDim.x*blockDim.x) {
        CoordType tmp_e_diff = (betot_ml[ii] - betot_dft[ii]) / binum[ii];
        local_sum += tmp_e_diff * tmp_e_diff;
    }
    s_part_x[tid] = local_sum;
    __syncthreads();

    for (int offset=blockDim.x>>1; offset>=32; offset>>=1) {
        if (tid < offset)
            s_part_x[tid] += s_part_x[tid+offset];
        __syncthreads();
    }
    for (int offset=16; offset>0; offset>>=1) {
        if (tid < offset)
            s_part_x[tid] += s_part_x[tid+offset];
        __syncwarp();
    }

    if (tid == 0) {
        atomicAdd(e_se_ptr, s_part_x[0]);
    }
}


template <typename CoordType>
__host__
void find_e_se_launcher(
    CoordType *h_e_se_ptr,
    int batch_size,
    int *h_binum,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft)
{
    int block_size_x = 1024;
    int grid_size_x = 32;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_e_se_ptr;
    int *d_binum;
    CoordType *d_betot_ml;
    CoordType *d_betot_dft;
    CHECK_CUDA_API( cudaMalloc((void**)&d_e_se_ptr, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ml, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_dft, sizeof(CoordType)*batch_size) );
    
    CHECK_CUDA_API( cudaMemset(d_e_se_ptr, 0, sizeof(CoordType)) );

    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_ml, h_betot_ml, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_dft, h_betot_dft, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );

    // Compute
    find_e_se_kernel KERNEL_ARG2(grid_size, block_size) (
        d_e_se_ptr,
        batch_size,
        d_binum,
        d_betot_ml,
        d_betot_dft);
    CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaMemcpy(h_e_se_ptr, d_e_se_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_e_se_ptr) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_betot_ml) );
    CHECK_CUDA_API( cudaFree(d_betot_dft) );
}


template <typename CoordType>
__global__
void find_f_se_kernel(
    CoordType *f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3])
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ CoordType s_part_x[1024];
    CoordType local_sum = 0.0;
    for (int i=nx; i<batch_size*natoms_pad; i+=gridDim.x*blockDim.x) {
        int istruct = i / natoms_pad;
        int ii = i % natoms_pad;
        int center_idx = bilist[istruct*natoms_pad + ii];
        if ((istruct < batch_size) && (ii < binum[istruct])) {
            CoordType tmp_diff_x = (bforce_ml[istruct*natoms_pad + center_idx][0] - bforce_dft[istruct*natoms_pad + center_idx][0]);
            CoordType tmp_diff_y = (bforce_ml[istruct*natoms_pad + center_idx][1] - bforce_dft[istruct*natoms_pad + center_idx][1]);
            CoordType tmp_diff_z = (bforce_ml[istruct*natoms_pad + center_idx][2] - bforce_dft[istruct*natoms_pad + center_idx][2]);
            local_sum += (tmp_diff_x * tmp_diff_x
                          + tmp_diff_y * tmp_diff_y
                          + tmp_diff_z * tmp_diff_z);
            }
    }
    s_part_x[tid] = local_sum;
    __syncthreads();

    for (int offset=blockDim.x>>1; offset>=32; offset>>=1) {
        if (tid < offset)
            s_part_x[tid] += s_part_x[tid + offset];
        __syncthreads();
    }
    for (int offset=16; offset>0; offset>>=1) {
        if (tid < offset)
            s_part_x[tid] += s_part_x[tid + offset];
        __syncwarp();
    }

    if (tid == 0) {
        atomicAdd(f_se_ptr, s_part_x[0]);
    }
}


template <typename CoordType>
__host__
void find_f_se_launcher(
    CoordType *h_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3])
{
    int block_size_x = 1024;
    int grid_size_x = 32;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_f_se_ptr;
    int *d_binum;
    int *d_bilist;
    CoordType (*d_bforce_ml)[3];
    CoordType (*d_bforce_dft)[3];

    CHECK_CUDA_API( cudaMalloc((void**)&d_f_se_ptr, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3) );

    CHECK_CUDA_API( cudaMemset(d_f_se_ptr, 0, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_ml, h_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_dft, h_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );

    // Compute
    find_f_se_kernel KERNEL_ARG2(grid_size, block_size) (
        d_f_se_ptr,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        d_bforce_ml,
        d_bforce_dft);
    CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaMemcpy(h_f_se_ptr, d_f_se_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_f_se_ptr) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_bforce_ml) );
    CHECK_CUDA_API( cudaFree(d_bforce_dft) );
}


template <typename CoordType>
__global__
void find_v_se_kernel(
    CoordType *v_se_ptr,
    int batch_size,
    int *binum,
    CoordType *bvirial_ml,
    CoordType *bvirial_dft)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ CoordType s_part_x[1024];

    CoordType local_sum = 0.0;
    for (int ii=nx; ii<batch_size; ii+=gridDim.x*blockDim.x) {
        for (int a=0; a<3; a++) {
            for (int b=0; b<3; b++) {
                CoordType tmp_diff = (bvirial_ml[ii*9 + a*3 + b] - bvirial_dft[ii*9 + a*3 + b]) / binum[ii];
                local_sum += tmp_diff * tmp_diff;
            }
        }
    }
    s_part_x[tid] = local_sum;
    __syncthreads();

    for (int offset=blockDim.x>>1; offset>=32; offset>>=1) {
        if (tid < offset)
            s_part_x[tid] += s_part_x[tid + offset];
        __syncthreads();
    }
    for (int offset=16; offset>0; offset>>=1) {
        if (tid < offset)
            s_part_x[tid] += s_part_x[tid + offset];
        __syncwarp();
    }

    if (tid == 0) {
        atomicAdd(v_se_ptr, s_part_x[0]);
    }
}


template <typename CoordType>
__host__
void find_v_se_launcher(
    CoordType *h_v_se_ptr,
    int batch_size,
    int *h_binum,
    CoordType *h_bvirial_ml,
    CoordType *h_bvirial_dft)
{
    int grid_size_x = 32;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_v_se_ptr;
    int *d_binum;
    CoordType *d_bvirial_ml;
    CoordType *d_bvirial_dft;
    
    CHECK_CUDA_API( cudaMalloc((void**)&d_v_se_ptr, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_ml, sizeof(CoordType)*batch_size*9) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_dft, sizeof(CoordType)*batch_size*9) );

    CHECK_CUDA_API( cudaMemset(d_v_se_ptr, 0, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bvirial_ml, h_bvirial_ml, sizeof(CoordType)*batch_size*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bvirial_dft, h_bvirial_dft, sizeof(CoordType)*batch_size*9, cudaMemcpyHostToDevice) );

    // Compute
    find_v_se_kernel KERNEL_ARG2(grid_size, block_size) (
        d_v_se_ptr,
        batch_size,
        d_binum,
        d_bvirial_ml,
        d_bvirial_dft);
    CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaMemcpy(h_v_se_ptr, d_v_se_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_v_se_ptr) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bvirial_ml) );
    CHECK_CUDA_API( cudaFree(d_bvirial_dft) );
}


};  // namespace : fitutils
};  // namespace : ai2pot

#endif