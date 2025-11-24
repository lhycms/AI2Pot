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

#ifndef AI2POT_FITUTILS_DESCRIPTORS_STATISTICS_CUH
#define AI2POT_FITUTILS_DESCRIPTORS_STATISTICS_CUH
#include <chrono>
#include <iostream>
#include "./fitutils_utilities.cuh"


namespace ai2pot {
namespace fitutils {


static __global__
void find_natoms_in_batch_kernel(
    int *natoms_in_batch_ptr,
    int batch_size,
    int *binum);


template <typename CoordType>
static __global__
void find_all_type_descriptors_mean_kernel(
    int *natoms_in_batch_ptr,
    CoordType *descriptors_mean,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors);


template <typename CoordType>
static __global__
void find_all_type_descriptors_M2_kernel(
    int *natoms_in_batch_ptr,
    CoordType *descriptors_mean,
    CoordType *descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors);


template <typename CoordType>
static __host__
void find_all_type_descriptors_statistics_launcher(
    int *h_natoms_in_batch_ptr,
    CoordType *h_descriptors_mean,
    CoordType *h_descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *h_binum,
    CoordType *h_bdescriptors);


template <typename CoordType>
static __global__
void find_all_type_descriptors_maxmin_kernel(
    CoordType *descriptors_max,
    CoordType *descriptors_min,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors);


template <typename CoordType>
static __host__
void find_all_type_descriptors_maxmin_launcher(
    CoordType *h_descriptors_max,
    CoordType *h_descriptors_min,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors);






__global__ void find_natoms_in_batch_kernel(
    int *natoms_in_batch_ptr,
    int batch_size,
    int *binum)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp_natoms_in_batch = 0;
    for (int i=nx; i<batch_size; i+=gridDim.x*blockDim.x) {
        tmp_natoms_in_batch += binum[i];
    }

    atomicAdd(natoms_in_batch_ptr, tmp_natoms_in_batch);
}


template <typename CoordType>
__global__
void find_all_type_descriptors_mean_kernel(
    int *natoms_in_batch_ptr,
    CoordType *descriptors_mean,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ CoordType s_part_x[1024];

    CoordType tmp_part_x = 0.0;
    for (int i=tid; i<batch_size*natoms_pad; i+=blockDim.x) {
        int bb = i / natoms_pad;
        int ii = i % natoms_pad;
        if (ii>=binum[bb])
            continue;
        tmp_part_x += bdescriptors[(bb*natoms_pad+ii)*descriptor_dim + bid];
    }
    s_part_x[tid] = tmp_part_x;
    __syncthreads();

    for (int offset=blockDim.x>>1; offset>=32; offset>>=1) {
        if (tid < offset) {
            s_part_x[tid] += s_part_x[tid+offset];
        }
        __syncthreads();
    }

    for (int offset=16; offset>0; offset>>=1) {
        if (tid < offset) {
            s_part_x[tid] += s_part_x[tid+offset];
        }
        __syncwarp();
    }

    if (tid == 0) {
        descriptors_mean[bid] = s_part_x[0] / (*natoms_in_batch_ptr);
    }
}


template <typename CoordType>
__global__
void find_all_type_descriptors_M2_kernel(
    int *natoms_in_batch_ptr,
    CoordType *descriptors_mean,
    CoordType *descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ CoordType s_part_x[1024];

    CoordType tmp_part_x = 0.0;
    for (int i=tid; i<batch_size*natoms_pad; i+=blockDim.x) {
        int bb = i / natoms_pad;
        int ii = i % natoms_pad;
        if (ii>=binum[bb])
            continue;
        tmp_part_x += std::pow((bdescriptors[(bb*natoms_pad+ii)*descriptor_dim + bid] 
                                - descriptors_mean[bid]), 2);
    }
    s_part_x[tid] = tmp_part_x;
    __syncthreads();

    for (int offset=blockDim.x>>1; offset>=32; offset>>=1) {
        if (tid < offset) {
            s_part_x[tid] += s_part_x[tid+offset];
        }
        __syncthreads();
    }

    for (int offset=16; offset>0; offset>>=1) {
        if (tid < offset) {
            s_part_x[tid] += s_part_x[tid+offset];
        }
        __syncwarp();
    }

    if (tid == 0) {
        descriptors_M2[bid] = s_part_x[0];
    }
}


template <typename CoordType>
__host__
void find_all_type_descriptors_statistics_launcher(
    int *h_natoms_in_batch_ptr,
    CoordType *h_descriptors_mean,
    CoordType *h_descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *h_binum,
    CoordType *h_bdescriptors)
{
    int *d_natoms_in_batch_ptr;
    CHECK_CUDA_API( cudaMalloc((void**)&d_natoms_in_batch_ptr, sizeof(int)) );
    CHECK_CUDA_API( cudaMemset(d_natoms_in_batch_ptr, 0, sizeof(int)) );

    int grid_size_x = descriptor_dim;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);
    
    CoordType *d_descriptors_mean;
    CoordType *d_descriptors_M2;
    CHECK_CUDA_API( cudaMalloc((void**)&d_descriptors_mean, sizeof(CoordType)*descriptor_dim) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_descriptors_M2, sizeof(CoordType)*descriptor_dim) );
    CHECK_CUDA_API( cudaMemset(d_descriptors_mean, 0, sizeof(CoordType)*descriptor_dim) );
    CHECK_CUDA_API( cudaMemset(d_descriptors_M2, 0, sizeof(CoordType)*descriptor_dim) );
    
    int *d_binum;
    CoordType *d_bdescriptors;
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bdescriptors, sizeof(CoordType)*batch_size*natoms_pad*descriptor_dim) );
    CHECK_CUDA_API( cudaMemcpy(d_bdescriptors, h_bdescriptors, sizeof(CoordType)*batch_size*natoms_pad*descriptor_dim, cudaMemcpyHostToDevice) );

    auto t1 = std::chrono::high_resolution_clock::now();
    // 1. natoms_in_batch_ptr
    find_natoms_in_batch_kernel KERNEL_ARG2(grid_size, block_size) (
        d_natoms_in_batch_ptr,
        batch_size,
        d_binum);
    CHECK_CUDA_KERNEL;

    // 2. mean
    find_all_type_descriptors_mean_kernel KERNEL_ARG2(grid_size, block_size) (
        d_natoms_in_batch_ptr,
        d_descriptors_mean,
        batch_size,
        natoms_pad,
        descriptor_dim,
        d_binum,
        d_bdescriptors);
    CHECK_CUDA_KERNEL;

    // 3. M2
    find_all_type_descriptors_M2_kernel KERNEL_ARG2(grid_size, block_size) (
        d_natoms_in_batch_ptr,
        d_descriptors_mean,
        d_descriptors_M2,
        batch_size,
        natoms_pad,
        descriptor_dim,
        d_binum,
        d_bdescriptors);


    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_all_type_descriptors_statistics_launcher() cost time: " << duration.count() << " us." << std::endl;

    CHECK_CUDA_API( cudaMemcpy(h_natoms_in_batch_ptr, d_natoms_in_batch_ptr, sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_descriptors_mean, d_descriptors_mean, sizeof(CoordType)*descriptor_dim, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_descriptors_M2, d_descriptors_M2, sizeof(CoordType)*descriptor_dim, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_natoms_in_batch_ptr) );
    CHECK_CUDA_API( cudaFree(d_descriptors_mean) );
    CHECK_CUDA_API( cudaFree(d_descriptors_M2) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bdescriptors) );
}


template <typename CoordType>
__global__ 
void find_all_type_descriptors_maxmin_kernel(
    CoordType *descriptors_max,
    CoordType *descriptors_min,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ CoordType s_max_min[1024];
    CoordType* s_max = s_max_min;
    CoordType* s_min = s_max_min + blockDim.x;

    s_max[tid] = -1000000.0;
    s_min[tid] = 1000000.0;
    for (int i=tid; i<batch_size*natoms_pad; i+=blockDim.x) {
        int bb = i / natoms_pad;
        int ii = i % natoms_pad;
        if (ii >= binum[bb])
            continue;

        CoordType tmp_descriptor = bdescriptors[(bb*natoms_pad+ii)*descriptor_dim + bid];
        s_max[tid] = (s_max[tid] > tmp_descriptor) ? s_max[tid] : tmp_descriptor;
        s_min[tid] = (s_min[tid] < tmp_descriptor) ? s_min[tid] : tmp_descriptor;
    }
    __syncthreads();


    for (int offset=blockDim.x>>2; offset>=32; offset>>=1) {
        if (tid < offset) {
            s_max[tid] = (s_max[tid] > s_max[tid+offset]) ? s_max[tid] : s_max[tid+offset];
            s_min[tid] = (s_min[tid] < s_min[tid+offset]) ? s_min[tid] : s_min[tid+offset];
        }
        __syncthreads();
    }

    for (int offset=16; offset>0; offset>>=1) {
        if (tid < offset) {
            s_max[tid] = (s_max[tid] > s_max[tid+offset]) ? s_max[tid] : s_max[tid+offset];
            s_min[tid] = (s_min[tid] < s_min[tid+offset]) ? s_min[tid] : s_min[tid+offset];
        }
        __syncwarp();
    }

    if (tid == 0) {
        descriptors_max[bid] = s_max[0];
        descriptors_min[bid] = s_min[0];
    }
}


template <typename CoordType>
__host__ 
void find_all_type_descriptors_maxmin_launcher(
    CoordType *h_descriptors_max,
    CoordType *h_descriptors_min,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *h_binum,
    CoordType *h_bdescriptors)
{   
    int grid_size_x = descriptor_dim;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);
    int sh_mem = sizeof(CoordType) * 2 * 1024;

    CoordType *d_descriptors_max;
    CoordType *d_descriptors_min;
    int *d_binum;
    CoordType *d_bdescriptors;
    CHECK_CUDA_API( cudaMalloc((void**)&d_descriptors_max, sizeof(CoordType)*descriptor_dim) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_descriptors_min, sizeof(CoordType)*descriptor_dim) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bdescriptors, sizeof(CoordType)*batch_size*natoms_pad*descriptor_dim) );

    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bdescriptors, h_bdescriptors, sizeof(CoordType)*batch_size*natoms_pad*descriptor_dim, cudaMemcpyHostToDevice) );

    find_all_type_descriptors_maxmin_kernel KERNEL_ARG3(grid_size, block_size, sh_mem) (
        d_descriptors_max,
        d_descriptors_min,
        batch_size,
        natoms_pad,
        descriptor_dim,
        d_binum,
        d_bdescriptors);
    CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaMemcpy(h_descriptors_max, d_descriptors_max, sizeof(CoordType)*descriptor_dim, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_descriptors_min, d_descriptors_min, sizeof(CoordType)*descriptor_dim, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_descriptors_max) );
    CHECK_CUDA_API( cudaFree(d_descriptors_min) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bdescriptors) );
}


};  // namespace : fitutils
};  // namespace : ai2pot


#endif