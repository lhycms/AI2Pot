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
#include "./fitutils_utilities.cuh"


namespace ai2pot {
namespace fitutils {

template <typename CoordType>
static __global__
void find_all_type_descriptors_sum_kernel(
    int natoms_in_batch,
    CoordType *descriptors_sum,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors);


template <typename CoordType>
static __global__
void find_all_type_descriptors_M2_kernel(
    int natoms_in_batch,
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
void find_each_type_descriptors_sum_kernel(
    int *natoms_in_batch,
    CoordType *descriptors_psum,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    int *bilist,
    int *btypes,
    int ntypes,
    CoordType *bdescriptors);


template <typename CoordType>
static __global__
void find_each_type_descriptors_M2_kernel(
    int *natoms_in_batch,
    CoordType *descriptors_pM2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    int *bilist,
    int *btypes,
    int ntypes,
    CoordType *bdescriptors);









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
    int natoms_in_batch = 0;
    for (int bb=0; bb<batch_size; bb++)
        natoms_in_batch += h_binum[bb];
    (*h_natoms_in_batch_ptr) = natoms_in_batch;

    int grid_size_x = descriptor_dim;
    int block_size_x = 1024;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);
    int sh_mem = sizeof(CoordType) * block_size_x;

    CoordType *d_descriptors_sum;
    CoordType *d_descriptors_M2;
    CHECK_CUDA_API( cudaMalloc((void**)&d_descriptors_sum, sizeof(CoordType)*descriptor_dim) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_descriptors_M2, sizeof(CoordType)*descriptor_dim) );
    CHECK_CUDA_API( cudaMemset(d_descriptors_sum, 0, sizeof(CoordType)*descriptor_dim) );
    CHECK_CUDA_API( cudaMemset(d_descriptors_M2, 0, sizeof(CoordType)*descriptor_dim) );

    int *d_binum;
    CoordType *d_bdescriptors;
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bdescriptors, sizeof(CoordType)*batch_size*natoms_pad*descriptor_dim) );
    CHECK_CUDA_API( cudaMemcpy(d_bdescriptors, h_bdescriptors, sizeof(CoordType)*batch_size*natoms_pad*descriptor_dim, cudaMemcpyHostToDevice) );

    //find_all_type_descriptors_sum_kernel KERNEL_ARG3(grid_size, block_size, sh_mem) (
        //natoms_in_batch,
        //d_descriptors_sum,
        //batch_size,
        //natoms_pad,
        //descriptor_dim,
        //d_binum,
        //d_bdescriptors);
    //CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaFree(d_descriptors_sum) );
    CHECK_CUDA_API( cudaFree(d_descriptors_M2) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bdescriptors) );
}


};  // namespace : fitutils
};  // namespace : ai2pot


#endif