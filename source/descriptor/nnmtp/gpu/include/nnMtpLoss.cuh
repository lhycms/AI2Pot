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
#include <chrono>
#include <iostream>

#include "./mtpUtilities.cuh"
#include "./mtpBasis.cuh"


namespace ai2pot {
namespace nnmtp {

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
static __device__
void find_loss_backward_atom(
    CoordType *loss_der2coeffs,
    CoordType *loss_der2w0,
    CoordType *loss_der2w1,
    CoordType *loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    CoordType *virial_ml,
    CoordType *virial_dft,
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);
                    
template <typename CoordType>
static __global__
void find_loss_backward_kernel(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2w0,
    CoordType *bloss_der2w1,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
    CoordType *bvirial_dft,
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);

template <typename CoordType>
static __host__
void find_loss_backward_launcher(
    CoordType *h_bloss_der2coeffs,
    CoordType *h_bloss_der2w0,
    CoordType *h_bloss_der2w1,
    CoordType *h_bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3],
    CoordType *h_bvirial_ml,
    CoordType *h_bvirial_dft,
    int chebyshev_size,
    int num_neurons,
    CoordType *h_coeffs,
    CoordType *h_w0,
    CoordType *h_w1,
    CoordType *h_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*h_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*h_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *h_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    int *h_bnumneigh,
    int *h_bfirstneigh,
    CoordType (*h_brcs)[3],
    int *h_btypes,
    int ntypes,
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);

template <typename CoordType>
static __device__
void find_ef_loss_backward_atom(
    CoordType *loss_der2coeffs,
    CoordType *loss_der2w0,
    CoordType *loss_der2w1,
    CoordType *loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);


template <typename CoordType>
static __global__
void find_ef_loss_backward_kernel(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2w0,
    CoordType *bloss_der2w1,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);


template <typename CoordType>
static __host__
void find_ef_loss_backward_launcher(
    CoordType *h_bloss_der2coeffs,
    CoordType *h_bloss_der2w0,
    CoordType *h_bloss_der2w1,
    CoordType *h_bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3],
    int chebyshev_size,
    int num_neurons,
    CoordType *h_coeffs,
    CoordType *h_w0,
    CoordType *h_w1,
    CoordType *h_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*h_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*h_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *h_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    int *h_bnumneigh,
    int *h_bfirstneigh,
    CoordType (*h_brcs)[3],
    int *h_btypes,
    int ntypes,
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);





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


template <typename CoordType>
__device__
void find_loss_backward_atom(
    CoordType *loss_der2coeffs,
    CoordType *loss_der2w0,
    CoordType *loss_der2w1,
    CoordType *loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    CoordType *virial_ml,
    CoordType *virial_dft,
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.0};
    CoordType e_site_der2mom[MAX_ALPHA_MOMENTS_COUNT] = {0.0};
    CoordType dloss_combination[MAX_ALPHA_MOMENTS_COUNT] = {0.0};

    int center_idx;
    int type_central;
    int Zi;
    int neigh_idx;
    int type_outer;
    int Zj;
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    CoordType auto_dist_powers_[MAX_ALPHA_INDEX_BASIC] = {0.0};
    CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3] = {{0.0}};

    center_idx = silist;
    type_central = types[center_idx];
    Zi = type_map[type_central];
    CoordType *type_central_w0 = &w0[type_central*num_neurons*alpha_scalar_moments];
    CoordType *type_central_w1 = &w1[type_central*num_neurons];


    // 1. Mom_vals
    for (int jj=0; jj<snumneigh; jj++)
    {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        Zj = type_map[type_outer];
        for (int aa=0; aa<3; aa++)
            NeighbVect[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                 + std::pow(NeighbVect[1], 2)
                                 + std::pow(NeighbVect[2], 2) );
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                     rq_chebyshev_der2r,
                                     chebyshev_size,
                                     rmax,
                                     rmin,
                                     distance_ij);

        auto_dist_powers_[0] = 1.0;
        for (int aa=0; aa<3; aa++)
            auto_coords_powers_[0][aa] = 1.0;
        for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++) {
            int mu = alpha_index_basic[i][0];
            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            CoordType powk = 1.0 / auto_dist_powers_[k];
            CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
            CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
            CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
            CoordType mult0 = pow0 * pow1 * pow2;

            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes + type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                CoordType A = rq_chebyshev_vals[xi];
                CoordType B = mult0;
                CoordType C = powk;
                CoordType A_ders[3] = {0.0, 0.0, 0.0};
                CoordType B_ders[3] = {0.0, 0.0, 0.0};
                CoordType C_ders[3] = {0.0, 0.0, 0.0};
                A_ders[0] = rq_chebyshev_der2r[xi] * NeighbVect[0] * distance_ij_inv;
                A_ders[1] = rq_chebyshev_der2r[xi] * NeighbVect[1] * distance_ij_inv;
                A_ders[2] = rq_chebyshev_der2r[xi] * NeighbVect[2] * distance_ij_inv;
                if (alpha_index_basic[i][1] != 0) {
                    B_ders[0] = alpha_index_basic[i][1]
                                * auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
                                * pow1
                                * pow2;
                }
                if (alpha_index_basic[i][2] != 0) {
                    B_ders[1] = alpha_index_basic[i][2]
                                * pow0
                                * auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
                                * pow2;
                }
                if (alpha_index_basic[i][3] != 0) {
                    B_ders[2] = alpha_index_basic[i][3]
                                * pow0
                                * pow1
                                * auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
                }
                C_ders[0] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[0];
                C_ders[1] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[1];
                C_ders[2] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[2];
                mom_vals[i] += coeffs[idx] * A * B * C;

                for (int aa=0; aa<3; aa++) {
                    CoordType tmp_deriv = coeffs[idx]
                                          * (A_ders[aa]*B*C + A*B_ders[aa]*C + A*B*C_ders[aa]);
                    dloss_combination[i] += 2*f_weight/(3*inum)
                                            * (force_ml[center_idx][aa] - force_dft[center_idx][aa])
                                            * tmp_deriv;
                    dloss_combination[i] -= 2*f_weight/(3*inum)
                                            * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa])
                                            * tmp_deriv;
                    for (int bb=0; bb<3; bb++) {
                        dloss_combination[i] -= 2*v_weight/(9*inum)
                                                * (virial_ml[aa*3+bb] - virial_dft[aa*3+bb])
                                                * NeighbVect[bb]
                                                * tmp_deriv;
                    }                    
                }
            }
        }
    }

    for (int i=0; i<alpha_index_times_count; i++) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];
        mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
        dloss_combination[alpha_index_times[i][3]] += (dloss_combination[alpha_index_times[i][0]] * val2 * val1
                                                       + dloss_combination[alpha_index_times[i][1]] * val2 * val0);
    }

    // 2.1. NN Energy derivative w.r.t. mom_vals
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        for (int k=0; k<alpha_scalar_moments; k++)
            hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
        for (int k=0; k<alpha_scalar_moments; k++)
            e_site_der2mom[alpha_moment_mapping[k]] += type_central_w1[p]
                                                       * activated_hidden_der
                                                       * type_central_w0[p*alpha_scalar_moments + k];
    }

    // 2.2. Pass to basic moments
    for (int i=alpha_index_times_count-1; i>=0; i--) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];

        e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val1;
        e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val0;
    }
    
    // 2.3. loss derivative w.r.t. coeffs
    for (int jj=0; jj<snumneigh; jj++)
    {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        Zj = type_map[type_outer];
        for (int aa=0; aa<3; aa++)
            NeighbVect[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                 + std::pow(NeighbVect[1], 2)
                                 + std::pow(NeighbVect[2], 2) );
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                     rq_chebyshev_der2r,
                                     chebyshev_size,
                                     rmax,
                                     rmin,
                                     distance_ij);
        auto_dist_powers_[0] = 1.0;
        for (int aa=0; aa<3; aa++)
            auto_coords_powers_[0][aa] = 1.0;
        for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++) {
            int mu = alpha_index_basic[i][0];
            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            CoordType powk = 1.0 / auto_dist_powers_[k];
            CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
            CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
            CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
            CoordType mult0 = pow0 * pow1 * pow2;

            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes + type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                CoordType A = rq_chebyshev_vals[xi];
                CoordType B = mult0;
                CoordType C = powk;
                CoordType A_ders[3] = {0.0, 0.0, 0.0};
                CoordType B_ders[3] = {0.0, 0.0, 0.0};
                CoordType C_ders[3] = {0.0, 0.0, 0.0};
                A_ders[0] = rq_chebyshev_der2r[xi] * NeighbVect[0] * distance_ij_inv;
                A_ders[1] = rq_chebyshev_der2r[xi] * NeighbVect[1] * distance_ij_inv;
                A_ders[2] = rq_chebyshev_der2r[xi] * NeighbVect[2] * distance_ij_inv;
                if (alpha_index_basic[i][1] != 0) {
                    B_ders[0] = alpha_index_basic[i][1]
                                * auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
                                * pow1
                                * pow2;
                }
                if (alpha_index_basic[i][2] != 0) {
                    B_ders[1] = alpha_index_basic[i][2]
                                * pow0
                                * auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
                                * pow2;
                }
                if (alpha_index_basic[i][3] != 0) {
                    B_ders[2] = alpha_index_basic[i][3]
                                * pow0
                                * pow1
                                * auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
                }
                C_ders[0] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[0];
                C_ders[1] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[1];
                C_ders[2] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[2];

                CoordType tmpe_loss_der2coeff = 2*e_weight/inum*(etot_ml - etot_dft)
                                                * e_site_der2mom[i]
                                                * A * B * C;

                CoordType tmpf_loss_der2coeff = 0;
                for (int aa=0; aa<3; aa++) 
                {
                    CoordType tmp_prefix = 0;
                    CoordType tmp_deriv = (A_ders[aa] * B * C
                                           + A * B_ders[aa] * C
                                           + A * B * C_ders[aa]);
                    tmp_prefix += 2*f_weight/(3*inum)
                                  * (force_ml[center_idx][aa] - force_dft[center_idx][aa]);
                    tmp_prefix -= 2*f_weight/(3*inum)
                                  * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa]);
                    for (int bb=0; bb<3; bb++) 
                    {
                        tmp_prefix -= 2*v_weight/(9*inum)
                                      * (virial_ml[aa*3+bb] - virial_dft[aa*3+bb])
                                      * NeighbVect[bb];
                    }
                    tmpf_loss_der2coeff += tmp_prefix * e_site_der2mom[i] * tmp_deriv;
                }
                atomicAdd(&loss_der2coeffs[idx], 
                          tmpe_loss_der2coeff + tmpf_loss_der2coeff);
            }
        }
    }


    // 2.4. loss derivative w.r.t. w1
    for (int p=0; p<num_neurons; p++) {
        CoordType tmpe_loss_der2w1 = 0.0;
        CoordType tmpf_loss_der2w1 = 0.0;
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        for (int k=0; k<alpha_scalar_moments; k++)
            hidden_val += type_central_w0[p*alpha_scalar_moments+k] * mom_vals[alpha_moment_mapping[k]];
        TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
        tmpe_loss_der2w1 = 2*e_weight/inum*(etot_ml - etot_dft)
                           * activated_hidden_val;
        for (int k=0; k<alpha_scalar_moments; k++)
            tmpf_loss_der2w1 += dloss_combination[alpha_moment_mapping[k]]
                                * activated_hidden_der
                                * type_central_w0[p*alpha_scalar_moments + k];
        
        atomicAdd(&loss_der2w1[type_central*num_neurons + p], tmpe_loss_der2w1 + tmpf_loss_der2w1);
    }
    
    // 2.5. loss derivative w.r.t. w0
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        CoordType activated_hidden_der2der = 0.0;
        for (int k=0; k<alpha_scalar_moments; k++)
            hidden_val += type_central_w0[p*alpha_scalar_moments+k] * mom_vals[alpha_moment_mapping[k]];
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
        TanhActivationFunc<CoordType>::find_der2der(activated_hidden_der2der, hidden_val);
        for (int k=0; k<alpha_scalar_moments; k++) {
            CoordType tmpe_loss_der2w0 = 2*e_weight/inum*(etot_ml - etot_dft)
                                         * type_central_w1[p]
                                         * activated_hidden_der
                                         * mom_vals[alpha_moment_mapping[k]];
            CoordType tmpf_loss_der2w0 = dloss_combination[alpha_moment_mapping[k]]
                                         * type_central_w1[p]
                                         * (activated_hidden_der
                                            + activated_hidden_der2der
                                              * type_central_w0[p*alpha_scalar_moments+k]
                                              * mom_vals[alpha_moment_mapping[k]]);

            atomicAdd(&loss_der2w0[type_central*num_neurons*alpha_scalar_moments + p*alpha_scalar_moments + k], tmpe_loss_der2w0 + tmpf_loss_der2w0);
        }
    }


    // 2.6. loss derivative w.r.t. type_bias
    atomicAdd(&loss_der2type_bias[type_central],
              2*e_weight/inum*(etot_ml - etot_dft));
}


template <typename CoordType>
__global__
void find_loss_backward_kernel(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2w0,
    CoordType *bloss_der2w1,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
    CoordType *bvirial_dft,
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int ii = nx % natoms_pad;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    CoordType *loss_der2coeffs = &bloss_der2coeffs[istruct*num_coeffs + 0];
    CoordType *loss_der2w0 = &bloss_der2w0[istruct*ntypes*num_neurons*alpha_scalar_moments + 0];
    CoordType *loss_der2w1 = &bloss_der2w1[istruct*ntypes*num_neurons + 0];
    CoordType *loss_der2type_bias = &bloss_der2type_bias[istruct*ntypes + 0];
    CoordType etot_ml = betot_ml[istruct];
    CoordType etot_dft = betot_dft[istruct];
    CoordType (*force_ml)[3] = &bforce_ml[istruct*natoms_pad + 0];
    CoordType (*force_dft)[3] = &bforce_dft[istruct*natoms_pad + 0];
    CoordType *virial_ml = &bvirial_ml[istruct*9 + 0];
    CoordType *virial_dft = &bvirial_dft[istruct*9 + 0];
    int inum = binum[istruct];
    int *types = &btypes[istruct*natoms_pad + 0];

    if (ii < inum) {
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = &brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        find_loss_backward_atom<CoordType>(
            loss_der2coeffs,
            loss_der2w0,
            loss_der2w1,
            loss_der2type_bias,
            e_weight,
            f_weight,
            v_weight,
            etot_ml,
            etot_dft,
            force_ml,
            force_dft,
            virial_ml,
            virial_dft,
            chebyshev_size,
            num_neurons,
            coeffs,
            w0,
            w1,
            type_bias,
            alpha_moments_count,
            alpha_index_basic_count,
            alpha_index_basic,
            alpha_index_times_count,
            alpha_index_times,
            alpha_scalar_moments,
            alpha_moment_mapping,
            nmus,
            inum,
            silist,
            snumneigh,
            sfirstneigh,
            srcs,
            types,
            ntypes,
            type_map,
            umax_num_neigh_atoms,
            nghost,
            rmax,
            rmin);
    }
}


template <typename CoordType>
__host__
void find_loss_backward_launcher(
    CoordType *h_bloss_der2coeffs,
    CoordType *h_bloss_der2w0,
    CoordType *h_bloss_der2w1,
    CoordType *h_bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3],
    CoordType *h_bvirial_ml,
    CoordType *h_bvirial_dft,
    int chebyshev_size,
    int num_neurons,
    CoordType *h_coeffs,
    CoordType *h_w0,
    CoordType *h_w1,
    CoordType *h_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*h_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*h_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *h_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    int *h_bnumneigh,
    int *h_bfirstneigh,
    CoordType (*h_brcs)[3],
    int *h_btypes,
    int ntypes,
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    CoordType *d_bloss_der2coeffs;
    CoordType *d_bloss_der2w0;
    CoordType *d_bloss_der2w1;
    CoordType *d_bloss_der2type_bias;
    CoordType *d_betot_ml;
    CoordType *d_betot_dft;
    CoordType (*d_bforce_ml)[3];
    CoordType (*d_bforce_dft)[3];
    CoordType *d_bvirial_ml;
    CoordType *d_bvirial_dft;
    CoordType *d_coeffs;
    CoordType *d_w0;
    CoordType *d_w1;
    CoordType *d_type_bias;
    int (*d_alpha_index_basic)[4];
    int (*d_alpha_index_times)[4];
    int *d_alpha_moment_mapping;
    int *d_binum;
    int *d_bilist;
    int *d_bnumneigh;
    int *d_bfirstneigh;
    CoordType (*d_brcs)[3];
    int *d_btypes;
    int *d_type_map;

    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2coeffs, sizeof(CoordType)*batch_size*num_coeffs) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2coeffs, 0.0, sizeof(CoordType)*batch_size*num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2w0, sizeof(CoordType)*batch_size*ntypes*num_neurons*alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2w0, 0.0, sizeof(CoordType)*batch_size*ntypes*num_neurons*alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2w1, sizeof(CoordType)*batch_size*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2w1, 0.0, sizeof(CoordType)*batch_size*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2type_bias, sizeof(CoordType)*batch_size*ntypes) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2type_bias, 0.0, sizeof(CoordType)*batch_size*ntypes) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ml, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_dft, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_ml, sizeof(CoordType)*batch_size*9) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_dft, sizeof(CoordType)*batch_size*9) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_w0, sizeof(CoordType) * ntypes * num_neurons * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_w1, sizeof(CoordType) * ntypes * num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int)*ntypes) );

    CHECK_CUDA_API( cudaMemcpy(d_betot_ml, h_betot_ml, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_dft, h_betot_dft, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_ml, h_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_dft, h_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bvirial_ml, h_bvirial_ml, sizeof(CoordType)*batch_size*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bvirial_dft, h_bvirial_dft, sizeof(CoordType)*batch_size*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*num_coeffs, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_w0, h_w0, sizeof(CoordType)*ntypes*num_neurons*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_w1, h_w1, sizeof(CoordType)*ntypes*num_neurons, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int)*alpha_index_times_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );


    auto t1 = std::chrono::high_resolution_clock::now();
    find_loss_backward_kernel KERNEL_ARG2(grid_size, block_size) (
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
        rmin);
    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_loss_backward_launcher() cost time: " << duration.count() << " us." << std::endl;


    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2coeffs, d_bloss_der2coeffs, sizeof(CoordType)*batch_size*num_coeffs, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2w0, d_bloss_der2w0, sizeof(CoordType)*batch_size*ntypes*num_neurons*alpha_scalar_moments, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2w1, d_bloss_der2w1, sizeof(CoordType)*batch_size*ntypes*num_neurons, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2type_bias, d_bloss_der2type_bias, sizeof(CoordType)*batch_size*ntypes, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_bloss_der2coeffs) );
    CHECK_CUDA_API( cudaFree(d_bloss_der2w0) );
    CHECK_CUDA_API( cudaFree(d_bloss_der2w1) );
    CHECK_CUDA_API( cudaFree(d_bloss_der2type_bias) );
    CHECK_CUDA_API( cudaFree(d_betot_ml) );
    CHECK_CUDA_API( cudaFree(d_betot_dft) );
    CHECK_CUDA_API( cudaFree(d_bforce_ml) );
    CHECK_CUDA_API( cudaFree(d_bforce_dft) );
    CHECK_CUDA_API( cudaFree(d_bvirial_ml) );
    CHECK_CUDA_API( cudaFree(d_bvirial_dft) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_w0) );
    CHECK_CUDA_API( cudaFree(d_w1) );
    CHECK_CUDA_API( cudaFree(d_type_bias) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_basic) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_times) );
    CHECK_CUDA_API( cudaFree(d_alpha_moment_mapping) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_bnumneigh) );
    CHECK_CUDA_API( cudaFree(d_bfirstneigh) );
    CHECK_CUDA_API( cudaFree(d_brcs) );
    CHECK_CUDA_API( cudaFree(d_btypes) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
}


template <typename CoordType>
__device__
void find_ef_loss_backward_atom(
    CoordType *loss_der2coeffs,
    CoordType *loss_der2w0,
    CoordType *loss_der2w1,
    CoordType *loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.0};
    CoordType e_site_der2mom[MAX_ALPHA_MOMENTS_COUNT] = {0.0};
    CoordType dloss_combination[MAX_ALPHA_MOMENTS_COUNT] = {0.0};

    int center_idx;
    int type_central;
    int Zi;
    int neigh_idx;
    int type_outer;
    int Zj;
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    CoordType auto_dist_powers_[MAX_ALPHA_INDEX_BASIC] = {0.0};
    CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3] = {{0.0}};

    center_idx = silist;
    type_central = types[center_idx];
    Zi = type_map[type_central];
    CoordType *type_central_w0 = &w0[type_central*num_neurons*alpha_scalar_moments];
    CoordType *type_central_w1 = &w1[type_central*num_neurons];


    // 1. Mom_vals
    for (int jj=0; jj<snumneigh; jj++)
    {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        Zj = type_map[type_outer];
        for (int aa=0; aa<3; aa++)
            NeighbVect[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                 + std::pow(NeighbVect[1], 2)
                                 + std::pow(NeighbVect[2], 2) );
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                     rq_chebyshev_der2r,
                                     chebyshev_size,
                                     rmax,
                                     rmin,
                                     distance_ij);

        auto_dist_powers_[0] = 1.0;
        for (int aa=0; aa<3; aa++)
            auto_coords_powers_[0][aa] = 1.0;
        for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++) {
            int mu = alpha_index_basic[i][0];
            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            CoordType powk = 1.0 / auto_dist_powers_[k];
            CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
            CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
            CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
            CoordType mult0 = pow0 * pow1 * pow2;

            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes + type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                CoordType A = rq_chebyshev_vals[xi];
                CoordType B = mult0;
                CoordType C = powk;
                CoordType A_ders[3] = {0.0, 0.0, 0.0};
                CoordType B_ders[3] = {0.0, 0.0, 0.0};
                CoordType C_ders[3] = {0.0, 0.0, 0.0};
                A_ders[0] = rq_chebyshev_der2r[xi] * NeighbVect[0] * distance_ij_inv;
                A_ders[1] = rq_chebyshev_der2r[xi] * NeighbVect[1] * distance_ij_inv;
                A_ders[2] = rq_chebyshev_der2r[xi] * NeighbVect[2] * distance_ij_inv;
                if (alpha_index_basic[i][1] != 0) {
                    B_ders[0] = alpha_index_basic[i][1]
                                * auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
                                * pow1
                                * pow2;
                }
                if (alpha_index_basic[i][2] != 0) {
                    B_ders[1] = alpha_index_basic[i][2]
                                * pow0
                                * auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
                                * pow2;
                }
                if (alpha_index_basic[i][3] != 0) {
                    B_ders[2] = alpha_index_basic[i][3]
                                * pow0
                                * pow1
                                * auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
                }
                C_ders[0] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[0];
                C_ders[1] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[1];
                C_ders[2] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[2];
                mom_vals[i] += coeffs[idx] * A * B * C;

                for (int aa=0; aa<3; aa++) {
                    CoordType tmp_deriv = coeffs[idx]
                                          * (A_ders[aa]*B*C + A*B_ders[aa]*C + A*B*C_ders[aa]);
                    dloss_combination[i] += 2*f_weight/(3*inum)
                                            * (force_ml[center_idx][aa] - force_dft[center_idx][aa])
                                            * tmp_deriv;
                    dloss_combination[i] -= 2*f_weight/(3*inum)
                                            * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa])
                                            * tmp_deriv;   
                }
            }
        }
    }

    for (int i=0; i<alpha_index_times_count; i++) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];
        mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
        dloss_combination[alpha_index_times[i][3]] += (dloss_combination[alpha_index_times[i][0]] * val2 * val1
                                                       + dloss_combination[alpha_index_times[i][1]] * val2 * val0);
    }

    // 2.1. NN Energy derivative w.r.t. mom_vals
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        for (int k=0; k<alpha_scalar_moments; k++)
            hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
        for (int k=0; k<alpha_scalar_moments; k++)
            e_site_der2mom[alpha_moment_mapping[k]] += type_central_w1[p]
                                                       * activated_hidden_der
                                                       * type_central_w0[p*alpha_scalar_moments + k];
    }

    // 2.2. Pass to basic moments
    for (int i=alpha_index_times_count-1; i>=0; i--) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];

        e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val1;
        e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val0;
    }
    
    // 2.3. loss derivative w.r.t. coeffs
    for (int jj=0; jj<snumneigh; jj++)
    {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        Zj = type_map[type_outer];
        for (int aa=0; aa<3; aa++)
            NeighbVect[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                 + std::pow(NeighbVect[1], 2)
                                 + std::pow(NeighbVect[2], 2) );
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                     rq_chebyshev_der2r,
                                     chebyshev_size,
                                     rmax,
                                     rmin,
                                     distance_ij);
        auto_dist_powers_[0] = 1.0;
        for (int aa=0; aa<3; aa++)
            auto_coords_powers_[0][aa] = 1.0;
        for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++) {
            int mu = alpha_index_basic[i][0];
            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            CoordType powk = 1.0 / auto_dist_powers_[k];
            CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
            CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
            CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
            CoordType mult0 = pow0 * pow1 * pow2;

            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes + type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                CoordType A = rq_chebyshev_vals[xi];
                CoordType B = mult0;
                CoordType C = powk;
                CoordType A_ders[3] = {0.0, 0.0, 0.0};
                CoordType B_ders[3] = {0.0, 0.0, 0.0};
                CoordType C_ders[3] = {0.0, 0.0, 0.0};
                A_ders[0] = rq_chebyshev_der2r[xi] * NeighbVect[0] * distance_ij_inv;
                A_ders[1] = rq_chebyshev_der2r[xi] * NeighbVect[1] * distance_ij_inv;
                A_ders[2] = rq_chebyshev_der2r[xi] * NeighbVect[2] * distance_ij_inv;
                if (alpha_index_basic[i][1] != 0) {
                    B_ders[0] = alpha_index_basic[i][1]
                                * auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
                                * pow1
                                * pow2;
                }
                if (alpha_index_basic[i][2] != 0) {
                    B_ders[1] = alpha_index_basic[i][2]
                                * pow0
                                * auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
                                * pow2;
                }
                if (alpha_index_basic[i][3] != 0) {
                    B_ders[2] = alpha_index_basic[i][3]
                                * pow0
                                * pow1
                                * auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
                }
                C_ders[0] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[0];
                C_ders[1] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[1];
                C_ders[2] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[2];

                CoordType tmpe_loss_der2coeff = 2*e_weight/inum*(etot_ml - etot_dft)
                                                * e_site_der2mom[i]
                                                * A * B * C;

                CoordType tmpf_loss_der2coeff = 0;
                for (int aa=0; aa<3; aa++) 
                {
                    CoordType tmp_prefix = 0;
                    CoordType tmp_deriv = (A_ders[aa] * B * C
                                           + A * B_ders[aa] * C
                                           + A * B * C_ders[aa]);
                    tmp_prefix += 2*f_weight/(3*inum)
                                  * (force_ml[center_idx][aa] - force_dft[center_idx][aa]);
                    tmp_prefix -= 2*f_weight/(3*inum)
                                  * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa]);
                    tmpf_loss_der2coeff += tmp_prefix * tmp_deriv;
                }
                atomicAdd(&loss_der2coeffs[idx],
                          tmpe_loss_der2coeff
                          + tmpf_loss_der2coeff * e_site_der2mom[i]);
            }
        }
    }


    // 2.4. loss derivative w.r.t. w1
    for (int p=0; p<num_neurons; p++) {
        CoordType tmpe_loss_der2w1 = 0.0;
        CoordType tmpf_loss_der2w1 = 0.0;
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        for (int k=0; k<alpha_scalar_moments; k++)
            hidden_val += type_central_w0[p*alpha_scalar_moments+k] * mom_vals[alpha_moment_mapping[k]];
        TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
        tmpe_loss_der2w1 = 2*e_weight/inum*(etot_ml - etot_dft)
                           * activated_hidden_val;
        for (int k=0; k<alpha_scalar_moments; k++)
            tmpf_loss_der2w1 += dloss_combination[alpha_moment_mapping[k]]
                                * activated_hidden_der
                                * type_central_w0[p*alpha_scalar_moments + k];
        
        atomicAdd(&loss_der2w1[type_central*num_neurons + p], tmpe_loss_der2w1 + tmpf_loss_der2w1);
    }
    
    // 2.5. loss derivative w.r.t. w0
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        CoordType activated_hidden_der2der = 0.0;
        for (int k=0; k<alpha_scalar_moments; k++)
            hidden_val += type_central_w0[p*alpha_scalar_moments+k] * mom_vals[alpha_moment_mapping[k]];
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
        TanhActivationFunc<CoordType>::find_der2der(activated_hidden_der2der, hidden_val);
        for (int k=0; k<alpha_scalar_moments; k++) {
            CoordType tmpe_loss_der2w0 = 2*e_weight/inum*(etot_ml - etot_dft)
                                         * type_central_w1[p]
                                         * activated_hidden_der
                                         * mom_vals[alpha_moment_mapping[k]];
            CoordType tmpf_loss_der2w0 = dloss_combination[alpha_moment_mapping[k]]
                                         * type_central_w1[p]
                                         * (activated_hidden_der
                                            + activated_hidden_der2der
                                              * type_central_w0[p*alpha_scalar_moments+k]
                                              * mom_vals[alpha_moment_mapping[k]]);

            atomicAdd(&loss_der2w0[type_central*num_neurons*alpha_scalar_moments + p*alpha_scalar_moments + k], tmpe_loss_der2w0 + tmpf_loss_der2w0);
        }
    }


    // 2.6. loss derivative w.r.t. type_bias
    atomicAdd(&loss_der2type_bias[type_central],
              2*e_weight/inum*(etot_ml - etot_dft));
}


template <typename CoordType>
__global__
void find_ef_loss_backward_kernel(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2w0,
    CoordType *bloss_der2w1,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int ii = nx % natoms_pad;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    CoordType *loss_der2coeffs = &bloss_der2coeffs[istruct*num_coeffs + 0];
    CoordType *loss_der2w0 = &bloss_der2w0[istruct*ntypes*num_neurons*alpha_scalar_moments+ 0];
    CoordType *loss_der2w1 = &bloss_der2w1[istruct*ntypes*num_neurons + 0];
    CoordType *loss_der2type_bias = &bloss_der2type_bias[istruct*ntypes + 0];
    CoordType etot_ml = betot_ml[istruct];
    CoordType etot_dft = betot_dft[istruct];
    CoordType (*force_ml)[3] = &bforce_ml[istruct*natoms_pad + 0];
    CoordType (*force_dft)[3] = &bforce_dft[istruct*natoms_pad + 0];
    int inum = binum[istruct];
    int *types = &btypes[istruct*natoms_pad + 0];
    int *ilist = &bilist[istruct*natoms_pad + 0];
    int *numneigh = &bnumneigh[istruct*natoms_pad + 0];
    int *firstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + 0];
    CoordType (*rcs)[3] = &brcs[istruct*natoms_pad*umax_num_neigh_atoms + 0];

    if (ii < inum) {
        int silist = ilist[ii];
        int snumneigh = numneigh[ii];
        int *sfirstneigh = &firstneigh[ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = (CoordType (*)[3])(&rcs[ii*umax_num_neigh_atoms][0]);
        find_ef_loss_backward_atom<CoordType>(
            loss_der2coeffs,
            loss_der2w0,
            loss_der2w1,
            loss_der2type_bias,
            e_weight,
            f_weight,
            etot_ml,
            etot_dft,
            force_ml,
            force_dft,
            chebyshev_size,
            num_neurons,
            coeffs,
            w0,
            w1,
            type_bias,
            alpha_moments_count,
            alpha_index_basic_count,
            alpha_index_basic,
            alpha_index_times_count,
            alpha_index_times,
            alpha_scalar_moments,
            alpha_moment_mapping,
            nmus,
            inum,
            silist,
            snumneigh,
            sfirstneigh,
            srcs,
            types,
            ntypes,
            type_map,
            umax_num_neigh_atoms,
            nghost,
            rmax,
            rmin);
    }
}


template <typename CoordType>
__host__
void find_ef_loss_backward_launcher(
    CoordType *h_bloss_der2coeffs,
    CoordType *h_bloss_der2w0,
    CoordType *h_bloss_der2w1,
    CoordType *h_bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3],
    int chebyshev_size,
    int num_neurons,
    CoordType *h_coeffs,
    CoordType *h_w0,
    CoordType *h_w1,
    CoordType *h_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*h_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*h_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *h_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    int *h_bnumneigh,
    int *h_bfirstneigh,
    CoordType (*h_brcs)[3],
    int *h_btypes,
    int ntypes,
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    CoordType *d_bloss_der2coeffs;
    CoordType *d_bloss_der2w0;
    CoordType *d_bloss_der2w1;
    CoordType *d_bloss_der2type_bias;
    CoordType *d_betot_ml;
    CoordType *d_betot_dft;
    CoordType (*d_bforce_ml)[3];
    CoordType (*d_bforce_dft)[3];
    CoordType *d_coeffs;
    CoordType *d_w0;
    CoordType *d_w1;
    CoordType *d_type_bias;
    int (*d_alpha_index_basic)[4];
    int (*d_alpha_index_times)[4];
    int *d_alpha_moment_mapping;
    int *d_binum;
    int *d_bilist;
    int *d_bnumneigh;
    int *d_bfirstneigh;
    CoordType (*d_brcs)[3];
    int *d_btypes;
    int *d_type_map;

    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2coeffs, sizeof(CoordType)*batch_size*num_coeffs) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2coeffs, 0.0, sizeof(CoordType)*batch_size*num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2w0, sizeof(CoordType)*batch_size*ntypes*num_neurons*alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2w0, 0.0, sizeof(CoordType)*batch_size*ntypes*num_neurons*alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2w1, sizeof(CoordType)*batch_size*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2w1, 0.0, sizeof(CoordType)*batch_size*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2type_bias, sizeof(CoordType)*batch_size*ntypes) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2type_bias, 0.0, sizeof(CoordType)*batch_size*ntypes) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ml, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_dft, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_w0, sizeof(CoordType) * ntypes * num_neurons * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_w1, sizeof(CoordType) * ntypes * num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );

    CHECK_CUDA_API( cudaMemcpy(d_betot_ml, h_betot_ml, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_dft, h_betot_dft, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_ml, h_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_dft, h_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*num_coeffs, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_w0, h_w0, sizeof(CoordType)*ntypes*num_neurons*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_w1, h_w1, sizeof(CoordType)*ntypes*num_neurons, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int)*alpha_index_times_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );


    auto t1 = std::chrono::high_resolution_clock::now();
    find_ef_loss_backward_kernel KERNEL_ARG2(grid_size, block_size) (
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
        rmin);
    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_ef_loss_backward_launcher() cost time: " << duration.count() << " us." << std::endl;


    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2coeffs, d_bloss_der2coeffs, sizeof(CoordType)*batch_size*num_coeffs, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2w0, d_bloss_der2w0, sizeof(CoordType)*batch_size*ntypes*num_neurons*alpha_scalar_moments, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2w1, d_bloss_der2w1, sizeof(CoordType)*batch_size*ntypes*num_neurons, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2type_bias, d_bloss_der2type_bias, sizeof(CoordType)*batch_size*ntypes, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_bloss_der2coeffs) );
    CHECK_CUDA_API( cudaFree(d_bloss_der2w0) );
    CHECK_CUDA_API( cudaFree(d_bloss_der2w1) );
    CHECK_CUDA_API( cudaFree(d_bloss_der2type_bias) );
    CHECK_CUDA_API( cudaFree(d_betot_ml) );
    CHECK_CUDA_API( cudaFree(d_betot_dft) );
    CHECK_CUDA_API( cudaFree(d_bforce_ml) );
    CHECK_CUDA_API( cudaFree(d_bforce_dft) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_w0) );
    CHECK_CUDA_API( cudaFree(d_w1) );
    CHECK_CUDA_API( cudaFree(d_type_bias) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_basic) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_times) );
    CHECK_CUDA_API( cudaFree(d_alpha_moment_mapping) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_bnumneigh) );
    CHECK_CUDA_API( cudaFree(d_bfirstneigh) );
    CHECK_CUDA_API( cudaFree(d_brcs) );
    CHECK_CUDA_API( cudaFree(d_btypes) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
}




};  // namespace : nnmtp
};  // namespace : ai2pot


#endif