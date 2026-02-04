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

#include "./nep_basis.cuh"
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
static __device__
void find_ef_loss_backward_atom(
    CoordType *loss_der2coeffs,
    CoordType *loss_der2w0,
    CoordType *loss_der2b0,
    CoordType *loss_der2w1,
    CoordType *loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *b0,
    CoordType *w1,
    CoordType *type_bias,
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
    CoordType rmax_radial,
    CoordType rmax_angular,
    CoordType *q_scaler);


template <typename CoordType>
static __global__
void find_ef_loss_backward_kernel(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2w0,
    CoordType *bloss_der2b0,
    CoordType *bloss_der2w1,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *b0,
    CoordType *w1,
    CoordType *type_bias,
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
    CoordType rmax_radial,
    CoordType rmax_angular,
    CoordType *q_scaler);


template <typename CoordType>
static __host__
void find_ef_loss_backward_launcher(
    CoordType *h_bloss_der2coeffs,
    CoordType *h_bloss_der2w0,
    CoordType *h_bloss_der2b0,
    CoordType *h_bloss_der2w1,
    CoordType *h_bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    CoordType *h_coeffs,
    CoordType *h_w0,
    CoordType *h_b0,
    CoordType *h_w1,
    CoordType *h_type_bias,
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
    CoordType rmax_radial,
    CoordType rmax_angular,
    CoordType *h_q_scaler);


template <typename CoordType>
static __global__ 
void rescale_f_kernel(
    CoordType (*bforce_ml)[3],
    int batch_size,
    int natoms_pad,
    int *binum,
    CoordType force_scaler);


template <typename CoordType>
static __host__ 
void rescale_f_launcher(
    CoordType (*h_bforce_ml)[3],
    int batch_size,
    int natoms_pad,
    int *h_binum,
    CoordType force_scaler);


template <typename CoordType>
static __global__ 
void rescale_fv_kernel(
    CoordType (*bforce_ml)[3],
    CoordType *bvirial_ml,
    int batch_size,
    int natoms_pad,
    int *binum,
    CoordType force_scaler);


template <typename CoordType>
static __host__ 
void rescale_fv_launcher(
    CoordType (*h_bforce_ml)[3],
    CoordType *h_bvirial_ml,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    CoordType force_scaler);






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
void find_ef_loss_backward_atom(
    CoordType *loss_der2coeffs,
    CoordType *loss_der2w0,
    CoordType *loss_der2b0,
    CoordType *loss_der2w1,
    CoordType *loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *b0,
    CoordType *w1,
    CoordType *type_bias,
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
    CoordType rmax_radial,
    CoordType rmax_angular,
    CoordType *q_scaler)
{
    // Step 1.
    CoordType mom_vals[MAX_NUM_SINLM] = {0.0};
    CoordType e_sites_der2mom[MAX_NUM_SINLM] = {0.0};
    CoordType dloss_combination_mom[MAX_NUM_SINLM] = {0.0};
    CoordType dod_vals[MAX_NUM_DESCRIPTORS] = {0.0};
    CoordType e_sites_der2dod[MAX_NUM_DESCRIPTORS] = {0.0};
    CoordType dloss_combination_dod[MAX_NUM_DESCRIPTORS] = {0.0};
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_Sinlm = NepIndex::get_num_Sinlm(n_angular_basis, l_max);

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    CoordType auto_dist_powers_[L_MAX_PLUS_ONE] = {0.0};

    center_idx = silist;
    type_central = types[center_idx];
    CoordType *type_central_w0 = &w0[type_central*num_neurons*num_descriptors];
    CoordType *type_central_b0 = &b0[type_central*num_neurons];
    CoordType *type_central_w1 = &w1[type_central*num_neurons];

    // Step 2. Calculate descriptors
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                 + std::pow(neigh_vec[1], 2)
                                 + std::pow(neigh_vec[2], 2) );
        if (distance_ij > rmax_radial)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals_radial[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r_radial[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_vals_angular[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r_angular[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals_radial,
                                     rq_chebyshev_der2r_radial,
                                     chebyshev_size,
                                     rmax_radial,
                                     distance_ij);
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals_angular,
                                     rq_chebyshev_der2r_angular,
                                     chebyshev_size,
                                     rmax_angular,
                                     distance_ij);
        
        auto_dist_powers_[0] = 1.0;
        for (int k=1; k<=l_max; k++)
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
        
        // Step 2.1. Radial Forward
        for (int mu=0; mu<n_radial_basis; mu++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                          + mu*chebyshev_size + xi;
                dod_vals[mu] += coeffs[idx] * rq_chebyshev_vals_radial[xi];
                for (int aa=0; aa<3; aa++) {
                    dloss_combination_dod[mu] += 2*f_weight/(3*inum)
                                                 * (force_ml[center_idx][aa] - force_dft[center_idx][aa])
                                                 * coeffs[idx] * rq_chebyshev_der2r_radial[xi] * neigh_vec[aa] * distance_ij_inv;
                    dloss_combination_dod[mu] -= 2*f_weight/(3*inum)
                                                 * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa])
                                                 * coeffs[idx] * rq_chebyshev_der2r_radial[xi] * neigh_vec[aa] * distance_ij_inv;
                }
            }
        }

        // Step 2.2. Angular forward: basic
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    CoordType B = 0.0;
                    Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                    CoordType C = 1.0 / auto_dist_powers_[l];
                    CoordType B_ders[3] = {0.0};
                    CoordType C_ders[3] = {0.0};
                    Blm<CoordType>::find_blm_der2xyz(B_ders, l, mp, neigh_vec, distance_ij);
                    for (int aa=0; aa<3; aa++)
                        C_ders[aa] = -l / (auto_dist_powers_[l] * distance_ij) * (neigh_vec[aa] * distance_ij_inv);

                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                  + (n_radial_basis+mu)*chebyshev_size + xi;

                        CoordType A = rq_chebyshev_vals_angular[xi];
                        CoordType A_ders[3] = {0.0};
                        for (int aa=0; aa<3; aa++)
                            A_ders[aa] = rq_chebyshev_der2r_angular[xi] * neigh_vec[aa] * distance_ij_inv;
                        
                        mom_vals[idx_Sinlm] += coeffs[idx] * A * B * C;

                        for (int aa=0; aa<3; aa++) {
                            CoordType tmp_deriv = coeffs[idx] * (A_ders[aa] * B * C
                                                                 + A * B_ders[aa] * C
                                                                 + A * B * C_ders[aa]);
                            dloss_combination_mom[idx_Sinlm] += 2*f_weight/(3*inum)
                                                                * (force_ml[center_idx][aa] - force_dft[center_idx][aa])
                                                                * tmp_deriv;
                            dloss_combination_mom[idx_Sinlm] -= 2*f_weight/(3*inum)
                                                                * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa])
                                                                * tmp_deriv;
                        }
                    }
                }
            }
        }
    }

    // Step 2.3. Angular forward ; times
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            for (int mp=0; mp<2*l+1; mp++) {
                int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                int idx_Clm = NepIndex::get_Clm_index(l, mp);
                int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);

                if (mp == 0) {
                    dod_vals[n_radial_basis+idx_qinl] += (CoordType)C3B[idx_Clm] * std::pow(mom_vals[idx_Sinlm], 2);
                    dloss_combination_dod[n_radial_basis+idx_qinl] += dloss_combination_mom[idx_Sinlm]
                                                                      * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
                } else {
                    dod_vals[n_radial_basis+idx_qinl] += 2 * (CoordType)C3B[idx_Clm] * std::pow(mom_vals[idx_Sinlm], 2);
                    dloss_combination_dod[n_radial_basis+idx_qinl] += 2 * dloss_combination_mom[idx_Sinlm]
                                                                      * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
                }
            }
        }
    }

    // Step 3. Backward
    // Step 3.1. 
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        for (int k=0; k<num_descriptors; k++)
            hidden_val += type_central_w0[p*num_descriptors+k] * dod_vals[k] / q_scaler[k];
        hidden_val += type_central_b0[p];
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);

        for (int k=0; k<num_descriptors; k++)
            e_sites_der2dod[k] += type_central_w1[p]
                                  * activated_hidden_der
                                  * type_central_w0[p*num_descriptors + k]
                                  / q_scaler[k];
    }

    // Step 3.2. 
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            for (int mp=0; mp<2*l+1; mp++) {
                int idx_Clm = NepIndex::get_Clm_index(l, mp);
                int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);

                if (mp == 0) {
                    e_sites_der2mom[idx_Sinlm] = e_sites_der2dod[n_radial_basis + idx_qinl]
                                                 * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
                } else {
                    e_sites_der2mom[idx_Sinlm] = 2 * e_sites_der2dod[n_radial_basis + idx_qinl]
                                                 * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
                }
            }
        }
    }

    // Step 4.1. loss_der2coeffs
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                 + std::pow(neigh_vec[1], 2)
                                 + std::pow(neigh_vec[2], 2));
        if (distance_ij > rmax_radial)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals_radial[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r_radial[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_vals_angular[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r_angular[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals_radial,
                                     rq_chebyshev_der2r_radial,
                                     chebyshev_size,
                                     rmax_radial,
                                     distance_ij);
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals_angular,
                                     rq_chebyshev_der2r_angular,
                                     chebyshev_size,
                                     rmax_angular,
                                     distance_ij);
        auto_dist_powers_[0] = 1.0;
        for (int k=1; k<=l_max; k++)
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
        
        // Step 4.1.1. loss_der2coeffs -- Radial
        for (int mu=0; mu<n_radial_basis; mu++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                          + mu*chebyshev_size + xi;
                
                CoordType tmpe_loss_der2coeff = 0.0;
                tmpe_loss_der2coeff = 2*e_weight/inum*(etot_ml - etot_dft)
                                      * e_sites_der2dod[mu]
                                      * rq_chebyshev_vals_radial[xi];
                
                CoordType tmpf_loss_der2coeff = 0.0;
                for (int aa=0; aa<3; aa++) {
                    CoordType tmp_prefix = 0.0;
                    CoordType tmp_deriv = rq_chebyshev_der2r_radial[xi] * neigh_vec[aa] * distance_ij_inv;

                    tmp_prefix += 2*f_weight/(3*inum)
                                  * (force_ml[center_idx][aa] - force_dft[center_idx][aa]);
                    tmp_prefix -= 2*f_weight/(3*inum)
                                  * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa]);
                    tmpf_loss_der2coeff += tmp_prefix * tmp_deriv;
                }
                atomicAdd(&loss_der2coeffs[idx],
                          tmpe_loss_der2coeff
                          + tmpf_loss_der2coeff * e_sites_der2dod[mu]);
            }
        }

        // Step 4.1.2. loss_der2coeffs -- Angular
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Clm = NepIndex::get_Clm_index(l, mp);
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);
                    CoordType B = 0.0;
                    Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                    CoordType C = 1.0 / auto_dist_powers_[l];
                    CoordType B_ders[3] = {0.0};
                    CoordType C_ders[3] = {0.0};
                    Blm<CoordType>::find_blm_der2xyz(B_ders, l, mp, neigh_vec, distance_ij);
                    for (int aa=0; aa<3; aa++)
                        C_ders[aa] = -l / (auto_dist_powers_[l]*distance_ij)
                                     * (neigh_vec[aa] * distance_ij_inv);

                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                  + (n_radial_basis+mu)*chebyshev_size + xi;
                        
                        CoordType A = rq_chebyshev_vals_angular[xi];
                        CoordType A_ders[3] = {0.0};
                        for (int aa=0; aa<3; aa++)
                            A_ders[aa] = rq_chebyshev_der2r_angular[xi] * neigh_vec[aa] * distance_ij_inv;
                        
                        CoordType tmpe_loss_der2coeff = 0.0;
                        tmpe_loss_der2coeff = 2*e_weight/inum*(etot_ml-etot_dft)
                                              * e_sites_der2mom[idx_Sinlm]
                                              * A * B * C;
                        CoordType tmpf_loss_der2coeff = 0.0;
                        for (int aa=0; aa<3; aa++) {
                            CoordType tmp_prefix = 0.0;
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
                                  + tmpf_loss_der2coeff * e_sites_der2mom[idx_Sinlm]);
                    }
                }
            }
        }
    }

    // Step 4.2. loss_der2w0
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        CoordType activated_hidden_der2der = 0.0;
        for (int k=0; k<num_descriptors; k++)
            hidden_val += type_central_w0[p*num_descriptors+k] * dod_vals[k] / q_scaler[k];
        hidden_val += type_central_b0[p];
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
        TanhActivationFunc<CoordType>::find_der2der(activated_hidden_der2der, hidden_val);

        // New code
        CoordType dloss_combination_dod_sum = 0.0;
        for (int k=0; k<num_descriptors; k++)
            dloss_combination_dod_sum += dloss_combination_dod[k]
                                         * type_central_w0[p*num_descriptors + k]
                                         / q_scaler[k];
        for (int k=0; k<num_descriptors; k++) {
            CoordType tmpe_loss_der2w0 = 2*e_weight/inum*(etot_ml-etot_dft)
                                         * type_central_w1[p]
                                         * activated_hidden_der
                                         * dod_vals[k]
                                         / q_scaler[k];
            CoordType tmpf_loss_der2w0_p1 = activated_hidden_der2der
                                            * dod_vals[k]
                                            * dloss_combination_dod_sum;
            CoordType tmpf_loss_der2w0_p2 = activated_hidden_der
                                            * dloss_combination_dod[k];
            CoordType tmpf_loss_der2w0 = type_central_w1[p] / q_scaler[k]
                                         * (tmpf_loss_der2w0_p1 + tmpf_loss_der2w0_p2);

            atomicAdd(&loss_der2w0[type_central*num_neurons*num_descriptors + p*num_descriptors + k],
                      tmpe_loss_der2w0 + tmpf_loss_der2w0);
        }
        /*
        // Old code : Wrong
        for (int k=0; k<num_descriptors; k++) {
            CoordType tmpe_loss_der2w0 = 2*e_weight/inum*(etot_ml-etot_dft)
                                         * type_central_w1[p]
                                         * activated_hidden_der
                                         * dod_vals[k]
                                         / q_scaler[k];
            CoordType tmpf_loss_der2w0 = dloss_combination_dod[k]
                                         * type_central_w1[p]
                                         / q_scaler[k]
                                         * (activated_hidden_der
                                            + activated_hidden_der2der
                                              * type_central_w0[p*num_descriptors+k]
                                              / q_scaler[k]
                                              * dod_vals[k]);
            atomicAdd(&loss_der2w0[type_central*num_neurons*num_descriptors + p*num_descriptors + k],
                      tmpe_loss_der2w0 + tmpf_loss_der2w0);
        }
        */
       
        CoordType tmpe_loss_der2b0 = 2*e_weight/inum*(etot_ml-etot_dft)
                                     * type_central_w1[p]
                                     * activated_hidden_der;
        CoordType tmpf_loss_der2b0 = type_central_w1[p]
                                     * activated_hidden_der2der
                                     * dloss_combination_dod_sum;
        atomicAdd(&loss_der2b0[type_central*num_neurons + p], tmpe_loss_der2b0 + tmpf_loss_der2b0);
    }

    // Step 4.3. loss_der2w1
    for (int p=0; p<num_neurons; p++) {
        CoordType tmpe_loss_der2w1 = 0.0;
        CoordType tmpf_loss_der2w1 = 0.0;
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        for (int k=0; k<num_descriptors; k++)
            hidden_val += type_central_w0[p*num_descriptors+k] * dod_vals[k] / q_scaler[k];
        hidden_val += type_central_b0[p];
        TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
        tmpe_loss_der2w1 = 2*e_weight/inum*(etot_ml-etot_dft)
                           * activated_hidden_val;
        for (int k=0; k<num_descriptors; k++) {
            tmpf_loss_der2w1 += dloss_combination_dod[k]
                                * activated_hidden_der
                                * type_central_w0[p*num_descriptors + k]
                                / q_scaler[k];
        }
        atomicAdd(&loss_der2w1[type_central*num_neurons+p],
                  tmpe_loss_der2w1 + tmpf_loss_der2w1);
    }

    // Step 4.4. loss_der2type_bias
    atomicAdd(&loss_der2type_bias[type_central], 2*e_weight/inum*(etot_ml-etot_dft));
}


template <typename CoordType>
__global__
void find_ef_loss_backward_kernel(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2w0,
    CoordType *bloss_der2b0,
    CoordType *bloss_der2w1,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *b0,
    CoordType *w1,
    CoordType *type_bias,
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
    CoordType rmax_radial,
    CoordType rmax_angular,
    CoordType *q_scaler)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int ii = nx % natoms_pad;
    int num_coeffs = ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size;
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);

    CoordType *loss_der2coeffs = &bloss_der2coeffs[istruct*num_coeffs + 0];
    CoordType *loss_der2w0 = &bloss_der2w0[istruct*ntypes*num_neurons*num_descriptors+ 0];
    CoordType *loss_der2b0 = &bloss_der2b0[istruct*ntypes*num_neurons + 0];
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
            loss_der2b0,
            loss_der2w1,
            loss_der2type_bias,
            e_weight,
            f_weight,
            etot_ml,
            etot_dft,
            force_ml,
            force_dft,
            chebyshev_size,
            n_radial_basis,
            n_angular_basis,
            l_max,
            num_neurons,
            coeffs,
            w0,
            b0,
            w1,
            type_bias,
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
            rmax_radial,
            rmax_angular,
            q_scaler);
    }
}


template <typename CoordType>
__host__
void find_ef_loss_backward_launcher(
    CoordType *h_bloss_der2coeffs,
    CoordType *h_bloss_der2w0,
    CoordType *h_bloss_der2b0,
    CoordType *h_bloss_der2w1,
    CoordType *h_bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *h_betot_ml,
    CoordType *h_betot_dft,
    CoordType (*h_bforce_ml)[3],
    CoordType (*h_bforce_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    CoordType *h_coeffs,
    CoordType *h_w0,
    CoordType *h_b0,
    CoordType *h_w1,
    CoordType *h_type_bias,
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
    CoordType rmax_radial,
    CoordType rmax_angular,
    CoordType *h_q_scaler)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    int num_coeffs = ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size;
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    CoordType *d_bloss_der2coeffs;
    CoordType *d_bloss_der2w0;
    CoordType *d_bloss_der2b0;
    CoordType *d_bloss_der2w1;
    CoordType *d_bloss_der2type_bias;
    CoordType *d_betot_ml;
    CoordType *d_betot_dft;
    CoordType (*d_bforce_ml)[3];
    CoordType (*d_bforce_dft)[3];
    CoordType *d_coeffs;
    CoordType *d_w0;
    CoordType *d_b0;
    CoordType *d_w1;
    CoordType *d_type_bias;
    int *d_binum;
    int *d_bilist;
    int *d_bnumneigh;
    int *d_bfirstneigh;
    CoordType (*d_brcs)[3];
    int *d_btypes;
    int *d_type_map;
    CoordType *d_q_scaler;

    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2coeffs, sizeof(CoordType)*batch_size*num_coeffs) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2coeffs, 0.0, sizeof(CoordType)*batch_size*num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2w0, sizeof(CoordType)*batch_size*ntypes*num_neurons*num_descriptors) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2w0, 0.0, sizeof(CoordType)*batch_size*ntypes*num_neurons*num_descriptors) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2b0, sizeof(CoordType)*batch_size*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2b0, 0.0, sizeof(CoordType)*batch_size*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2w1, sizeof(CoordType)*batch_size*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2w1, 0.0, sizeof(CoordType)*batch_size*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bloss_der2type_bias, sizeof(CoordType)*batch_size*ntypes) );
    CHECK_CUDA_API( cudaMemset(d_bloss_der2type_bias, 0.0, sizeof(CoordType)*batch_size*ntypes) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ml, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_dft, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_w0, sizeof(CoordType) * ntypes * num_neurons * num_descriptors) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_b0, sizeof(CoordType) * ntypes * num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_w1, sizeof(CoordType) * ntypes * num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_q_scaler, sizeof(CoordType)*num_descriptors) );

    //
    CHECK_CUDA_API( cudaMemcpy(d_betot_ml, h_betot_ml, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_betot_dft, h_betot_dft, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_ml, h_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_dft, h_bforce_dft, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*num_coeffs, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_w0, h_w0, sizeof(CoordType)*ntypes*num_neurons*num_descriptors, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_b0, h_b0, sizeof(CoordType)*ntypes*num_neurons, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_w1, h_w1, sizeof(CoordType)*ntypes*num_neurons, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_q_scaler, h_q_scaler, sizeof(CoordType)*num_descriptors, cudaMemcpyHostToDevice) );

    // Compute
    auto t1 = std::chrono::high_resolution_clock::now();
    find_ef_loss_backward_kernel KERNEL_ARG2(grid_size, block_size) (
        d_bloss_der2coeffs,
        d_bloss_der2w0,
        d_bloss_der2b0,
        d_bloss_der2w1,
        d_bloss_der2type_bias,
        e_weight,
        f_weight,
        d_betot_ml,
        d_betot_dft,
        d_bforce_ml,
        d_bforce_dft,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        d_coeffs,
        d_w0,
        d_b0,
        d_w1,
        d_type_bias,
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
        rmax_radial,
        rmax_angular,
        d_q_scaler);
    CHECK_CUDA_KERNEL;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_ef_loss_backward_launcher() cost time: " << duration.count() << " us." << std::endl;

    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2coeffs, d_bloss_der2coeffs, sizeof(CoordType)*batch_size*num_coeffs, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2w0, d_bloss_der2w0, sizeof(CoordType)*batch_size*ntypes*num_neurons*num_descriptors, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2b0, d_bloss_der2b0, sizeof(CoordType)*batch_size*ntypes*num_neurons, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2w1, d_bloss_der2w1, sizeof(CoordType)*batch_size*ntypes*num_neurons, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bloss_der2type_bias, d_bloss_der2type_bias, sizeof(CoordType)*batch_size*ntypes, cudaMemcpyDeviceToHost) );

    // Step . Free
    CHECK_CUDA_API( cudaFree(d_bloss_der2coeffs) );
    CHECK_CUDA_API( cudaFree(d_bloss_der2w0) );
    CHECK_CUDA_API( cudaFree(d_bloss_der2b0) );
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
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_bnumneigh) );
    CHECK_CUDA_API( cudaFree(d_bfirstneigh) );
    CHECK_CUDA_API( cudaFree(d_brcs) );
    CHECK_CUDA_API( cudaFree(d_btypes) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
    CHECK_CUDA_API( cudaFree(d_q_scaler) );
}


template <typename CoordType>
__global__ 
void rescale_f_kernel(
    CoordType (*bforce_ml)[3],
    int batch_size,
    int natoms_pad,
    int *binum,
    CoordType force_scaler)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int ii = nx % natoms_pad;

    if (ii < binum[istruct])
        for (int aa=0; aa<3; aa++)
            bforce_ml[istruct*natoms_pad + ii][aa] = bforce_ml[istruct*natoms_pad + ii][aa] / force_scaler;
}


template <typename CoordType>
__host__
void rescale_f_launcher(
    CoordType (*h_bforce_ml)[3],
    int batch_size,
    int natoms_pad,
    int *h_binum,
    CoordType force_scaler)
{
    int block_size_x = 256;
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType (*d_bforce_ml)[3];
    int *d_binum;

    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    
    CHECK_CUDA_API( cudaMemcpy(d_bforce_ml, h_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );

    rescale_f_kernel KERNEL_ARG2(grid_size, block_size) (
        d_bforce_ml,
        batch_size,
        natoms_pad,
        d_binum,
        force_scaler);
    CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaMemcpy(h_bforce_ml, d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_bforce_ml) );
    CHECK_CUDA_API( cudaFree(d_binum) );
}


template <typename CoordType>
__global__
void rescale_fv_kernel(
    CoordType (*bforce_ml)[3],
    CoordType *bvirial_ml,
    int batch_size,
    int natoms_pad,
    int *binum,
    CoordType force_scaler)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int ii = nx % natoms_pad;

    if (ii < binum[istruct]) {
        for (int aa=0; aa<3; aa++)
            bforce_ml[istruct*natoms_pad + ii][aa] = bforce_ml[istruct*natoms_pad + ii][aa] / force_scaler;
        
        if (ii == 0) {
            for (int aa=0; aa<3; aa++)
                for (int bb=0; bb<3; bb++)
                    bvirial_ml[istruct*9 + aa*3 + bb] = bvirial_ml[istruct*9 + aa*3 + bb] / force_scaler;
        }
    }
}


template <typename CoordType>
__host__
void rescale_fv_launcher(
    CoordType (*h_bforce_ml)[3],
    CoordType *h_bvirial_ml,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    CoordType force_scaler)
{
    int block_size_x = 256;
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType (*d_bforce_ml)[3];
    CoordType *d_bvirial_ml;
    int *d_binum;

    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_ml, sizeof(CoordType)*batch_size*9) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );

    CHECK_CUDA_API( cudaMemcpy(d_bforce_ml, h_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bvirial_ml, h_bvirial_ml, sizeof(CoordType)*batch_size*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );

    rescale_fv_kernel KERNEL_ARG2(grid_size, block_size) (
        d_bforce_ml,
        d_bvirial_ml,
        batch_size,
        natoms_pad,
        d_binum,
        force_scaler);
    CHECK_CUDA_KERNEL;

    CHECK_CUDA_API( cudaMemcpy(h_bforce_ml, d_bforce_ml, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bvirial_ml, d_bvirial_ml, sizeof(CoordType)*batch_size*9, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_bforce_ml) );
    CHECK_CUDA_API( cudaFree(d_bvirial_ml) );
    CHECK_CUDA_API( cudaFree(d_binum) );
}


};  // namespace : nep
};  // namespace : ai2pot

#endif