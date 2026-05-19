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

#ifndef AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_CUH
#define AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_CUH


#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <iostream>

#include "./mtp_utilities.cuh"
#include "./mtpBasis.cuh"


namespace ai2pot {
namespace mtpr {


template <typename CoordType>
static __device__
void accumulate_efv_components_atom(
    CoordType *energy_components,
    CoordType *force_components,
    CoordType *virial_components,
    int chebyshev_size,
    CoordType *coeffs,
    CoordType *linear_coeffs,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srcs)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_scaler);


template <typename CoordType>
static __global__
void find_efv_components_kernel(
    CoordType *benergy_components,
    CoordType *bforce_components,
    CoordType *bvirial_components,
    int chebyshev_size,
    CoordType *coeffs,
    CoordType *linear_coeffs,
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
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_scaler);


template <typename CoordType>
static __host__
void find_efv_components_launcher(
    CoordType *h_benergy_components,
    CoordType *h_bforce_components,
    CoordType *h_bvirial_components,
    int chebyshev_size,
    CoordType *h_coeffs,
    CoordType *h_linear_coeffs,
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
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);


template <typename CoordType>
static __global__
void find_lin_matrix_lin_vector_kernel(
    CoordType *lin_matrix,
    CoordType *lin_vector,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_residual,
    CoordType (*bforce_residual)[3],
    CoordType *bvirial_residual,
    CoordType *benergy_components,
    CoordType *bforce_components,
    CoordType *bvirial_components,
    int alpha_scalar_moments,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int ntypes);

template <typename CoordType>
static __global__
void mirror_lin_matrix(
    CoordType *lin_matrix,
    int num_parameters);

template <typename CoordType>
static __host__
void find_lin_matrix_lin_vector_launcher(
    CoordType *h_lin_matrix,
    CoordType *h_lin_vector,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *h_betot_residual,
    CoordType (*h_bforce_residual)[3],
    CoordType *h_bvirial_residual,
    int chebyshev_size,
    CoordType *h_coeffs,
    CoordType *h_linear_coeffs,
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
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_q_scaler);






template <typename CoordType>
__device__
void accumulate_efv_components_atom(
    CoordType *energy_components,
    CoordType *force_components,
    CoordType *virial_components,
    int chebyshev_size,
    CoordType *coeffs,
    CoordType *linear_coeffs,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srcs)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_scaler)
{
    // Step 1. 
    CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.};
    CoordType moms_der2xyz[MAX_ALPHA_MOMENTS_COUNT] = {0.};
    CoordType auto_dist_powers_[MAX_ALPHA_INDEX_BASIC];
    CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3];
    int num_parameters = alpha_scalar_moments + ntypes;

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    center_idx = silist;
    type_central = types[center_idx];

    // Step 2.
    // Step 2.1. energy_components
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        distance_ij = sqrt( neigh_vec[0] * neigh_vec[0]
                            + neigh_vec[1] * neigh_vec[1]
                            + neigh_vec[2] * neigh_vec[2] );
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                     rq_chebyshev_ders2r,
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
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * neigh_vec[aa];
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
                int idx = (type_central*ntypes+type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                CoordType A = rq_chebyshev_vals[xi];
                CoordType B = mult0;
                CoordType C = powk;
                mom_vals[i] += coeffs[idx] * A * B * C;
            }
        }
    }

    // Step 2.2.
    for (int i=0; i<alpha_index_times_count; i++)
    {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];
        mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
    }

    // energy_components
    atomicAdd(&energy_components[alpha_scalar_moments+type_central], 1.0);
    for (int k=0; k<alpha_scalar_moments; k++)
        atomicAdd(&energy_components[k], mom_vals[alpha_moment_mapping[k]] / q_scaler[k]);
    
    // Step 2.2. force_components && virial_components
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                 + std::pow(neigh_vec[1], 2)
                                 + std::pow(neigh_vec[2], 2) );
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals,
            rq_chebyshev_ders2r,
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
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * neigh_vec[aa];
        }

        for (int aa=0; aa<3; aa++) {
            for (int p=0; p<MAX_ALPHA_MOMENTS_COUNT; p++)
                moms_der2xyz[p] = 0.0;
            for (int i=0; i<alpha_index_basic_count; i++) {
                int mu = alpha_index_basic[i][0];
                int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
                CoordType powk = 1.0 / auto_dist_powers_[k];
                CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
                CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
                CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
                CoordType mult0 = pow0 * pow1 * pow2;

                for (int xi=0; xi<chebyshev_size; xi++) {
                    int idx = (type_central*ntypes+type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                    CoordType A = rq_chebyshev_vals[xi];
                    CoordType B = mult0;
                    CoordType C = powk;
                    CoordType A_der = rq_chebyshev_ders2r[xi] * neigh_vec[aa] * distance_ij_inv;
                    CoordType B_ders[3] = {0.0};
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
                    CoordType C_der = -k * powk * distance_ij_inv * distance_ij_inv * neigh_vec[aa];

                    moms_der2xyz[i] += coeffs[idx]
                                       * (A_der*B*C + A*B_ders[aa]*C + A*B*C_der);
                }
            }

            for (int i=0; i<alpha_index_times_count; i++) {
                CoordType val0 = mom_vals[alpha_index_times[i][0]];
                CoordType val1 = mom_vals[alpha_index_times[i][1]];
                CoordType val2 = alpha_index_times[i][2];
                
                moms_der2xyz[alpha_index_times[i][3]] += val2 * (moms_der2xyz[alpha_index_times[i][0]] * val1
                                                                 + val0 * moms_der2xyz[alpha_index_times[i][1]]);
            }

            // force_components
            for (int k=0; k<alpha_scalar_moments; k++) {
                atomicAdd(&force_components[center_idx*3*num_parameters + aa*num_parameters + k], 
                          moms_der2xyz[alpha_moment_mapping[k]] / q_scaler[k]);
                atomicAdd(&force_components[neigh_idx*3*num_parameters + aa*num_parameters + k],
                          -moms_der2xyz[alpha_moment_mapping[k]] / q_scaler[k]);
            }

            // virial_components
            for (int bb=0; bb<3; bb++) {
                for (int k=0; k<alpha_scalar_moments; k++) {
                    atomicAdd(&virial_components[(aa*3+bb)*num_parameters + k],
                              -(moms_der2xyz[alpha_moment_mapping[k]] / q_scaler[k]) * neigh_vec[bb]);
                }
            }
        }
    }
}


template <typename CoordType>
__global__
void find_efv_components_kernel(
    CoordType *benergy_components,
    CoordType *bforce_components,
    CoordType *bvirial_components,
    int chebyshev_size,
    CoordType *coeffs,
    CoordType *linear_coeffs,
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
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_scaler)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    int ii = nx % natoms_pad;
    if (istruct >= batch_size)
        return;
    int inum = binum[istruct];
    
    int num_parameters = alpha_scalar_moments + ntypes;

    if (ii<inum) {
        CoordType *energy_components = &benergy_components[istruct*num_parameters];
        CoordType *force_components = &bforce_components[istruct*natoms_pad*3*num_parameters];
        CoordType *virial_components = &bvirial_components[istruct*3*3*num_parameters];
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = &brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        int *types = &btypes[istruct*natoms_pad];

        accumulate_efv_components_atom(
            energy_components,
            force_components,
            virial_components,
            chebyshev_size,
            coeffs,
            linear_coeffs,
            type_bias,
            alpha_moments_count,
            alpha_index_basic_count,
            alpha_index_basic,
            alpha_index_times_count,
            alpha_index_times,
            alpha_scalar_moments,
            alpha_moment_mapping,
            nmus,
            silist,
            snumneigh,
            sfirstneigh,
            srcs,
            types,
            ntypes,
            umax_num_neigh_atoms,
            nghost,
            rmax,
            rmin,
            q_scaler);
    }
}


template <typename CoordType>
__host__
void find_efv_components_launcher(
    CoordType *h_benergy_components,
    CoordType *h_bforce_components,
    CoordType *h_bvirial_components,
    int chebyshev_size,
    CoordType *h_coeffs,
    CoordType *h_linear_coeffs,
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
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_q_scaler)
{
    int block_size_x = 128;
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);
    int num_parameters = alpha_scalar_moments + ntypes;

    CoordType *d_benergy_components;
    CoordType *d_bforce_components;
    CoordType *d_bvirial_components;
    CHECK_CUDA_API( cudaMalloc((void**)&d_benergy_components, sizeof(CoordType)*batch_size*num_parameters) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_components, sizeof(CoordType)*batch_size*natoms_pad*3*num_parameters) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_components, sizeof(CoordType)*batch_size*3*3*num_parameters) );
    //
    CHECK_CUDA_API( cudaMemset(d_benergy_components, 0, sizeof(CoordType)*batch_size*num_parameters) );
    CHECK_CUDA_API( cudaMemset(d_bforce_components, 0, sizeof(CoordType)*batch_size*natoms_pad*3*num_parameters) );
    CHECK_CUDA_API( cudaMemset(d_bvirial_components, 0, sizeof(CoordType)*batch_size*3*3*num_parameters) );

    CoordType *d_coeffs;
    CoordType *d_linear_coeffs;
    CoordType *d_type_bias;
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType)*ntypes*ntypes*nmus*chebyshev_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_linear_coeffs, sizeof(CoordType)*alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType)*ntypes) );
    // 
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*ntypes*ntypes*nmus*chebyshev_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_linear_coeffs, h_linear_coeffs, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType)*ntypes, cudaMemcpyHostToDevice) );

    int (*d_alpha_index_basic)[4];
    int (*d_alpha_index_times)[4];
    int *d_alpha_moment_mapping;
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int)*alpha_index_times_count*4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments) );
    //
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int)*alpha_index_times_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments, cudaMemcpyHostToDevice) );


    int *d_binum;
    int *d_bilist;
    int *d_bnumneigh;
    int *d_bfirstneigh;
    CoordType (*d_brcs)[3];
    int *d_btypes;
    CoordType *d_q_scaler;
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_q_scaler, sizeof(CoordType)*alpha_scalar_moments) );
    //
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_q_scaler, h_q_scaler, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    
    // Step 2.


    // Step 3.
    auto t1 = std::chrono::high_resolution_clock::now();
    find_efv_components_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_benergy_components,
        d_bforce_components,
        d_bvirial_components,
        chebyshev_size,
        d_coeffs,
        d_linear_coeffs,
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
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        d_q_scaler);
    CHECK_CUDA_KERNEL;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_efv_components_kernel() cost time: " << duration.count() << " us." << std::endl;

    CHECK_CUDA_API( cudaMemcpy(h_benergy_components, d_benergy_components, sizeof(CoordType)*batch_size*num_parameters, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bforce_components, d_bforce_components, sizeof(CoordType)*batch_size*natoms_pad*3*num_parameters, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bvirial_components, d_bvirial_components, sizeof(CoordType)*batch_size*3*3*num_parameters, cudaMemcpyDeviceToHost) );

    // Step . Free
    CHECK_CUDA_API( cudaFree(d_benergy_components) );
    CHECK_CUDA_API( cudaFree(d_bforce_components) );
    CHECK_CUDA_API( cudaFree(d_bvirial_components) );

    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_linear_coeffs) );
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
    CHECK_CUDA_API( cudaFree(d_q_scaler) );
}


template <typename CoordType>
__global__
void find_lin_matrix_lin_vector_kernel(
    CoordType *lin_matrix,
    CoordType *lin_vector,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_residual,
    CoordType (*bforce_residual)[3],
    CoordType *bvirial_residual,
    CoordType *benergy_components,
    CoordType *bforce_components,
    CoordType *bvirial_components,
    int alpha_scalar_moments,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int ntypes)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_parameters = alpha_scalar_moments + ntypes;
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int ii = nx % natoms_pad;

    if (ii<binum[istruct]) {
        // 1. force term
        int center_idx = bilist[istruct*natoms_pad + ii];
        for (int aa=0; aa<3; aa++) {
            for (int k1=0; k1<num_parameters; k1++) {
                for (int k2=k1; k2<num_parameters; k2++) {
                    CoordType tmp = f_weight
                                    * bforce_components[(istruct*natoms_pad*3+center_idx*3+aa)*num_parameters + k1]
                                    * bforce_components[(istruct*natoms_pad*3+center_idx*3+aa)*num_parameters + k2];
                    atomicAdd(&lin_matrix[k1*num_parameters + k2], tmp);
                }
                CoordType tmp = f_weight
                                * bforce_residual[istruct*natoms_pad + center_idx][aa]
                                * bforce_components[(istruct*natoms_pad*3+center_idx*3+aa)*num_parameters + k1];
                atomicAdd(&lin_vector[k1], tmp);
            }
        }
    }

    if (ii == 0) {
        // energy term
        for (int k1=0; k1<num_parameters; k1++) {
            for (int k2=k1; k2<num_parameters; k2++) {
                CoordType tmp = e_weight
                                * benergy_components[istruct*num_parameters + k1]
                                * benergy_components[istruct*num_parameters + k2];
                atomicAdd(&lin_matrix[k1*num_parameters + k2], tmp);
            }
            CoordType tmp = e_weight
                            * betot_residual[istruct]
                            * benergy_components[istruct*num_parameters + k1];
            atomicAdd(&lin_vector[k1], tmp);
        }

        // virial term
        for (int alpha=0; alpha<3; alpha++) {
            for (int beta=0; beta<3; beta++) {
                for (int k1=0; k1<num_parameters; k1++) {
                    for (int k2=k1; k2<num_parameters; k2++) {
                        CoordType tmp = v_weight
                                        * bvirial_components[(istruct*3*3+alpha*3+beta)*num_parameters + k1]
                                        * bvirial_components[(istruct*3*3+alpha*3+beta)*num_parameters + k2];
                        atomicAdd(&lin_matrix[k1*num_parameters + k2], tmp);
                    }
                    CoordType tmp = v_weight
                                    * bvirial_residual[istruct*3*3 + alpha*3 + beta]
                                    * bvirial_components[(istruct*3*3+alpha*3+beta)*num_parameters + k1];
                    atomicAdd(&lin_vector[k1], tmp);
                }
            }
        }
    }
}


template <typename CoordType>
__global__
void mirror_lin_matrix(
    CoordType *lin_matrix,
    int num_parameters)
{
    int k1 = blockIdx.x * blockDim.x + threadIdx.x;
    int k2 = blockIdx.y * blockDim.y + threadIdx.y;

    if ((k1<num_parameters) && (k2<num_parameters) && (k1>k2)) {
        lin_matrix[k1 * num_parameters + k2] = lin_matrix[k2 * num_parameters + k1];
    }
}


template <typename CoordType>
__host__
void find_lin_matrix_lin_vector_launcher(
    CoordType *h_lin_matrix,
    CoordType *h_lin_vector,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *h_betot_residual,
    CoordType (*h_bforce_residual)[3],
    CoordType *h_bvirial_residual,
    int chebyshev_size,
    CoordType *h_coeffs,
    CoordType *h_linear_coeffs,
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
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_q_scaler)
{
    int block_size_x = 128;
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);
    int num_parameters = alpha_scalar_moments + ntypes;

    CoordType *d_lin_matrix;
    CoordType *d_lin_vector;
    CHECK_CUDA_API( cudaMalloc((void**)&d_lin_matrix, sizeof(CoordType)*num_parameters*num_parameters) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_lin_vector, sizeof(CoordType)*num_parameters) );
    //
    CHECK_CUDA_API( cudaMemset(d_lin_matrix, 0, sizeof(CoordType)*num_parameters*num_parameters) );
    CHECK_CUDA_API( cudaMemset(d_lin_vector, 0, sizeof(CoordType)*num_parameters) );

    CoordType *d_betot_residual;
    CoordType (*d_bforce_residual)[3];
    CoordType *d_bvirial_residual;
    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_residual, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_residual, sizeof(CoordType)*batch_size*natoms_pad*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_residual, sizeof(CoordType)*batch_size*3*3) );
    // 
    CHECK_CUDA_API( cudaMemcpy(d_betot_residual, h_betot_residual, sizeof(CoordType)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bforce_residual, h_bforce_residual, sizeof(CoordType)*batch_size*natoms_pad*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bvirial_residual, h_bvirial_residual, sizeof(CoordType)*batch_size*3*3, cudaMemcpyHostToDevice) );

    CoordType *d_benergy_components;
    CoordType *d_bforce_components;
    CoordType *d_bvirial_components;
    CHECK_CUDA_API( cudaMalloc((void**)&d_benergy_components, sizeof(CoordType)*batch_size*num_parameters) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce_components, sizeof(CoordType)*batch_size*natoms_pad*3*num_parameters) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial_components, sizeof(CoordType)*batch_size*3*3*num_parameters) );
    //
    CHECK_CUDA_API( cudaMemset(d_benergy_components, 0, sizeof(CoordType)*batch_size*num_parameters) );
    CHECK_CUDA_API( cudaMemset(d_bforce_components, 0, sizeof(CoordType)*batch_size*natoms_pad*3*num_parameters) );
    CHECK_CUDA_API( cudaMemset(d_bvirial_components, 0, sizeof(CoordType)*batch_size*3*3*num_parameters) );

    CoordType *d_coeffs;
    CoordType *d_linear_coeffs;
    CoordType *d_type_bias;
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType)*ntypes*ntypes*nmus*chebyshev_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_linear_coeffs, sizeof(CoordType)*alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType)*ntypes) );
    // 
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*ntypes*ntypes*nmus*chebyshev_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_linear_coeffs, h_linear_coeffs, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType)*ntypes, cudaMemcpyHostToDevice) );

    int (*d_alpha_index_basic)[4];
    int (*d_alpha_index_times)[4];
    int *d_alpha_moment_mapping;
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int)*alpha_index_times_count*4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments) );
    //
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int)*alpha_index_times_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments, cudaMemcpyHostToDevice) );


    int *d_binum;
    int *d_bilist;
    int *d_bnumneigh;
    int *d_bfirstneigh;
    CoordType (*d_brcs)[3];
    int *d_btypes;
    CoordType *d_q_scaler;
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_q_scaler, sizeof(CoordType)*alpha_scalar_moments) );
    //
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_q_scaler, h_q_scaler, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    
    // Step 2.


    // Step 3.
    auto t1 = std::chrono::high_resolution_clock::now();
    find_efv_components_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_benergy_components,
        d_bforce_components,
        d_bvirial_components,
        chebyshev_size,
        d_coeffs,
        d_linear_coeffs,
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
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        d_q_scaler);
    CHECK_CUDA_KERNEL;
    find_lin_matrix_lin_vector_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_lin_matrix,
        d_lin_vector,
        e_weight,
        f_weight,
        v_weight,
        d_betot_residual, //
        d_bforce_residual,
        d_bvirial_residual,
        d_benergy_components,
        d_bforce_components,
        d_bvirial_components,
        alpha_scalar_moments,
        nmus,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        ntypes);
    CHECK_CUDA_KERNEL;

    int mirror_block_size_x = 16;
    int mirror_block_size_y = 16;
    dim3 mirror_block_size(mirror_block_size_x, mirror_block_size_y);
    int mirror_grid_size_x = (num_parameters - 1) / mirror_block_size_x + 1;
    int mirror_grid_size_y = (num_parameters - 1) / mirror_block_size_y + 1;
    dim3 mirror_grid_size(mirror_grid_size_x, mirror_grid_size_y);
    mirror_lin_matrix<CoordType> KERNEL_ARG2(mirror_grid_size, mirror_block_size) (d_lin_matrix, num_parameters);
    CHECK_CUDA_KERNEL;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_lin_matrix_lin_vector_kernel() cost time: " << duration.count() << " us." << std::endl;

    CHECK_CUDA_API( cudaMemcpy(h_lin_matrix, d_lin_matrix, sizeof(CoordType)*num_parameters*num_parameters, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_lin_vector, d_lin_vector, sizeof(CoordType)*num_parameters, cudaMemcpyDeviceToHost) );

    // Step . Free
    CHECK_CUDA_API( cudaFree(d_lin_matrix) );
    CHECK_CUDA_API( cudaFree(d_lin_vector) );

    CHECK_CUDA_API( cudaFree(d_betot_residual) );
    CHECK_CUDA_API( cudaFree(d_bforce_residual) );
    CHECK_CUDA_API( cudaFree(d_bvirial_residual) );

    CHECK_CUDA_API( cudaFree(d_benergy_components) );
    CHECK_CUDA_API( cudaFree(d_bforce_components) );
    CHECK_CUDA_API( cudaFree(d_bvirial_components) );

    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_linear_coeffs) );
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
    CHECK_CUDA_API( cudaFree(d_q_scaler) );
}




};  // namespace : mtpr
};  // namespace : ai2pot

#endif