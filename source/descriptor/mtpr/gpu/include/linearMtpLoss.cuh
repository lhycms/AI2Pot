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

#ifndef AI2POT_LINEAR_MTP_LOSS_CUH
#define AI2POT_LINEAR_MTP_LOSS_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "./mtpBasis.cuh"


namespace ai2pot {
namespace mtpr {

template <typename CoordType>
static __global__
void find_loss_kernel(CoordType &loss,
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
void find_loss_launcher(CoordType &h_loss,
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
void find_ef_loss_launcher(CoordType &h_loss,
                           int inum,
                           int *h_ilist,
                           CoordType e_weight,
                           CoordType f_weight,
                           CoordType etot_ml,
                           CoordType etot_dft,
                           CoordType (*h_force_ml)[3],
                           CoordType (*h_force_dft)[3]);


template <typename CoordType>
static __device__
void find_loss_backward_atom(CoordType *loss_der2coeffs,
                             CoordType *loss_der2linear_coeffs,
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
                             CoordType rmin,
                             CoordType zbl_rmax,
                             CoordType zbl_rmin,
                             CoordType *zbl_cks,
                             CoordType *zbl_dks);


template <typename CoordType>
static __global__
void find_loss_backward_kernel(CoordType *loss_der2coeffs,
                                CoordType *loss_der2linear_coeffs,
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
                                int inum,
                                int *ilist,
                                int *numneigh,
                                int *firstneigh,
                                CoordType (*rcs)[3],
                                int *types,
                                int ntypes,
                                int *type_map,
                                int umax_num_neigh_atoms,
                                int nghost,
                                CoordType rmax,
                                CoordType rmin,
                                CoordType zbl_rmax,
                                CoordType zbl_rmin,
                                CoordType *zbl_cks,
                                CoordType *zbl_dks);


template <typename CoordType>
static __host__
void find_loss_backward_launcher(CoordType *h_loss_der2coeffs,
                                 CoordType *h_loss_der2linear_coeffs,
                                 CoordType *h_loss_der2type_bias,
                                 CoordType e_weight,
                                 CoordType f_weight,
                                 CoordType v_weight,
                                 CoordType etot_ml,
                                 CoordType etot_dft,
                                 CoordType (*h_force_ml)[3],
                                 CoordType (*h_force_dft)[3],
                                 CoordType *h_virial_ml,
                                 CoordType *h_virial_dft,
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
                                 int inum,
                                 int *h_ilist,
                                 int *h_numneigh,
                                 int *h_firstneigh,
                                 CoordType (*h_rcs)[3],
                                 int *h_types,
                                 int ntypes,
                                 int *h_type_map,
                                 int umax_num_neigh_atoms,
                                 int nghost,
                                 CoordType rmax,
                                 CoordType rmin,
                                 CoordType zbl_rmax,
                                 CoordType zbl_rmin,
                                 CoordType *h_zbl_cks,
                                 CoordType *h_zbl_dks);








template <typename CoordType>
__global__ 
void find_loss_kernel(CoordType &loss,
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
        atomicAdd(&loss, f_loss);
    }


    if (nx == 0) {
        CoordType v_loss = 0.0;
        for (int aa=0; aa<3; aa++)
            for (int bb=0; bb<3; bb++)
                v_loss += std::pow(virial_ml[aa*3+bb] - virial_dft[aa*3+bb], 2);
        v_loss = v_weight / (9*inum) * v_loss;
        atomicAdd(&loss, v_loss);

        CoordType e_loss = 0.0;
        e_loss = e_weight / inum * std::pow(etot_ml - etot_dft, 2);
        atomicAdd(&loss, e_loss);
    }
}


template <typename CoordType>
void find_loss_launcher(CoordType &h_loss,
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
    find_loss_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (*d_loss_ptr,
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

    CHECK_CUDA_API( cudaMemcpy(&h_loss, d_loss_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_loss_ptr) );
    CHECK_CUDA_API( cudaFree(d_ilist) );
    CHECK_CUDA_API( cudaFree(d_force_ml) );
    CHECK_CUDA_API( cudaFree(d_force_dft) );
    CHECK_CUDA_API( cudaFree(d_virial_ml) );
    CHECK_CUDA_API( cudaFree(d_virial_dft) );
}


template <typename CoordType>
__global__
void find_ef_loss_kernel(CoordType &loss,
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
        atomicAdd(&loss, f_loss);
    }

    if (nx == 0) {
        CoordType e_loss = 0;
        e_loss = e_weight / inum * std::pow(etot_ml - etot_dft, 2);
        atomicAdd(&loss, e_loss);
    }
}


template <typename CoordType>
void find_ef_loss_launcher(CoordType &h_loss,
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
    find_ef_loss_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (*d_loss_ptr,
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

    CHECK_CUDA_API( cudaMemcpy(&h_loss, d_loss_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_loss_ptr) );
    CHECK_CUDA_API( cudaFree(d_ilist) );
    CHECK_CUDA_API( cudaFree(d_force_ml) );
    CHECK_CUDA_API( cudaFree(d_force_dft) );
}


template <typename CoordType>
__device__
void find_loss_backward_atom(CoordType *loss_der2coeffs,
                             CoordType *loss_der2linear_coeffs,
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
                             CoordType rmin,
                             CoordType zbl_rmax,
                             CoordType zbl_rmin,
                             CoordType *zbl_cks,
                             CoordType *zbl_dks)
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

        auto_dist_powers_[0] = distance_ij;
        for (int aa=0; aa<3; aa++)
            auto_coords_powers_[0][aa] = NeighbVect[aa];
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
                    CoordType tmp_deriv = coeffs[idx] * (A_ders[aa] * B * C
                                                         + A * B_ders[aa] * C
                                                         + A * B * C_ders[aa]);
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

    for (int i=alpha_index_times_count-1; i>=0; i--) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];
        mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
        dloss_combination[alpha_index_times[i][3]] += (dloss_combination[alpha_index_times[i][0]] * val2 * val1
                                                       + dloss_combination[alpha_index_times[i][1]] * val2 * val0);
        
    }

    // 2.1. Linear Energy derivative w.r.t. mom_vals
    for (int i=0; i<alpha_scalar_moments; i++)
        e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i];

    // 2.2. Pass to basic moments
    for (int i=alpha_index_times_count-1; i>=0; i--) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];

        e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val1
                                                   * val2;
        e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val0
                                                   * val2;
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
        auto_dist_powers_[0] = distance_ij;
        for (int aa=0; aa<3; aa++)
            auto_coords_powers_[0][aa] = NeighbVect[aa];
        for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++) {
            int mu = alpha_index_basic[i][0];
            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            CoordType powk = 1 / auto_dist_powers_[k];
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

                CoordType tmpe_loss_der2coeff = 2*e_weight*(etot_ml - etot_dft)
                                                * e_site_der2mom[i]
                                                * A * B * C;
    
                CoordType tmpf_loss_der2coeff = 0;
                CoordType tmp_prefix = 0;
                for (int aa=0; aa<3; aa++) {
                    CoordType tmp_deriv = (A_ders[aa] * B * C
                                           + A * B_ders[aa] * C
                                           + A * B * C_ders[aa]);
                    tmp_prefix += 2*f_weight/(3*inum)
                                  * (force_ml[center_idx][aa] - force_dft[center_idx][aa]);
                    tmp_prefix -= 2*f_weight/(3*inum)
                                  * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa]);
                    for (int bb=0; bb<3; bb++) {
                        tmp_prefix -= -2*v_weight/(9*inum)
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


    // 2.4. loss derivative w.r.t. linear_coeffs
    for (int i=0; i<alpha_scalar_moments; i++)
        atomicAdd(&loss_der2linear_coeffs[i],
                  2*e_weight/inum
                  * (etot_ml - etot_dft)
                  * mom_vals[alpha_moment_mapping[i]]
                  + dloss_combination[alpha_moment_mapping[i]]);

    // 2.5. loss derivative w.r.t. type_bias
    atomicAdd(&loss_der2type_bias[type_central],
              2*e_weight/inum*(etot_ml - etot_dft));
}



template <typename CoordType>
__global__
void find_loss_backward_kernel(CoordType *loss_der2coeffs,
                                CoordType *loss_der2linear_coeffs,
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
                                int inum,
                                int *ilist,
                                int *numneigh,
                                int *firstneigh,
                                CoordType (*rcs)[3],
                                int *types,
                                int ntypes,
                                int *type_map,
                                int umax_num_neigh_atoms,
                                int nghost,
                                CoordType rmax,
                                CoordType rmin,
                                CoordType zbl_rmax,
                                CoordType zbl_rmin,
                                CoordType *zbl_cks,
                                CoordType *zbl_dks)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = nx;

    if (ii < inum) {
        int silist = ilist[ii];
        int snumneigh = numneigh[ii];
        int *sfirstneigh = &firstneigh[ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = (CoordType (*)[3])(&rcs[ii*umax_num_neigh_atoms][0]);
        find_loss_backward_atom<CoordType>(loss_der2coeffs,
                                           loss_der2linear_coeffs,
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
                                           rmin,
                                           zbl_rmax,
                                           zbl_rmin,
                                           zbl_cks,
                                           zbl_dks);
    }
}


template <typename CoordType>
__host__
void find_loss_backward_launcher(CoordType *h_loss_der2coeffs,
                                 CoordType *h_loss_der2linear_coeffs,
                                 CoordType *h_loss_der2type_bias,
                                 CoordType e_weight,
                                 CoordType f_weight,
                                 CoordType v_weight,
                                 CoordType etot_ml,
                                 CoordType etot_dft,
                                 CoordType (*h_force_ml)[3],
                                 CoordType (*h_force_dft)[3],
                                 CoordType *h_virial_ml,
                                 CoordType *h_virial_dft,
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
                                 int inum,
                                 int *h_ilist,
                                 int *h_numneigh,
                                 int *h_firstneigh,
                                 CoordType (*h_rcs)[3],
                                 int *h_types,
                                 int ntypes,
                                 int *h_type_map,
                                 int umax_num_neigh_atoms,
                                 int nghost,
                                 CoordType rmax,
                                 CoordType rmin,
                                 CoordType zbl_rmax,
                                 CoordType zbl_rmin,
                                 CoordType *h_zbl_cks,
                                 CoordType *h_zbl_dks)
{
    int block_size_x = 64;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    CoordType *d_loss_der2coeffs;
    CoordType *d_loss_der2linear_coeffs;
    CoordType *d_loss_der2type_bias;
    CoordType (*d_force_ml)[3];
    CoordType (*d_force_dft)[3];
    CoordType *d_virial_ml;
    CoordType *d_virial_dft;
    CoordType *d_coeffs;
    CoordType *d_linear_coeffs;
    CoordType *d_type_bias;
    int (*d_alpha_index_basic)[4];
    int (*d_alpha_index_times)[4];
    int *d_alpha_moment_mapping;
    int *d_ilist;
    int *d_numneigh;
    int *d_firstneigh;
    CoordType (*d_rcs)[3];
    int *d_types;
    int *d_type_map;
    CoordType *d_zbl_cks;
    CoordType *d_zbl_dks;

    CHECK_CUDA_API( cudaMalloc((void**)&d_loss_der2coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_loss_der2linear_coeffs, sizeof(CoordType) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_loss_der2type_bias, sizeof(CoordType) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force_ml, sizeof(CoordType) * inum * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force_dft, sizeof(CoordType) * inum * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_virial_ml, sizeof(CoordType) * 9) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_virial_dft, sizeof(CoordType) * 9) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_linear_coeffs, sizeof(CoordType) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_numneigh, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_firstneigh, sizeof(int) * inum * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_rcs, sizeof(CoordType) * inum * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_types, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_zbl_cks, sizeof(CoordType) * ntypes * ntypes * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_zbl_dks, sizeof(CoordType) * ntypes * ntypes * 4) );

    CHECK_CUDA_API( cudaMemcpy(d_force_ml, h_force_ml, sizeof(CoordType)*inum*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_force_dft, h_force_dft, sizeof(CoordType)*inum*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_virial_ml, h_virial_ml, sizeof(CoordType)*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_virial_dft, h_virial_dft, sizeof(CoordType)*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*num_coeffs, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_linear_coeffs, h_linear_coeffs, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int)*alpha_index_times_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_ilist, h_ilist, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_numneigh, h_numneigh, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_firstneigh, h_firstneigh, sizeof(int)*inum*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_rcs, h_rcs, sizeof(CoordType)*inum*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_types, h_types, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_zbl_cks, h_zbl_cks, sizeof(CoordType)*ntypes*ntypes*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_zbl_dks, h_zbl_dks, sizeof(CoordType)*ntypes*ntypes*4, cudaMemcpyHostToDevice) );


    find_loss_backward_kernel KERNEL_ARG2(grid_size, block_size) (d_loss_der2coeffs,
                                                                  d_loss_der2linear_coeffs,
                                                                  d_loss_der2type_bias,
                                                                  e_weight,
                                                                  f_weight,
                                                                  v_weight,
                                                                  etot_ml,
                                                                  etot_dft,
                                                                  d_force_ml,
                                                                  d_force_dft,
                                                                  d_virial_ml,
                                                                  d_virial_dft,
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
                                                                  inum,
                                                                  d_ilist,
                                                                  d_numneigh,
                                                                  d_firstneigh,
                                                                  d_rcs,
                                                                  d_types,
                                                                  ntypes,
                                                                  d_type_map,
                                                                  umax_num_neigh_atoms,
                                                                  nghost,
                                                                  rmax,
                                                                  rmin,
                                                                  zbl_rmax,
                                                                  zbl_rmin,
                                                                  d_zbl_cks,
                                                                  d_zbl_dks);
    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );


    CHECK_CUDA_API( cudaMemcpy(h_loss_der2coeffs, d_loss_der2coeffs, sizeof(CoordType)*num_coeffs, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_loss_der2linear_coeffs, d_loss_der2linear_coeffs, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_loss_der2type_bias, d_loss_der2type_bias, sizeof(CoordType)*ntypes, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_loss_der2coeffs) );
    CHECK_CUDA_API( cudaFree(d_loss_der2linear_coeffs) );
    CHECK_CUDA_API( cudaFree(d_loss_der2type_bias) );
    CHECK_CUDA_API( cudaFree(d_force_ml) );
    CHECK_CUDA_API( cudaFree(d_force_dft) );
    CHECK_CUDA_API( cudaFree(d_virial_ml) );
    CHECK_CUDA_API( cudaFree(d_virial_dft) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_linear_coeffs) );
    CHECK_CUDA_API( cudaFree(d_type_bias) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_basic) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_times) );
    CHECK_CUDA_API( cudaFree(d_ilist) );
    CHECK_CUDA_API( cudaFree(d_numneigh) );
    CHECK_CUDA_API( cudaFree(d_firstneigh) );
    CHECK_CUDA_API( cudaFree(d_rcs) );
    CHECK_CUDA_API( cudaFree(d_types) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
    CHECK_CUDA_API( cudaFree(d_zbl_cks) );
    CHECK_CUDA_API( cudaFree(d_zbl_dks) );



}



};  // namespace: mtpr
};  // namespace : ai2pot

#endif
