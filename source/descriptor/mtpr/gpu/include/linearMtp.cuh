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

#ifndef AI2POT_LINEAR_MTP_CUH
#define AI2POT_LINEAR_MTP_CUH
#include <math.h>
#include <chrono>
#include <iostream>

#include "./mtpBasis.cuh"


namespace ai2pot {
namespace mtpr {


template <typename CoordType>
static __host__ __device__
void find_efv_atom(CoordType *etot_ptr,
                   CoordType (*force)[3],
                   CoordType *virial,
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
                   int *type_map,
                   int umax_num_neigh_atoms,
                   int nghost,
                   CoordType rmax,
                   CoordType rmin,
                   CoordType *s_local_virial);


template <typename CoordType>
static __global__
void find_efv_kernel(CoordType *etot_ptr,
                     CoordType (*force)[3],
                     CoordType *virial,
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
                     CoordType rmin);


template <typename CoordType>
static __host__
void find_efv_launcher(CoordType *h_etot_ptr,
                       CoordType (*h_force)[3],
                       CoordType *h_virial,
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
                       CoordType rmin);
    

template <typename CoordType>
static __host__ __device__
void find_ef_atom(CoordType *etot_ptr,
                  CoordType (*force)[3],
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
                  int *type_map,
                  int umax_num_neigh_atoms,
                  int nghost,
                  CoordType rmax,
                  CoordType rmin);


template <typename CoordType>
static __global__ 
void find_ef_kernel(CoordType *etot_ptr,
                    CoordType (*force)[3],
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
                    CoordType rmin);


template <typename CoordType>
static __host__
void find_ef_launcher(CoordType *h_etot_ptr,
                      CoordType (*h_force)[3],
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
                      CoordType rmin);





template <typename CoordType>
__host__ __device__ 
void find_efv_atom(CoordType *etot_ptr,
                   CoordType (*force)[3],
                   CoordType *virial,
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
                   int *type_map,
                   int umax_num_neigh_atoms,
                   int nghost,
                   CoordType rmax,
                   CoordType rmin,
                   CoordType *s_local_virial)
{   
    // Step 1. Init temp array
    CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.};
    CoordType e_site_der2mom[MAX_ALPHA_MOMENTS_COUNT] = {0.};

    CoordType auto_dist_powers_[MAX_ALPHA_INDEX_BASIC] = {0.};
    CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3] = {{0.}};

    int center_idx;
    int type_central;
    int Zi;
    int neigh_idx;
    int type_outer;
    int Zj;
    //int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    
    center_idx = silist;
    type_central = types[center_idx];
    Zi = type_map[type_central];

    
    // Step 2. Calculate mom_vals
    // Step 2.1.
    for (int jj=0; jj<snumneigh; jj++) {
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
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++)
        {
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

    // Step 3. Calculate EFV for atom
    // Step 3.1. Energy
    CoordType e_site = type_bias[type_central];
    for (int i=0; i<alpha_scalar_moments; i++)
        e_site += linear_coeffs[i] * mom_vals[alpha_moment_mapping[i]];
    atomicAdd(etot_ptr, e_site);

    for (int i=0; i<alpha_scalar_moments; i++)
        e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i];

    for (int i=alpha_index_times_count-1; i>=0; i--) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];

        e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val0;
        e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val1;
    }

    // Step 3.2. Force and Virial.
    for (int jj=0; jj<snumneigh; jj++) {
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
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++) 
        {
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
                CoordType A_ders[3] = {0., 0., 0.};
                CoordType B_ders[3] = {0., 0., 0.};
                CoordType C_ders[3] = {0., 0., 0.};
                A_ders[0] = rq_chebyshev_ders2r[xi] * NeighbVect[0] * distance_ij_inv;
                A_ders[1] = rq_chebyshev_ders2r[xi] * NeighbVect[1] * distance_ij_inv;
                A_ders[2] = rq_chebyshev_ders2r[xi] * NeighbVect[2] * distance_ij_inv;
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

                for (int aa=0; aa<3; aa++) {
                    CoordType mom_der2xyz = coeffs[idx] 
                                            * (A_ders[aa]*B*C + A*B_ders[aa]*C + A*B*C_ders[aa]);
                    CoordType e_site_ders_ija = e_site_der2mom[i] * mom_der2xyz;
                    atomicAdd(&force[center_idx][aa], e_site_ders_ija);
                    atomicAdd(&force[neigh_idx][aa], -e_site_ders_ija);

                    for (int bb=0; bb<3; bb++)
                        //atomicAdd(&virial[aa*3 + bb], -e_site_ders_ija * NeighbVect[bb]);
                        s_local_virial[aa*3 + bb] -= e_site_ders_ija * NeighbVect[bb];
                 }
            }
        }
    }
}


template <typename CoordType>
static __global__
void find_efv_kernel(CoordType *etot_ptr,
                    CoordType (*force)[3],
                    CoordType *virial,
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
                    CoordType rmin)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = nx;

    __shared__ CoordType s_local_virial[128][9];
    int tid = threadIdx.x;
    for (int ii=0; ii<9; ii++)
        s_local_virial[tid][ii] = 0.0;
    //__syncthreads();

    if (ii < inum) {
        int silist = ilist[ii];
        int snumneigh = numneigh[ii];
        int *sfirstneigh = &firstneigh[ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = (CoordType (*)[3])(&rcs[ii*umax_num_neigh_atoms][0]);
        find_efv_atom<CoordType>(etot_ptr,
                                 force,
                                 virial,
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
                                 type_map,
                                 umax_num_neigh_atoms,
                                 nghost,
                                 rmax,
                                 rmin,
                                 s_local_virial[tid]);
    }
    __syncthreads();

    if (tid == 0) {
        for (int t=0; t<blockDim.x; t++)
            for (int i=0; i<9; i++)
                atomicAdd(&virial[i], s_local_virial[t][i]);
    }
}


template <typename CoordType>
__host__
void find_efv_launcher(CoordType *h_etot_ptr,
                       CoordType (*h_force)[3],
                       CoordType *h_virial,
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
                       CoordType rmin)
{
    int block_size_x = 128;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_etot_ptr;
    CoordType (*d_force)[3];
    CoordType *d_virial;
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

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    CHECK_CUDA_API( cudaMalloc((void**)&d_etot_ptr, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force, sizeof(CoordType) * (inum + nghost) * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_virial, sizeof(CoordType) * 9) );
    CHECK_CUDA_API( cudaMemset(d_etot_ptr, 0.0, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMemset(d_force, 0.0, sizeof(CoordType) * (inum + nghost) * 3) );
    CHECK_CUDA_API( cudaMemset(d_virial, 0.0, sizeof(CoordType) * 9) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_linear_coeffs, sizeof(CoordType) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_numneigh, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_firstneigh, sizeof(int) * inum * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_rcs, sizeof(CoordType) * inum * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_types, sizeof(int) * (inum + nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );

    //CHECK_CUDA_API( cudaMemcpy(d_etot_ptr, &h_etot, sizeof(CoordType), cudaMemcpyHostToDevice) );
    //CHECK_CUDA_API( cudaMemcpy(d_force, h_force, sizeof(CoordType)*(inum+nghost)*3, cudaMemcpyHostToDevice) );
    //CHECK_CUDA_API( cudaMemcpy(d_virial, h_virial, sizeof(CoordType)*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*ntypes*ntypes*nmus*chebyshev_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_linear_coeffs, h_linear_coeffs, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int)*alpha_index_times_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_ilist, h_ilist, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_numneigh, h_numneigh, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_firstneigh, h_firstneigh, sizeof(int)*inum*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_rcs, h_rcs, sizeof(CoordType)*inum*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_types, h_types, sizeof(int)*(inum+nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );

    // Call global function
    auto t1 = std::chrono::high_resolution_clock::now();
    find_efv_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_etot_ptr,
        d_force,
        d_virial,
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
        rmin);
    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_efv_kernel() cost time: " << duration.count() << " us.\n";

    CHECK_CUDA_API( cudaMemcpy(h_etot_ptr, d_etot_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_force, d_force, sizeof(CoordType)*(inum+nghost)*3, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_virial, d_virial, sizeof(CoordType)*9, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_etot_ptr) );
    CHECK_CUDA_API( cudaFree(d_force) );
    CHECK_CUDA_API( cudaFree(d_virial) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_linear_coeffs) );
    CHECK_CUDA_API( cudaFree(d_type_bias) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_basic) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_times) );
    CHECK_CUDA_API( cudaFree(d_alpha_moment_mapping) );
    CHECK_CUDA_API( cudaFree(d_ilist) );
    CHECK_CUDA_API( cudaFree(d_numneigh) );
    CHECK_CUDA_API( cudaFree(d_firstneigh) );
    CHECK_CUDA_API( cudaFree(d_rcs) );
    CHECK_CUDA_API( cudaFree(d_types) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
}


template <typename CoordType>
__host__ __device__
void find_ef_atom(CoordType *etot_ptr,
                  CoordType (*force)[3],
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
                  int *type_map,
                  int umax_num_neigh_atoms,
                  int nghost,
                  CoordType rmax,
                  CoordType rmin)
{
    // 1. Init temp array
    CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.0};
    CoordType e_site_der2mom[MAX_ALPHA_MOMENTS_COUNT] = {0.0};

    int center_idx;
    int type_central;
    int Zi;
    int neigh_idx;
    int type_outer;
    int Zj;
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    CoordType auto_dists_powers_[MAX_ALPHA_INDEX_BASIC] = {0.0};
    CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3] = {{0.0}};

    center_idx = silist;
    type_central = types[center_idx];
    Zi = type_map[type_central];

    // Step 2. Calculate mom_vals
    // Step 2.1.
    for (int jj=0; jj<snumneigh; jj++) {
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
        CoordType rq_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                     rq_chebyshev_ders2r,
                                     chebyshev_size,
                                     rmax,
                                     rmin,
                                     distance_ij);
        
        auto_dists_powers_[0] = 1.0;
        for (int aa=0; aa<3; aa++)
            auto_coords_powers_[0][aa] = 1.0;
        for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
            auto_dists_powers_[k] = auto_dists_powers_[k-1] * distance_ij;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++)
        {
            int mu = alpha_index_basic[i][0];
            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            CoordType powk = 1.0 / auto_dists_powers_[k];
            CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
            CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
            CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
            CoordType mult0 = pow0 * pow1 * pow2;

            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes + type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
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

    // Step 3. Calculate EFV for atom
    // Step 3.1. Energy
    CoordType e_site = type_bias[type_central];
    for (int i=0; i<alpha_scalar_moments; i++)
        e_site += linear_coeffs[i] * mom_vals[alpha_moment_mapping[i]];
    atomicAdd(etot_ptr, e_site);

    for (int i=0; i<alpha_scalar_moments; i++)
        e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i];

    for (int i=alpha_index_times_count-1; i>=0; i--) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];

        e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val1;
        e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val0;
    }
    
    // Step 3.2. Force and Virial
    for (int jj=0; jj<snumneigh; jj++) {
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
        CoordType rq_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                     rq_chebyshev_ders2r,
                                     chebyshev_size,
                                     rmax,
                                     rmin,
                                     distance_ij);
        auto_dists_powers_[0] = 1.0;
        for (int aa=0; aa<3; aa++)
            auto_coords_powers_[0][aa] = 1.0;
        for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
            auto_dists_powers_[k] = auto_dists_powers_[k-1] * distance_ij;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++)
        {
            int mu = alpha_index_basic[i][0];
            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            CoordType powk = 1.0 / auto_dists_powers_[k];
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
                A_ders[0] = rq_chebyshev_ders2r[xi] * NeighbVect[0] * distance_ij_inv;
                A_ders[1] = rq_chebyshev_ders2r[xi] * NeighbVect[1] * distance_ij_inv;
                A_ders[2] = rq_chebyshev_ders2r[xi] * NeighbVect[2] * distance_ij_inv;
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

                for (int aa=0; aa<3; aa++) {
                    CoordType mom_der2xyz = coeffs[idx]
                                            * (A_ders[aa]*B*C + A*B_ders[aa]*C + A*B*C_ders[aa]);
                    CoordType e_site_ders_ija = e_site_der2mom[i] * mom_der2xyz;
                    atomicAdd(&force[center_idx][aa], e_site_ders_ija);
                    atomicAdd(&force[neigh_idx][aa], -e_site_ders_ija);
                }
            }
        }
    }
}


template <typename CoordType>
__global__
void find_ef_kernel(CoordType *etot_ptr,
                    CoordType (*force)[3],
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
                    CoordType rmin)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = nx;

    if (ii<inum) {
        int silist = ilist[ii];
        int snumneigh = numneigh[ii];
        int *sfirstneigh = &firstneigh[ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = (CoordType (*)[3])(&rcs[ii*umax_num_neigh_atoms][0]);
        find_ef_atom<CoordType>(etot_ptr,
                                force,
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
                                type_map,
                                umax_num_neigh_atoms,
                                nghost,
                                rmax,
                                rmin);
    }
}


template <typename CoordType>
__host__
void find_ef_launcher(CoordType *h_etot_ptr,
                      CoordType (*h_force)[3],
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
                      CoordType rmin)
{
    int block_size_x = 128;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_etot_ptr;
    CoordType (*d_force)[3];
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

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    CHECK_CUDA_API( cudaMalloc((void**)&d_etot_ptr, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force, sizeof(CoordType) * (inum+nghost) * 3) );
    CHECK_CUDA_API( cudaMemset(d_etot_ptr, 0.0, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMemset(d_force, 0.0, sizeof(CoordType) * (inum+nghost) * 3) );
    
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_linear_coeffs, sizeof(CoordType) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_numneigh, sizeof(int) * inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_firstneigh, sizeof(int) * inum * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_rcs, sizeof(CoordType) * inum * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_types, sizeof(int) * (inum+nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );

    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType) * num_coeffs, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_linear_coeffs, h_linear_coeffs, sizeof(CoordType) * alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType) * ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int) * alpha_index_times_count * 4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_ilist, h_ilist, sizeof(int) * inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_numneigh, h_numneigh, sizeof(int) * inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_firstneigh, h_firstneigh, sizeof(int) * inum * umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_rcs, h_rcs, sizeof(CoordType) * inum * umax_num_neigh_atoms * 3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_types, h_types, sizeof(int) * (inum+nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int) * ntypes, cudaMemcpyHostToDevice) );


    // Launch kernel function
    auto t1 = std::chrono::high_resolution_clock::now();
    find_ef_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_etot_ptr,
        d_force,
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
        rmin);
    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_ef_launcher() cost time: " << duration.count() << " us.\n";

    CHECK_CUDA_API( cudaMemcpy(h_etot_ptr, d_etot_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_force, d_force, sizeof(CoordType) * (inum+nghost) * 3, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_etot_ptr) );
    CHECK_CUDA_API( cudaFree(d_force) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_linear_coeffs) );
    CHECK_CUDA_API( cudaFree(d_type_bias) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_basic) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_times) );
    CHECK_CUDA_API( cudaFree(d_alpha_moment_mapping) );
    CHECK_CUDA_API( cudaFree(d_ilist) );
    CHECK_CUDA_API( cudaFree(d_numneigh) );
    CHECK_CUDA_API( cudaFree(d_firstneigh) );
    CHECK_CUDA_API( cudaFree(d_rcs) );
    CHECK_CUDA_API( cudaFree(d_types) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
}


};  // namespace : mtpr
};  // namespace : ai2pot


#endif
