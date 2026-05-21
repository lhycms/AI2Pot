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
void find_efv_atom(
    CoordType *etot_ptr,
    CoordType (*force)[3],
    CoordType *virial,
    int chebyshev_size,
    CoordType scaling,
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
    CoordType *q_scaler,
    CoordType *s_local_virial);


template <typename CoordType>
static __global__
void find_efv_kernel(
    CoordType *betot_ptr,
    CoordType (*bforce)[3],
    CoordType *bvirial,
    int chebyshev_size,
    CoordType scaling,
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
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_scaler);


template <typename CoordType>
static __host__
void find_efv_launcher(
    CoordType *h_betot_ptr,
    CoordType (*h_bforce)[3],
    CoordType *h_bvirial,
    int chebyshev_size,
    CoordType scaling,
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
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_q_scaler);
    

template <typename CoordType>
static __host__ __device__
void find_ef_atom(
    CoordType *etot_ptr,
    CoordType (*force)[3],
    int chebyshev_size,
    CoordType scaling,
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
    CoordType *q_scaler);


template <typename CoordType>
static __global__ 
void find_ef_kernel(
    CoordType *betot_ptr,
    CoordType (*bforce)[3],
    int chebyshev_size,
    CoordType scaling,
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
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_scaler);


template <typename CoordType>
static __host__
void find_ef_launcher(
    CoordType *h_betot_ptr,
    CoordType (*h_bforce)[3],
    int chebyshev_size,
    CoordType scaling,
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
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_q_scaler);


template <typename CoordType>
static __host__ __device__
void find_descriptors_atom(
    CoordType *sdescriptors,
    int chebyshev_size,
    CoordType scaling,
    CoordType *coeffs,
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
void find_descriptors_kernel(
    CoordType *bdescriptors,
    int chebyshev_size,
    CoordType scaling,
    CoordType *coeffs,
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
void find_descriptors_launcher(
    CoordType *h_bdescriptors,
    int chebyshev_size,
    CoordType scaling,
    CoordType *h_coeffs,
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
__host__ __device__ 
void find_efv_atom(
    CoordType *etot_ptr,
    CoordType (*force)[3],
    CoordType *virial,
    int chebyshev_size,
    CoordType scaling,
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
    CoordType *q_scaler,
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
        
        distance_ij = std::sqrt( NeighbVect[0]*NeighbVect[0]
                                 + NeighbVect[1]*NeighbVect[1]
                                 + NeighbVect[2]*NeighbVect[2] );
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
        e_site += linear_coeffs[i] 
                  * mom_vals[alpha_moment_mapping[i]]
                  / q_scaler[i];
    atomicAdd(etot_ptr, e_site);

    for (int i=0; i<alpha_scalar_moments; i++)
        e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i] / q_scaler[i];

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
        CoordType tmp_force_ij[3] = {0.0};
        CoordType tmp_force_ji[3] = {0.0};

        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        Zj = type_map[type_outer];
        for (int aa=0; aa<3; aa++)
            NeighbVect[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( NeighbVect[0]*NeighbVect[0]
                                 + NeighbVect[1]*NeighbVect[1]
                                 + NeighbVect[2]*NeighbVect[2] );
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
                    tmp_force_ij[aa] += e_site_ders_ija;
                    tmp_force_ji[aa] -= e_site_ders_ija;

                    for (int bb=0; bb<3; bb++)
                        //atomicAdd(&virial[aa*3 + bb], -e_site_ders_ija * NeighbVect[bb]);
                        s_local_virial[aa*3 + bb] -= e_site_ders_ija * NeighbVect[bb];
                 }
            }
        }
        atomicAdd(&force[center_idx][0], tmp_force_ij[0]);
        atomicAdd(&force[center_idx][1], tmp_force_ij[1]);
        atomicAdd(&force[center_idx][2], tmp_force_ij[2]);
        atomicAdd(&force[neigh_idx][0], tmp_force_ji[0]);
        atomicAdd(&force[neigh_idx][1], tmp_force_ji[1]);
        atomicAdd(&force[neigh_idx][2], tmp_force_ji[2]);
    }
}


template <typename CoordType>
static __global__
void find_efv_kernel(
    CoordType *betot_ptr,
    CoordType (*bforce)[3],
    CoordType *bvirial,
    int chebyshev_size,
    CoordType scaling,
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
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_scaler)
{
    CoordType s_local_virial[9];
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    int ii = nx % natoms_pad;
    if (istruct >= batch_size)
        return;

    for (int v=0; v<9; v++)
        s_local_virial[v] = 0.0;

    CoordType *etot_ptr = &betot_ptr[istruct];
    CoordType (*force)[3] = &bforce[istruct*(natoms_pad+nghost) + 0];
    CoordType *virial = &bvirial[istruct*9 + 0];
    int inum = binum[istruct];
    int *types = &btypes[istruct*(natoms_pad+nghost)];

    if (ii < inum) {
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = (CoordType (*)[3])(&brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms][0]);
        find_efv_atom<CoordType>(
            etot_ptr,
            force,
            virial,
            chebyshev_size,
            scaling,
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
            q_scaler,
            s_local_virial);
        
        for (int v=0; v<9; v++)
            atomicAdd(&virial[v], s_local_virial[v]);
    }
}


template <typename CoordType>
__host__
void find_efv_launcher(
    CoordType *h_betot_ptr,
    CoordType (*h_bforce)[3],
    CoordType *h_bvirial,
    int chebyshev_size,
    CoordType scaling,
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
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_q_scaler)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size * natoms_pad- 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_betot_ptr;
    CoordType (*d_bforce)[3];
    CoordType *d_bvirial;
    CoordType *d_coeffs;
    CoordType *d_linear_coeffs;
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
    CoordType *d_q_scaler;

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ptr, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce, sizeof(CoordType) * batch_size * (natoms_pad + nghost) * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial, sizeof(CoordType) * batch_size * 9) );
    CHECK_CUDA_API( cudaMemset(d_betot_ptr, 0.0, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMemset(d_bforce, 0.0, sizeof(CoordType) * batch_size * (natoms_pad + nghost) * 3) );
    CHECK_CUDA_API( cudaMemset(d_bvirial, 0.0, sizeof(CoordType) * batch_size * 9) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_linear_coeffs, sizeof(CoordType) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int) * batch_size * (natoms_pad + nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_q_scaler, sizeof(CoordType) * alpha_scalar_moments) );

    //CHECK_CUDA_API( cudaMemcpy(d_etot_ptr, &h_etot, sizeof(CoordType), cudaMemcpyHostToDevice) );
    //CHECK_CUDA_API( cudaMemcpy(d_force, h_force, sizeof(CoordType)*(inum+nghost)*3, cudaMemcpyHostToDevice) );
    //CHECK_CUDA_API( cudaMemcpy(d_virial, h_virial, sizeof(CoordType)*9, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType)*ntypes*ntypes*nmus*chebyshev_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_linear_coeffs, h_linear_coeffs, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int)*alpha_index_basic_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int)*alpha_index_times_count*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int)*alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*batch_size*(natoms_pad+nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_q_scaler, h_q_scaler, sizeof(CoordType)*alpha_scalar_moments, cudaMemcpyHostToDevice) );

    // Call global function
    auto t1 = std::chrono::high_resolution_clock::now();
    find_efv_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_betot_ptr,
        d_bforce,
        d_bvirial,
        chebyshev_size,
        scaling,
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
        d_type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        d_q_scaler);
    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_efv_kernel() cost time: " << duration.count() << " us.\n";

    CHECK_CUDA_API( cudaMemcpy(h_betot_ptr, d_betot_ptr, sizeof(CoordType)*batch_size, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bforce, d_bforce, sizeof(CoordType)*batch_size*(natoms_pad+nghost)*3, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bvirial, d_bvirial, sizeof(CoordType)*batch_size*9, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_betot_ptr) );
    CHECK_CUDA_API( cudaFree(d_bforce) );
    CHECK_CUDA_API( cudaFree(d_bvirial) );
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
    CHECK_CUDA_API( cudaFree(d_type_map) );
    CHECK_CUDA_API( cudaFree(d_q_scaler) );
}


template <typename CoordType>
__host__ __device__
void find_ef_atom(CoordType *etot_ptr,
                  CoordType (*force)[3],
                  int chebyshev_size,
                  CoordType scaling,
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
                  CoordType *q_scaler)
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
        
        distance_ij = std::sqrt( NeighbVect[0]*NeighbVect[0]
                                 + NeighbVect[1]*NeighbVect[1]
                                 + NeighbVect[2]*NeighbVect[2] );
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
        e_site += linear_coeffs[i] 
                  * mom_vals[alpha_moment_mapping[i]]
                  / q_scaler[i];
    atomicAdd(etot_ptr, e_site);

    for (int i=0; i<alpha_scalar_moments; i++)
        e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i] / q_scaler[i];

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
        distance_ij = std::sqrt( NeighbVect[0]*NeighbVect[0]
                                 + NeighbVect[1]*NeighbVect[1]
                                 + NeighbVect[2]*NeighbVect[2] );
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
void find_ef_kernel(
    CoordType *betot_ptr,
    CoordType (*bforce)[3],
    int chebyshev_size,
    CoordType scaling,
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
    int *type_map,
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

    CoordType *etot_ptr = &betot_ptr[istruct];
    CoordType (*force)[3] = &bforce[istruct*(natoms_pad+nghost) + 0];
    int inum = binum[istruct];
    int *types = &btypes[istruct*(natoms_pad+nghost)];

    if (ii<inum) {
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = (CoordType (*)[3])(&brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms][0]);
        find_ef_atom<CoordType>(
            etot_ptr,
            force,
            chebyshev_size,
            scaling,
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
            q_scaler);
    }
}


template <typename CoordType>
__host__
void find_ef_launcher(
    CoordType *h_betot_ptr,
    CoordType (*h_bforce)[3],
    int chebyshev_size,
    CoordType scaling,
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
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_q_scaler)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_betot_ptr;
    CoordType (*d_bforce)[3];
    CoordType *d_coeffs;
    CoordType *d_linear_coeffs;
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
    CoordType *d_q_scaler;

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ptr, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce, sizeof(CoordType) * batch_size * (natoms_pad+nghost) * 3) );
    CHECK_CUDA_API( cudaMemset(d_betot_ptr, 0.0, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMemset(d_bforce, 0.0, sizeof(CoordType) * batch_size * (natoms_pad+nghost) * 3) );
    
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_linear_coeffs, sizeof(CoordType) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int) * batch_size * (natoms_pad+nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_q_scaler, sizeof(CoordType) * alpha_scalar_moments) );

    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType) * num_coeffs, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_linear_coeffs, h_linear_coeffs, sizeof(CoordType) * alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_bias, h_type_bias, sizeof(CoordType) * ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int) * alpha_index_times_count * 4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int) * batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int) * batch_size * natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int) * batch_size * natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int) * batch_size * (natoms_pad+nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int) * ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_q_scaler, h_q_scaler, sizeof(CoordType) * alpha_scalar_moments, cudaMemcpyHostToDevice) );

    // Launch kernel function
    auto t1 = std::chrono::high_resolution_clock::now();
    find_ef_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_betot_ptr,
        d_bforce,
        chebyshev_size,
        scaling,
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
        d_type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        d_q_scaler);
    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    std::cout << "find_ef_kernel() cost time: " << duration.count() << " us.\n";

    CHECK_CUDA_API( cudaMemcpy(h_betot_ptr, d_betot_ptr, sizeof(CoordType) * batch_size, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bforce, d_bforce, sizeof(CoordType) * batch_size * (natoms_pad+nghost) * 3, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_betot_ptr) );
    CHECK_CUDA_API( cudaFree(d_bforce) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_linear_coeffs) );
    CHECK_CUDA_API( cudaFree(d_type_bias) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_basic) );
    CHECK_CUDA_API( cudaFree(d_alpha_index_times) );
    CHECK_CUDA_API( cudaFree(d_alpha_moment_mapping) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_bnumneigh) );
    CHECK_CUDA_API( cudaFree(d_bfirstneigh) );
    CHECK_CUDA_API( cudaFree(d_brcs) );
    CHECK_CUDA_API( cudaFree(d_btypes) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
    CHECK_CUDA_API( cudaFree(d_q_scaler) );
}


template <typename CoordType>
__host__ __device__
void find_descriptors_atom(
    CoordType *sdescriptors,
    int chebyshev_size,
    CoordType scaling,
    CoordType *coeffs,
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
    // Step 1. Init temp array
    CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.};

    CoordType auto_dist_powers_[MAX_ALPHA_INDEX_BASIC] = {0.};
    CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3] = {{0.}};

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    center_idx = silist;
    type_central = types[center_idx];

    // Step 2. Calculate mom_vals
    // Step 2.1.
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];

        distance_ij = std::sqrt( neigh_vec[0]*neigh_vec[0]
                                 + neigh_vec[1]*neigh_vec[1]
                                 + neigh_vec[2]*neigh_vec[2] );
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
                mom_vals[i] += coeffs[idx] * A * B * C;
            }
        }
    }

    // Step 2.2.
    for (int i=0; i<alpha_index_times_count; i++) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];
        mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
    }

    // Step 2.3.
    for (int k=0; k<alpha_scalar_moments; k++)
        sdescriptors[k] = mom_vals[alpha_moment_mapping[k]];
}


template <typename CoordType>
__global__ void find_descriptors_kernel(
    CoordType *bdescriptors,
    int chebyshev_size,
    CoordType scaling,
    CoordType *coeffs,
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
    int inum = binum[istruct];
    int ii = nx % natoms_pad;

    int *types = &btypes[istruct*(natoms_pad+nghost) + 0];

    if (ii<inum) {
        CoordType *sdescriptors = &bdescriptors[istruct*natoms_pad*alpha_scalar_moments + ii*alpha_scalar_moments + 0];
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms + 0];
        CoordType (*srcs)[3] = &brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms + 0];


        find_descriptors_atom<CoordType>(
            sdescriptors,
            chebyshev_size,
            scaling,
            coeffs,
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
static __host__
void find_descriptors_launcher(
    CoordType *h_bdescriptors,
    int chebyshev_size,
    CoordType scaling,
    CoordType *h_coeffs,
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
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_bdescriptors;
    CoordType *d_coeffs;
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

    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    CHECK_CUDA_API( cudaMalloc((void**)&d_bdescriptors, sizeof(CoordType) * batch_size * natoms_pad * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMemset(d_bdescriptors, 0, sizeof(CoordType) * batch_size * natoms_pad * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int) * batch_size * (natoms_pad + nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );

    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType) * num_coeffs, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_basic, h_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_index_times, h_alpha_index_times, sizeof(int) * alpha_index_times_count * 4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_alpha_moment_mapping, h_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int) * batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int) * batch_size * natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int) * batch_size * natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int) * batch_size * (natoms_pad + nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int) * ntypes, cudaMemcpyHostToDevice) );

    find_descriptors_kernel KERNEL_ARG2(grid_size, block_size) (
        d_bdescriptors,
        chebyshev_size,
        scaling,
        d_coeffs,
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

    CHECK_CUDA_API( cudaMemcpy(h_bdescriptors, d_bdescriptors, sizeof(CoordType) * batch_size * natoms_pad * alpha_scalar_moments, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_bdescriptors) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
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


};  // namespace : mtpr
};  // namespace : ai2pot


#endif
