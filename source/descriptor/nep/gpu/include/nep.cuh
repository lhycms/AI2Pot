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

#ifndef AI2POT_NEP_NEP_CUH
#define AI2POT_NEP_NEP_CUH

#include <math.h>
#include <chrono>
#include <iostream>

#include "./nep_utilities.cuh"
#include "./nep_basis.cuh"


namespace ai2pot {
namespace nep {

template <typename CoordType>
static __host__ __device__
void find_ef_atom(
    CoordType *etot_ptr,
    CoordType (*force)[3],
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
void find_ef_kernel(
    CoordType *betot_ptr,
    CoordType (*bforce)[3],
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
void find_ef_launcher(
    CoordType *h_betot_ptr,
    CoordType (*h_bforce)[3],
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
static __host__ __device__
void find_efv_atom(
    CoordType *etot_ptr,
    CoordType (*force)[3],
    CoordType *virial,
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
    CoordType *q_scaler,
    CoordType *s_local_virial);

template <typename CoordType>
static __global__
void find_efv_kernel(
    CoordType *betot_ptr,
    CoordType (*bforce)[3],
    CoordType *bvirial,
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
static __host__ __device__
void find_descriptors_atom(
    CoordType *sbdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *coeffs,
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
    CoordType rmax_angular);


template <typename CoordType>
static __global__
void find_descriptors_kernel(
    CoordType *bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *coeffs,
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
    CoordType rmax_angular);


template <typename CoordType>
static __host__
void find_descriptors_launcher(
    CoordType *h_bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *d_coeffs,
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
    CoordType rmax_angular);






template <typename CoordType>
__host__ __device__
void find_ef_atom(
    CoordType *etot_ptr,
    CoordType (*force)[3],
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
    // Step 1. temp array
    CoordType mom_vals[MAX_NUM_SINLM] = {0.};
    CoordType e_sites_der2mom[MAX_NUM_SINLM] = {0.};
    CoordType dod_vals[MAX_NUM_DESCRIPTORS] = {0.};
    CoordType e_sites_der2dod[MAX_NUM_DESCRIPTORS] = {0.};
    CoordType auto_dist_powers_[L_MAX_PLUS_ONE] = {0.};

    //int num_coeffs = ntypes * ntypes * (n_radial_basis + n_angular_basis) * chebyshev_size;
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_Sinlm = NepIndex::get_num_Sinlm(n_angular_basis, l_max);

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    center_idx = silist;
    type_central = types[center_idx];
    CoordType *type_central_w0 = &w0[type_central * num_neurons * num_descriptors];
    CoordType *type_central_b0 = &b0[type_central * num_neurons];
    CoordType *type_central_w1 = &w1[type_central * num_neurons];

    // Step 2. forward
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        
        distance_ij = std::sqrt( neigh_vec[0] * neigh_vec[0]
                                 + neigh_vec[1] * neigh_vec[1]
                                 + neigh_vec[2] * neigh_vec[2] );
        if (distance_ij > rmax_radial)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals_radial[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_ders2r_radial[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_vals_angular[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_ders2r_angular[MAX_CHEBYSHEV_SIZE] = {0.};
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_radial,
            rq_chebyshev_ders2r_radial,
            chebyshev_size,
            rmax_radial,
            distance_ij);
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_angular,
            rq_chebyshev_ders2r_angular,
            chebyshev_size,
            rmax_angular,
            distance_ij);
        
        auto_dist_powers_[0] = 1.0;
        for (int k=1; k<=l_max; k++)
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;

        // Step 2.1. Radial forward
        if (distance_ij <= rmax_radial)
        for (int mu=0; mu<n_radial_basis; mu++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                          + mu*chebyshev_size + xi;
                dod_vals[mu] += coeffs[idx] * rq_chebyshev_vals_radial[xi];
            }
        }

        // Step 2.2. Angular forward: basic
        if (distance_ij <= rmax_angular)
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                CoordType C = 1/auto_dist_powers_[l];

                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    CoordType B = 0.0;
                    Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);

                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                  + (n_radial_basis+mu)*chebyshev_size + xi;

                        CoordType A = rq_chebyshev_vals_angular[xi];
                        mom_vals[idx_Sinlm] += coeffs[idx] * A * B * C;
                    }
                }
            }
        }
    }

    // Step 2.3. Angular forward: times
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);

            for (int mp=0; mp<2*l+1; mp++) {
                int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                int idx_Clm = NepIndex::get_Clm_index(l, mp);
                if (mp == 0)
                    dod_vals[n_radial_basis+idx_qinl] += (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm] * mom_vals[idx_Sinlm];
                else
                    dod_vals[n_radial_basis+idx_qinl] += 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm] * mom_vals[idx_Sinlm];
            }
        }
    }

    // Step 2.4. Ei
    CoordType e_site = type_bias[type_central];
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_val = 0.0;
        for (int k=0; k<num_descriptors; k++)
            hidden_val += type_central_w0[p*num_descriptors + k] * dod_vals[k] / q_scaler[k];
        hidden_val += type_central_b0[p];
        TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
        e_site += type_central_w1[p] * activated_hidden_val;
    }
    atomicAdd(etot_ptr, e_site);

    // Step 3. Backward (Force)
    // Step 3.1. RA basic
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        for (int k=0; k<num_descriptors; k++)
            hidden_val += type_central_w0[p*num_descriptors + k] * dod_vals[k] / q_scaler[k];
        hidden_val += type_central_b0[p];
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);

        for (int k=0; k<num_descriptors; k++)
            e_sites_der2dod[k] += type_central_w1[p]
                                  * activated_hidden_der
                                  * type_central_w0[p*num_descriptors + k]
                                  / q_scaler[k];
    }

    // Step 3.2. Angular_times
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            for (int mp=0; mp<2*l+1; mp++) {
                int idx_Clm = NepIndex::get_Clm_index(l, mp);
                int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);

                if (mp == 0)
                    e_sites_der2mom[idx_Sinlm] = e_sites_der2dod[n_radial_basis + idx_qinl]
                                                 * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
                else
                    e_sites_der2mom[idx_Sinlm] = 2 * e_sites_der2dod[n_radial_basis + idx_qinl]
                                                 * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
            }
        }
    }

    // Step 3.3. NN Force
    for (int jj=0; jj<snumneigh; jj++) {
        CoordType tmp_force_ij[3] = {0.0};
        CoordType tmp_force_ji[3] = {0.0};
        
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( neigh_vec[0] * neigh_vec[0]
                                 + neigh_vec[1] * neigh_vec[1]
                                 + neigh_vec[2] * neigh_vec[2] );
        if (distance_ij > rmax_radial)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals_radial[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_ders2r_radial[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_vals_angular[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_ders2r_angular[MAX_CHEBYSHEV_SIZE] = {0.};
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_radial,
            rq_chebyshev_ders2r_radial,
            chebyshev_size,
            rmax_radial,
            distance_ij);
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_angular,
            rq_chebyshev_ders2r_angular,
            chebyshev_size,
            rmax_angular,
            distance_ij);
        
        auto_dist_powers_[0] = 1.0;
        for (int k=1; k<=l_max; k++)
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;

        // Step 3.3.1. Raidal contribution
        if (distance_ij <= rmax_radial)
        for (int mu=0; mu<n_radial_basis; mu++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                            + mu*chebyshev_size + xi;

                CoordType prefix_mom_der2xyz = coeffs[idx] * rq_chebyshev_ders2r_radial[xi] * distance_ij_inv;
                for (int aa=0; aa<3; aa++) {
                    CoordType mom_der2xyz = prefix_mom_der2xyz * neigh_vec[aa];
                    CoordType e_site_ders_ija = e_sites_der2dod[mu] * mom_der2xyz;
                    tmp_force_ij[aa] += e_site_ders_ija;
                    tmp_force_ji[aa] += -e_site_ders_ija;
                    //atomicAdd(&force[center_idx][aa], e_site_ders_ija);
                    //atomicAdd(&force[neigh_idx][aa], -e_site_ders_ija);
                }
            }
        }

        // Step 3.3.2. Angular contribution
        if (distance_ij <= rmax_angular)
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    CoordType B = 0.0;
                    Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                    CoordType C = 1.0/auto_dist_powers_[l];
                    CoordType B_ders[3] = {0.0};
                    CoordType C_ders[3] = {0.0};
                    Blm<CoordType>::find_blm_der2xyz(B_ders, l, mp, neigh_vec, distance_ij);
                    C_ders[0] = -l / (auto_dist_powers_[l]*distance_ij)
                                    * (neigh_vec[0] * distance_ij_inv);
                    C_ders[1] = -l / (auto_dist_powers_[l]*distance_ij)
                                    * (neigh_vec[1] * distance_ij_inv);
                    C_ders[2] = -l / (auto_dist_powers_[l]*distance_ij)
                                    * (neigh_vec[2] * distance_ij_inv);

                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                    + (n_radial_basis+mu)*chebyshev_size + xi;
                        CoordType A = rq_chebyshev_vals_angular[xi];

                        CoordType A_ders[3] = {0.0};
                        A_ders[0] = rq_chebyshev_ders2r_angular[xi] * neigh_vec[0] * distance_ij_inv;
                        A_ders[1] = rq_chebyshev_ders2r_angular[xi] * neigh_vec[1] * distance_ij_inv;
                        A_ders[2] = rq_chebyshev_ders2r_angular[xi] * neigh_vec[2] * distance_ij_inv;   
                        
                        for (int aa=0; aa<3; aa++) {
                            CoordType mom_der2xyz = coeffs[idx] *
                                                    (A_ders[aa] * B * C
                                                    + A * B_ders[aa] * C
                                                    + A * B * C_ders[aa]);
                            CoordType e_site_ders_ija = e_sites_der2mom[idx_Sinlm] * mom_der2xyz;
                            tmp_force_ij[aa] += e_site_ders_ija;
                            tmp_force_ji[aa] += -e_site_ders_ija;
                        }
                    }
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
__global__
void find_ef_kernel(
    CoordType *betot_ptr,
    CoordType (*bforce)[3],
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

    CoordType *etot_ptr = &betot_ptr[istruct];
    CoordType (*force)[3] = &bforce[istruct*(natoms_pad+nghost)];
    int inum = binum[istruct];
    int *types = &btypes[istruct*(natoms_pad+nghost)];

    if (ii < inum) {
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = &brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];

        find_ef_atom<CoordType>(
            etot_ptr,
            force,
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
void find_ef_launcher(
    CoordType *h_betot_ptr,
    CoordType (*h_bforce)[3],
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

    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_coeffs = ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size;

    CoordType *d_betot_ptr;
    CoordType (*d_bforce)[3];
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

    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ptr, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce, sizeof(CoordType)*batch_size*(natoms_pad+nghost)*3) );
    CHECK_CUDA_API( cudaMemset(d_betot_ptr, 0, sizeof(CoordType)*batch_size) );
    CHECK_CUDA_API( cudaMemset(d_bforce, 0, sizeof(CoordType)*batch_size*(natoms_pad+nghost)*3) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType)*num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_w0, sizeof(CoordType)*ntypes*num_neurons*num_descriptors) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_b0, sizeof(CoordType)*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_w1, sizeof(CoordType)*ntypes*num_neurons) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_bias, sizeof(CoordType)*ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int)*batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int)*batch_size*natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int)*batch_size*(natoms_pad+nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int)*ntypes) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_q_scaler, sizeof(CoordType)*num_descriptors) );

    // 
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
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*batch_size*(natoms_pad+nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_q_scaler, h_q_scaler, sizeof(CoordType)*num_descriptors, cudaMemcpyHostToDevice) );

    // Compute
    auto t1 = std::chrono::high_resolution_clock::now();
    find_ef_kernel KERNEL_ARG2(grid_size, block_size) (
        d_betot_ptr,
        d_bforce,
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
    std::cout << "find_ef_kernel() cost time: " << duration.count() << " us.\n";

    // 
    CHECK_CUDA_API( cudaMemcpy(h_betot_ptr, d_betot_ptr, sizeof(CoordType)*batch_size, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bforce, d_bforce, sizeof(CoordType)*batch_size*(natoms_pad+nghost)*3, cudaMemcpyDeviceToHost) );

    // Step . Free
    CHECK_CUDA_API( cudaFree(d_betot_ptr) );
    CHECK_CUDA_API( cudaFree(d_bforce) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_w0) );
    CHECK_CUDA_API( cudaFree(d_b0) );
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
__host__ __device__
void find_efv_atom(
    CoordType *etot_ptr,
    CoordType (*force)[3],
    CoordType *virial,
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
    CoordType *q_scaler,
    CoordType *s_local_virial)
{
    // Step 1. temp array
    CoordType mom_vals[MAX_NUM_SINLM] = {0.};
    CoordType e_sites_der2mom[MAX_NUM_SINLM] = {0.};
    CoordType dod_vals[MAX_NUM_DESCRIPTORS] = {0.};
    CoordType e_sites_der2dod[MAX_NUM_DESCRIPTORS] = {0.};
    CoordType auto_dist_powers_[L_MAX_PLUS_ONE] {0.};

    //int num_coeffs = ntypes * ntypes * (n_radial_basis + n_angular_basis) * chebyshev_size;
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_Sinlm = NepIndex::get_num_Sinlm(n_angular_basis, l_max);

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    center_idx = silist;
    type_central = types[center_idx];
    CoordType *type_central_w0 = &w0[type_central * num_neurons * num_descriptors];
    CoordType *type_central_b0 = &b0[type_central * num_neurons];
    CoordType *type_central_w1 = &w1[type_central * num_neurons];

    // Step 2. forward
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        
        distance_ij = std::sqrt( neigh_vec[0] * neigh_vec[0]
                                 + neigh_vec[1] * neigh_vec[1]
                                 + neigh_vec[2] * neigh_vec[2] );
        if (distance_ij > rmax_radial)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals_radial[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_ders2r_radial[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_vals_angular[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_ders2r_angular[MAX_CHEBYSHEV_SIZE] = {0.};
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_radial,
            rq_chebyshev_ders2r_radial,
            chebyshev_size,
            rmax_radial,
            distance_ij);
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_angular,
            rq_chebyshev_ders2r_angular,
            chebyshev_size,
            rmax_angular,
            distance_ij);
        
        auto_dist_powers_[0] = 1.0;
        for (int k=1; k<=l_max; k++)
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
        
        // Step 2.1. Radial forward
        if (distance_ij <= rmax_radial)
        for (int mu=0; mu<n_radial_basis; mu++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                          + mu*chebyshev_size + xi;
                dod_vals[mu] += coeffs[idx] * rq_chebyshev_vals_radial[xi];
            }
        }

        // Step 2.2. Angular forward: basic
        if (distance_ij <= rmax_angular)
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                CoordType C = 1/auto_dist_powers_[l];

                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    CoordType B = 0.0;
                    Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);

                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                  + (n_radial_basis+mu)*chebyshev_size + xi;
                        
                        CoordType A = rq_chebyshev_vals_angular[xi];
                        mom_vals[idx_Sinlm] += coeffs[idx] * A * B * C;
                    }
                }
            }
        }
    }

    // Step 2.3. Angular forward: times
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);

            for (int mp=0; mp<2*l+1; mp++) {
                int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                int idx_Clm = NepIndex::get_Clm_index(l, mp);
                if (mp == 0)
                    dod_vals[n_radial_basis+idx_qinl] += (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm] * mom_vals[idx_Sinlm];
                else
                    dod_vals[n_radial_basis+idx_qinl] += 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm] * mom_vals[idx_Sinlm];
            }
        }
    }

    // Step 2.4. Ei
    CoordType e_site = type_bias[type_central];
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_val = 0.0;
        for (int k=0; k<num_descriptors; k++)
            hidden_val += type_central_w0[p*num_descriptors + k] * dod_vals[k] / q_scaler[k];
        hidden_val += type_central_b0[p];
        TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
        e_site += type_central_w1[p] * activated_hidden_val;
    }
    atomicAdd(etot_ptr, e_site);


    // Step 3. Backward (Force)
    // Step 3.1. RA basic
    for (int p=0; p<num_neurons; p++) {
        CoordType hidden_val = 0.0;
        CoordType activated_hidden_der = 0.0;
        for (int k=0; k<num_descriptors; k++)
            hidden_val += type_central_w0[p*num_descriptors + k] * dod_vals[k] / q_scaler[k];
        hidden_val += type_central_b0[p];
        TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);

        for (int k=0; k<num_descriptors; k++)
            e_sites_der2dod[k] += type_central_w1[p]
                                  * activated_hidden_der
                                  * type_central_w0[p*num_descriptors + k]
                                  / q_scaler[k];
    }

    // Step 3.2. Angular_times
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            for (int mp=0; mp<2*l+1; mp++) {
                int idx_Clm = NepIndex::get_Clm_index(l, mp);
                int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);

                if (mp == 0)
                    e_sites_der2mom[idx_Sinlm] = e_sites_der2dod[n_radial_basis + idx_qinl]
                                                 * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
                else
                    e_sites_der2mom[idx_Sinlm] = 2 * e_sites_der2dod[n_radial_basis + idx_qinl]
                                                 * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
            }
        }
    }

    // Step 3.3. NN Force
    for (int jj=0; jj<snumneigh; jj++) {
        CoordType tmp_force_ij[3] = {0.0};
        CoordType tmp_force_ji[3] = {0.0};

        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( neigh_vec[0] * neigh_vec[0] 
                                 + neigh_vec[1] * neigh_vec[1]
                                 + neigh_vec[2] * neigh_vec[2] );
        if (distance_ij > rmax_radial)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals_radial[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_ders2r_radial[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_vals_angular[MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_ders2r_angular[MAX_CHEBYSHEV_SIZE] = {0.};
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_radial,
            rq_chebyshev_ders2r_radial,
            chebyshev_size,
            rmax_radial,
            distance_ij);
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_angular,
            rq_chebyshev_ders2r_angular,
            chebyshev_size,
            rmax_angular,
            distance_ij);
        
        auto_dist_powers_[0] = 1.0;
        for (int k=1; k<=l_max; k++)
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
        
        // Step 3.3.1. Radial contribution
        if (distance_ij <= rmax_radial)
        for (int mu=0; mu<n_radial_basis; mu++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                          + mu*chebyshev_size + xi;
                
                CoordType prefix_mom_der2xyz = coeffs[idx] * rq_chebyshev_ders2r_radial[xi] * distance_ij_inv;
                for (int aa=0; aa<3; aa++) {
                    CoordType mom_der2xyz = prefix_mom_der2xyz * neigh_vec[aa];
                    CoordType e_site_ders_ija = e_sites_der2dod[mu] * mom_der2xyz;
                    tmp_force_ij[aa] += e_site_ders_ija;
                    tmp_force_ji[aa] += -e_site_ders_ija;

                    for (int bb=0; bb<3; bb++)
                        s_local_virial[aa*3 + bb] -= e_site_ders_ija * neigh_vec[bb];
                }
            }
        }

        // Step 3.3.2. Angular contribution
        if (distance_ij <= rmax_angular)
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    CoordType B = 0.0;
                    Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                    CoordType C = 1.0/auto_dist_powers_[l];
                    CoordType B_ders[3] = {0.0};
                    CoordType C_ders[3] = {0.0};
                    Blm<CoordType>::find_blm_der2xyz(B_ders, l, mp, neigh_vec, distance_ij);
                    C_ders[0] = -l / (auto_dist_powers_[l]*distance_ij)
                                    * (neigh_vec[0] * distance_ij_inv);
                    C_ders[1] = -l / (auto_dist_powers_[l]*distance_ij)
                                    * (neigh_vec[1] * distance_ij_inv);
                    C_ders[2] = -l / (auto_dist_powers_[l]*distance_ij)
                                    * (neigh_vec[2] * distance_ij_inv);
                    
                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                  + (n_radial_basis+mu)*chebyshev_size + xi;
                        CoordType A = rq_chebyshev_vals_angular[xi];

                        CoordType A_ders[3] = {0.0};
                        A_ders[0] = rq_chebyshev_ders2r_angular[xi] * neigh_vec[0] * distance_ij_inv;
                        A_ders[1] = rq_chebyshev_ders2r_angular[xi] * neigh_vec[1] * distance_ij_inv;
                        A_ders[2] = rq_chebyshev_ders2r_angular[xi] * neigh_vec[2] * distance_ij_inv;

                        for (int aa=0; aa<3; aa++) {
                            CoordType mom_der2xyz = coeffs[idx] * 
                                                    (A_ders[aa] * B * C
                                                     + A * B_ders[aa] * C
                                                     + A * B * C_ders[aa]);
                            CoordType e_site_ders_ija = e_sites_der2mom[idx_Sinlm] * mom_der2xyz;
                            tmp_force_ij[aa] += e_site_ders_ija;
                            tmp_force_ji[aa] += -e_site_ders_ija;

                            for (int bb=0; bb<3; bb++)
                                s_local_virial[aa*3 + bb] -= e_site_ders_ija * neigh_vec[bb];
                        }
                    }
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
__host__ __device__
void find_descriptors_atom(
    CoordType *sdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *coeffs,
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
    CoordType rmax_angular)
{
    // Step 1. Init temp array
    CoordType mom_vals[MAX_NUM_SINLM] = {0.0};
    CoordType dod_vals[MAX_NUM_DESCRIPTORS] = {0.0};
    CoordType auto_dist_powers_[L_MAX_PLUS_ONE] = {0.0};
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    center_idx = silist;
    type_central = types[center_idx];

    // Step 2. Calculate descriptors
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        
        distance_ij = std::sqrt( neigh_vec[0] * neigh_vec[0]
                                 + neigh_vec[1] * neigh_vec[1]
                                 + neigh_vec[2] * neigh_vec[2] );
        if (distance_ij > rmax_radial)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals_radial[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r_radial[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_vals_angular[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_der2r_angular[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_radial,
            rq_chebyshev_der2r_radial,
            chebyshev_size,
            rmax_radial,
            distance_ij);
        find_rq_chebyshev<CoordType>(
            rq_chebyshev_vals_angular,
            rq_chebyshev_der2r_angular,
            chebyshev_size,
            rmax_angular,
            distance_ij);
        
        auto_dist_powers_[0] = 1.0;
        for (int k=1; k<=l_max; k++)
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
        
        // Step 2.1. 
        if (distance_ij <= rmax_radial)
        for (int mu=0; mu<n_radial_basis; mu++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                          + mu*chebyshev_size + xi;
                dod_vals[mu] += coeffs[idx] * rq_chebyshev_vals_radial[xi];
            }
        }

        // Step 2.2.
        if (distance_ij <= rmax_angular)
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    CoordType B = 0.0;
                    Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                    CoordType C = 1.0 / auto_dist_powers_[l];

                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                   + (n_radial_basis+mu)*chebyshev_size + xi;
                        CoordType A = rq_chebyshev_vals_angular[xi];
                        mom_vals[idx_Sinlm] += coeffs[idx] * A * B * C;
                    }
                }
            }
        }
    }
    // Step 2.3. 
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            for (int mp=0; mp<2*l+1; mp++) {
                int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);
                int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                int idx_Clm = NepIndex::get_Clm_index(l, mp);
                
                if (mp == 0)
                    dod_vals[n_radial_basis+idx_qinl] += (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm] * mom_vals[idx_Sinlm];
                else
                    dod_vals[n_radial_basis+idx_qinl] += 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm] * mom_vals[idx_Sinlm];
            }
        }
    }

    for (int k=0; k<num_descriptors; k++)
        sdescriptors[k] = dod_vals[k];
}


template <typename CoordType>
__global__
void find_descriptors_kernel(
    CoordType *bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *coeffs,
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
    CoordType rmax_angular)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    if (istruct >= batch_size)
        return;
    int inum = binum[istruct];
    int ii = nx % natoms_pad;
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);

    int *types = &btypes[istruct*natoms_pad + 0];

    if (ii < inum) {
        CoordType *sdescriptors = &bdescriptors[istruct*natoms_pad*num_descriptors + ii*num_descriptors];
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = &brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];

        find_descriptors_atom<CoordType>(
            sdescriptors,
            chebyshev_size,
            n_radial_basis,
            n_angular_basis,
            l_max,
            coeffs,
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
            rmax_angular);
    }
}


template <typename CoordType>
__host__
void find_descriptors_launcher(
    CoordType *h_bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *h_coeffs,
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
    CoordType rmax_angular)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_coeffs = ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size;

    CoordType *d_bdescriptors;
    CoordType *d_coeffs;
    int *d_binum;
    int *d_bilist;
    int *d_bnumneigh;
    int *d_bfirstneigh;
    CoordType (*d_brcs)[3];
    int *d_btypes;
    int *d_type_map;

    CHECK_CUDA_API( cudaMalloc((void**)&d_bdescriptors, sizeof(CoordType) * batch_size * natoms_pad * num_descriptors) );
    CHECK_CUDA_API( cudaMemset(d_bdescriptors, 0, sizeof(CoordType) * batch_size * natoms_pad * num_descriptors) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * num_coeffs) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int) * batch_size * (natoms_pad + nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );

    //
    CHECK_CUDA_API( cudaMemcpy(d_coeffs, h_coeffs, sizeof(CoordType) * num_coeffs, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int) * batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int) * batch_size * natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int) * batch_size * natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int) * batch_size * (natoms_pad + nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int) * ntypes, cudaMemcpyHostToDevice) );

    // Compute
    find_descriptors_kernel KERNEL_ARG2(grid_size, block_size) (
        d_bdescriptors,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        d_coeffs,
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
        rmax_angular);
    CHECK_CUDA_KERNEL;

    // 
    CHECK_CUDA_API( cudaMemcpy(h_bdescriptors, d_bdescriptors, sizeof(CoordType) * batch_size * natoms_pad * num_descriptors, cudaMemcpyDeviceToHost) );
    
    // Step . Free
    CHECK_CUDA_API( cudaFree(d_bdescriptors) );
    CHECK_CUDA_API( cudaFree(d_coeffs) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_bnumneigh) );
    CHECK_CUDA_API( cudaFree(d_bfirstneigh) );
    CHECK_CUDA_API( cudaFree(d_brcs) );
    CHECK_CUDA_API( cudaFree(d_btypes) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
}


};  // namespace : nep
};  // namespace : ai2pot

#endif
