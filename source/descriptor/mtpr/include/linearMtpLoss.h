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

#ifndef AI2POT_MTP_LOSS_H
#define AI2POT_MTP_LOSS_H
#include <stdlib.h>
#include <cstring>
#include <cmath>

#include "./basis.h"


namespace ai2pot {
namespace mtpr {

template <typename CoordType>
class LinearMtpLoss {
public:
    static void find_loss(
        CoordType &loss,
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
    
    static void find_ef_loss(
        CoordType &loss,
        int inum,
        int *ilist,
        CoordType e_weight,
        CoordType f_weight,
        CoordType etot_ml,
        CoordType etot_dft,
        CoordType (*force_ml)[3],
        CoordType (*force_dft)[3]);
    

    static void find_loss_backward(
        CoordType *loss_der2coeffs,
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
        CoordType (*relative_coords)[3],
        int *types,
        int ntypes,
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin,
        CoordType *q_shifter,
        CoordType *q_scaler);


    static void find_ef_loss_backward(
        CoordType *loss_der2coeffs,
        CoordType *loss_der2linear_coeffs,
        CoordType *loss_der2type_bias,
        CoordType e_weight,
        CoordType f_weight,
        CoordType etot_ml,
        CoordType etot_dft,
        CoordType (*force_ml)[3],
        CoordType (*force_dft)[3],
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
        CoordType (*relative_coords)[3],
        int *types,
        int ntypes,
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin,
        CoordType *q_shifter,
        CoordType *q_scaler);
};  // class : LinearMtpLoss


template <typename CoordType>
void LinearMtpLoss<CoordType>::find_loss(
    CoordType &loss,
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
    loss = 0.0;

    // Force term
    CoordType f_loss = 0;
    for (int ii=0; ii<inum; ii++) {
        int center_idx = ilist[ii];
        for (int aa=0; aa<3; aa++) {
            f_loss += std::pow(force_ml[center_idx][aa] - force_dft[center_idx][aa], 2);
        }
    }
    f_loss = f_weight / (3 * inum) * f_loss;

    // Virial term
    CoordType v_loss = 0;
    for (int aa=0; aa<3; aa++)
        for (int bb=0; bb<3; bb++)
            v_loss += std::pow(virial_ml[aa*3+bb] - virial_dft[aa*3+bb], 2);
    v_loss = v_weight / (9 * inum) * v_loss;

    // Energy term
    CoordType e_loss = 0;
    e_loss += e_weight / inum * std::pow(etot_ml - etot_dft, 2);

    loss = f_loss + v_loss + e_loss;
}


template <typename CoordType>
void LinearMtpLoss<CoordType>::find_ef_loss(
    CoordType &loss,
    int inum,
    int *ilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3])
{
    loss = 0.0;

    // Force term
    CoordType f_loss = 0.0;
    for (int ii=0; ii<inum; ii++) {
        int center_idx = ilist[ii];
        for (int aa=0; aa<3; aa++) {
            f_loss += std::pow(force_ml[center_idx][aa] - force_dft[center_idx][aa], 2);
        }
    }
    f_loss = f_weight / (3 * inum) * f_loss;
    
    // Energy term
    CoordType e_loss = 0.0;
    e_loss = e_weight / inum * std::pow(etot_ml - etot_dft, 2);

    loss = f_loss + e_loss;
}


template <typename CoordType>
void LinearMtpLoss<CoordType>::find_loss_backward(
    CoordType *loss_der2coeffs,
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
    CoordType (*relative_coords)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_shifter,
    CoordType *q_scaler)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType *e_site_der2mom;
    CoordType *dloss_combination;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    int max_alpha_index_basic = 0;
    for (int ii=0; ii<alpha_index_basic_count; ii++) {
        int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][3];
        if (now_alpha_index_basic > max_alpha_index_basic)
            max_alpha_index_basic = now_alpha_index_basic;
    }
    max_alpha_index_basic++;

    CoordType *auto_dist_powers_;
    CoordType (*auto_coords_powers_)[3];
    RQ_Chebyshev<CoordType> *p_RadialBasis;

    // Step 2. efv : delete

    // Step 3.
#ifdef USE_OPENMP
#pragma omp parallel private(mom_vals, e_site_der2mom, dloss_combination, auto_dist_powers_, auto_coords_powers_, p_RadialBasis)
{
#endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    e_site_der2mom = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    dloss_combination = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic);
    auto_coords_powers_ = (CoordType (*)[3])malloc(sizeof(CoordType) * max_alpha_index_basic * 3);
    p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);
    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    #ifdef USE_OPENMP
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++) {
        // Step 3. mom_vals
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(e_site_der2mom, 0, sizeof(CoordType) * alpha_moments_count);
        memset(dloss_combination, 0, sizeof(CoordType) * alpha_moments_count);
        center_idx = ilist[ii];
        type_central = types[center_idx];

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            NeighbVect[0] = relative_coords[ii*umax_num_neigh_atoms + jj][0];
            NeighbVect[1] = relative_coords[ii*umax_num_neigh_atoms + jj][1];
            NeighbVect[2] = relative_coords[ii*umax_num_neigh_atoms + jj][2];
            distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                     + std::pow(NeighbVect[1], 2)
                                     + std::pow(NeighbVect[2], 2) );
            if (distance_ij > rmax)
                continue;
            distance_ij_inv = 1.0 / distance_ij;
            p_RadialBasis->build(distance_ij);

            auto_dist_powers_[0] = 1.0;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[0][aa] = 1.0;
            for (int k=1; k<max_alpha_index_basic; k++) {
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
                    CoordType A = p_RadialBasis->vals()[xi];
                    CoordType B = mult0;
                    CoordType C = powk;
                    CoordType A_ders[3] = {0.0, 0.0, 0.0};
                    CoordType B_ders[3] = {0.0, 0.0, 0.0};
                    CoordType C_ders[3] = {0.0, 0.0, 0.0};
                    A_ders[0] = p_RadialBasis->ders2r()[xi] * NeighbVect[0] * distance_ij_inv;
                    A_ders[1] = p_RadialBasis->ders2r()[xi] * NeighbVect[1] * distance_ij_inv;
                    A_ders[2] = p_RadialBasis->ders2r()[xi] * NeighbVect[2] * distance_ij_inv;
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

        // Step 4.1. Linear Energy derivative w.r.t. xyz
        for (int i=0; i<alpha_scalar_moments; i++)
            e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i] / q_scaler[i];

        // Step 4.2. Pass to basic moments
        for (int i=alpha_index_times_count-1; i>=0; i--) {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];

            e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2 * val1;
            e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2 * val0;
        }

        // Step 4.3. Loss derivative w.r.t. coeffs
        for (int jj=0; jj<numneigh[ii]; jj++) 
        {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            NeighbVect[0] = relative_coords[ii*umax_num_neigh_atoms+jj][0];
            NeighbVect[1] = relative_coords[ii*umax_num_neigh_atoms+jj][1];
            NeighbVect[2] = relative_coords[ii*umax_num_neigh_atoms+jj][2];
            distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                    + std::pow(NeighbVect[1], 2)
                                    + std::pow(NeighbVect[2], 2) );
            if (distance_ij > rmax)
                continue;
            distance_ij_inv = 1.0 / distance_ij;
            p_RadialBasis->build(distance_ij);

            auto_dist_powers_[0] = 1.0;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[0][aa] = 1.0;
            for (int k=1; k<max_alpha_index_basic; k++) {
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
                    CoordType A = p_RadialBasis->vals()[xi];
                    CoordType B = mult0;
                    CoordType C = powk;
                    CoordType A_ders[3] = {0., 0., 0.};
                    CoordType B_ders[3] = {0., 0., 0.};
                    CoordType C_ders[3] = {0., 0., 0.};
                    A_ders[0] = p_RadialBasis->ders2r()[xi] * NeighbVect[0] * distance_ij_inv;
                    A_ders[1] = p_RadialBasis->ders2r()[xi] * NeighbVect[1] * distance_ij_inv;
                    A_ders[2] = p_RadialBasis->ders2r()[xi] * NeighbVect[2] * distance_ij_inv;
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

                    CoordType tmpe_loss_der2coeff = 0.0;
                    tmpe_loss_der2coeff = 2*e_weight/inum*(etot_ml - etot_dft) 
                                          * e_site_der2mom[i]
                                          * A * B * C;
                    
                    CoordType tmpf_loss_der2coeff = 0.0;
                    for (int aa=0; aa<3; aa++) 
                    {
                        CoordType tmp_prefix = 0.0;
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
                        tmpf_loss_der2coeff += tmp_prefix * tmp_deriv;
                    }
                    #ifdef USE_OPENMP
                    #pragma omp atomic
                    #endif
                    loss_der2coeffs[idx] += (tmpe_loss_der2coeff + tmpf_loss_der2coeff * e_site_der2mom[i]);
                }
            }
        }

        // Step 4.4. Loss derivative w.r.t. linear_coeffs
        for (int i=0; i<alpha_scalar_moments; i++) {
            CoordType tmp_loss_der2linear_coeff = 2*e_weight/inum
                                                  * (etot_ml - etot_dft)
                                                  * (mom_vals[alpha_moment_mapping[i]] - q_shifter[i])
                                                  + dloss_combination[alpha_moment_mapping[i]];
            #ifdef USE_OPENMP
            #pragma omp atomic
            #endif
            loss_der2linear_coeffs[i] += tmp_loss_der2linear_coeff / q_scaler[i];
        }

        // Step 4.5. Loss derivative w.r.t. type_bias
        #ifdef USE_OPENMP
        #pragma omp atomic
        #endif
        loss_der2type_bias[type_central] += 2*e_weight/inum*(etot_ml - etot_dft);
    }

    // Step . Free
    free(mom_vals);
    free(e_site_der2mom);
    free(dloss_combination);
    free(auto_dist_powers_);
    free(auto_coords_powers_);
    delete p_RadialBasis;

#ifdef USE_OPENMP
}
#endif
}



template <typename CoordType>
void LinearMtpLoss<CoordType>::find_ef_loss_backward(
    CoordType *loss_der2coeffs,
    CoordType *loss_der2linear_coeffs,
    CoordType *loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
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
    CoordType (*relative_coords)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_shifter,
    CoordType *q_scaler)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType *e_site_der2mom;
    CoordType *dloss_combination;
    CoordType *de22m0m1_dloss_combinations;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    int max_alpha_index_basic = 0;
    for (int ii=0; ii<alpha_index_basic_count; ii++) {
        int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][3];
        if (now_alpha_index_basic > max_alpha_index_basic)
            max_alpha_index_basic = now_alpha_index_basic;
    }
    max_alpha_index_basic++;

    CoordType *auto_dist_powers_;
    CoordType (*auto_coords_powers_)[3];
    RQ_Chebyshev<CoordType> *p_RadialBasis;

    // Step 2. ef : delete

    // Step 3.
#ifdef USE_OPENMP
#pragma omp parallel private(mom_vals, e_site_der2mom, dloss_combination, de22m0m1_dloss_combinations, auto_dist_powers_, auto_coords_powers_, p_RadialBasis)
{
#endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    e_site_der2mom = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    dloss_combination = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    de22m0m1_dloss_combinations = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic);
    auto_coords_powers_ = (CoordType (*)[3])malloc(sizeof(CoordType) * max_alpha_index_basic * 3);
    p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);
    
    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    #ifdef USE_OPENMP
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++) {
        // Step 3. mom_vals
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(e_site_der2mom, 0, sizeof(CoordType) * alpha_moments_count);
        memset(dloss_combination, 0, sizeof(CoordType) * alpha_moments_count);
        memset(de22m0m1_dloss_combinations, 0, sizeof(CoordType) * alpha_moments_count);
        center_idx = ilist[ii];
        type_central = types[center_idx];

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            NeighbVect[0] = relative_coords[ii*umax_num_neigh_atoms + jj][0];
            NeighbVect[1] = relative_coords[ii*umax_num_neigh_atoms + jj][1];
            NeighbVect[2] = relative_coords[ii*umax_num_neigh_atoms + jj][2];
            distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                     + std::pow(NeighbVect[1], 2)
                                     + std::pow(NeighbVect[2], 2));
            if (distance_ij > rmax)
                continue;
            distance_ij_inv = 1.0 / distance_ij;
            p_RadialBasis->build(distance_ij);

            auto_dist_powers_[0] = 1.0;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[0][aa] = 1.0;
            for (int k=1; k<max_alpha_index_basic; k++) {
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
                    CoordType A = p_RadialBasis->vals()[xi];
                    CoordType B = mult0;
                    CoordType C = powk;
                    CoordType A_ders[3] = {0., 0., 0.};
                    CoordType B_ders[3] = {0., 0., 0.};
                    CoordType C_ders[3] = {0., 0., 0.};
                    A_ders[0] = p_RadialBasis->ders2r()[xi] * NeighbVect[0] * distance_ij_inv;
                    A_ders[1] = p_RadialBasis->ders2r()[xi] * NeighbVect[1] * distance_ij_inv;
                    A_ders[2] = p_RadialBasis->ders2r()[xi] * NeighbVect[2] * distance_ij_inv;
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

        // Step 4.1. Linear Energy derivative w.r.t. xyz
        for (int i=0; i<alpha_scalar_moments; i++)
            e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i] / q_scaler[i];

        // Step 4.2. Pass to basic moments
        for (int i=alpha_index_times_count-1; i>=0; i--) {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];

            e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2 * val1;
            e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2 * val0;
        }

        // New code
        for (int i=0; i<alpha_index_times_count; i++) {
            CoordType val2 = alpha_index_times[i][2];
            de22m0m1_dloss_combinations[alpha_index_times[i][1]] += val2 * e_site_der2mom[alpha_index_times[i][3]]
                                                                    * dloss_combination[alpha_index_times[i][0]];
            de22m0m1_dloss_combinations[alpha_index_times[i][0]] += val2 * e_site_der2mom[alpha_index_times[i][3]]
                                                                    * dloss_combination[alpha_index_times[i][1]];
        }

        for (int i=alpha_index_times_count-1; i>=0; i--) {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];
            
            de22m0m1_dloss_combinations[alpha_index_times[i][0]] += de22m0m1_dloss_combinations[alpha_index_times[i][3]]
                                                                    * val2 * val1;
            de22m0m1_dloss_combinations[alpha_index_times[i][1]] += de22m0m1_dloss_combinations[alpha_index_times[i][3]]
                                                                    * val2 * val0;
        }
        // New code

        // Step 4.3. Loss derivative w.r.t. coeffs
        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            NeighbVect[0] = relative_coords[ii*umax_num_neigh_atoms + jj][0];
            NeighbVect[1] = relative_coords[ii*umax_num_neigh_atoms + jj][1];
            NeighbVect[2] = relative_coords[ii*umax_num_neigh_atoms + jj][2];
            distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                     + std::pow(NeighbVect[1], 2)
                                     + std::pow(NeighbVect[2], 2));
            if (distance_ij > rmax)
                continue;
            distance_ij_inv = 1.0 / distance_ij;
            p_RadialBasis->build(distance_ij);

            auto_dist_powers_[0] = 1.0;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[0][aa] = 1.0;
            for (int k=1; k<max_alpha_index_basic; k++) {
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
                    CoordType A = p_RadialBasis->vals()[xi];
                    CoordType B = mult0;
                    CoordType C = powk;
                    CoordType A_ders[3] = {0., 0., 0.};
                    CoordType B_ders[3] = {0., 0., 0.};
                    CoordType C_ders[3] = {0., 0., 0.};
                    A_ders[0] = p_RadialBasis->ders2r()[xi] * NeighbVect[0] * distance_ij_inv;
                    A_ders[1] = p_RadialBasis->ders2r()[xi] * NeighbVect[1] * distance_ij_inv;
                    A_ders[2] = p_RadialBasis->ders2r()[xi] * NeighbVect[2] * distance_ij_inv;
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


                    CoordType tmpe_loss_der2coeff = 0.0;
                    tmpe_loss_der2coeff = 2*e_weight/inum*(etot_ml - etot_dft)
                                          * e_site_der2mom[i]
                                          * A * B * C;
                    
                    CoordType tmpf_loss_der2coeff = de22m0m1_dloss_combinations[i] * A * B * C;
                    for (int aa=0; aa<3; aa++) {
                        CoordType tmp_prefix = 0.0;
                        CoordType tmp_deriv = (A_ders[aa] * B * C 
                                               + A * B_ders[aa] * C 
                                               + A * B * C_ders[aa]);
                        
                        tmp_prefix += 2*f_weight/(3*inum)
                                      * (force_ml[center_idx][aa] - force_dft[center_idx][aa]);
                        tmp_prefix -= 2*f_weight/(3*inum)
                                      * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa]);
                        tmpf_loss_der2coeff += tmp_prefix * e_site_der2mom[i] * tmp_deriv;
                    }

                    #ifdef USE_OPENMP
                    #pragma omp atomic
                    #endif 
                    loss_der2coeffs[idx] += (tmpe_loss_der2coeff + tmpf_loss_der2coeff);
                }
            }
        }

        // Step 4.4. Loss derivative w.r.t. linear_coeffs
        for (int i=0; i<alpha_scalar_moments; i++) {
            CoordType tmp_loss_der2linear_coeff = 2*e_weight/inum 
                                                  * (etot_ml - etot_dft) 
                                                  * (mom_vals[alpha_moment_mapping[i]] - q_shifter[i])
                                                  + dloss_combination[alpha_moment_mapping[i]];
            
            #ifdef USE_OPENMP
            #pragma omp atomic
            #endif
            loss_der2linear_coeffs[i] += tmp_loss_der2linear_coeff / q_scaler[i];
        }

        // Step 4.5. Loss derivative w.r.t. type_bias
        #ifdef USE_OPENMP
        #pragma omp atomic
        #endif
        loss_der2type_bias[type_central] += 2*e_weight/inum*(etot_ml - etot_dft);
    }

    // Step . Free
    free(mom_vals);
    free(e_site_der2mom);
    free(dloss_combination);
    free(de22m0m1_dloss_combinations);
    free(auto_dist_powers_);
    free(auto_coords_powers_);
    delete p_RadialBasis;
#ifdef USE_OPENMP
}
#endif
}


};  // namespace : mtpr
};  // namespace : ai2pot

#endif
