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

#ifndef AI2POT_NEP_NEP_LOSS_H
#define AI2POT_NEP_NEP_LOSS_H

#include <omp.h>
#include <stdlib.h>
#include <math.h>

#include "./basis.h"
#include "./nep_utilities.h"


namespace ai2pot {
namespace nep {


template <typename CoordType>
class NepLoss {
public:
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
    
    static void find_ef_loss_backward(
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
        int n_radial_basis,
        int n_angular_basis,
        int l_max,
        int num_neurons,
        CoordType *coeffs,
        CoordType *w0,
        CoordType *w1,
        CoordType *type_bias,
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
        CoordType *q_scaler);
};  // namespace : NepLoss


template <typename CoordType>
void NepLoss<CoordType>::find_ef_loss(
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
    f_loss = f_weight / (3*inum) * f_loss;

    // Energy term
    CoordType e_loss = 0.0;
    e_loss = e_weight / inum * std::pow(etot_ml - etot_dft, 2);

    loss = f_loss + e_loss;
}


template <typename CoordType>
void NepLoss<CoordType>::find_loss(
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
    CoordType f_loss = 0.0;
    for (int ii=0; ii<inum; ii++) {
        int center_idx = ilist[ii];
        for (int aa=0; aa<3; aa++) {
            f_loss += std::pow(force_ml[center_idx][aa] - force_dft[center_idx][aa], 2);
        }
    }
    f_loss = f_weight / (3*inum) * f_loss;

    // Virial term
    CoordType v_loss = 0.0;
    for (int aa=0; aa<3; aa++)
        for (int bb=0; bb<3; bb++)
            v_loss += std::pow(virial_ml[aa*3+bb] - virial_dft[aa*3+bb], 2);
    v_loss = v_weight / (9*inum) * v_loss;

    // Energy term
    CoordType e_loss = 0;
    e_loss = e_weight / inum * std::pow(etot_ml - etot_dft, 2);

    loss = f_loss + v_loss + e_loss;
}


template <typename CoordType>
void NepLoss<CoordType>::find_ef_loss_backward(
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
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
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
    CoordType *q_scaler)
{
    // Step 1. 
    CoordType *mom_vals;
    CoordType *dod_vals;
    CoordType *e_sites_der2mom;
    CoordType *e_sites_der2dod;
    CoordType *dloss_combination_mom;
    CoordType *dloss_combination_dod;
    int num_coeffs = ntypes * ntypes * (n_radial_basis + n_angular_basis) * chebyshev_size;
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_Sinlm = NepIndex::get_num_Sinlm(n_angular_basis, l_max);

    // Step 2. Calculate descriptors
    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp parallel private(mom_vals, dod_vals, e_sites_der2mom, e_sites_der2dod, dloss_combination_mom, dloss_combination_dod)
    {
    #endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * num_Sinlm);
    dod_vals = (CoordType*)malloc(sizeof(CoordType) * num_descriptors);
    e_sites_der2mom = (CoordType*)malloc(sizeof(CoordType) * num_Sinlm);
    e_sites_der2dod = (CoordType*)malloc(sizeof(CoordType) * num_descriptors);
    dloss_combination_mom = (CoordType*)malloc(sizeof(CoordType) * num_Sinlm);
    dloss_combination_dod = (CoordType*)malloc(sizeof(CoordType) * num_descriptors);
    
    int center_idx;
    int type_central;
    CoordType *type_central_w0;
    CoordType *type_central_w1;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    RQ_Chebyshev<CoordType> *p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);
    CoordType *auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * (l_max+1));

    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++) {
        // 
        memset(mom_vals, 0, sizeof(CoordType) * num_Sinlm);
        memset(dod_vals, 0, sizeof(CoordType) * num_descriptors);
        memset(e_sites_der2mom, 0, sizeof(CoordType) * num_Sinlm);
        memset(e_sites_der2dod, 0, sizeof(CoordType) * num_descriptors);
        memset(dloss_combination_mom, 0, sizeof(CoordType) * num_Sinlm);
        memset(dloss_combination_dod, 0, sizeof(CoordType) * num_descriptors);

        center_idx = ilist[ii];
        type_central = types[center_idx];
        type_central_w0 = &w0[type_central*num_neurons*num_descriptors];
        type_central_w1 = &w1[type_central*num_neurons];


        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            for (int aa=0; aa<3; aa++)
                neigh_vec[aa] = rcs[ii*umax_num_neigh_atoms+jj][aa];
            distance_ij = std::sqrt(std::pow(neigh_vec[0], 2)
                                    + std::pow(neigh_vec[1], 2)
                                    + std::pow(neigh_vec[2], 2));
            if (distance_ij > rmax)
                continue;
            p_RadialBasis->build(distance_ij);
            auto_dist_powers_[0] = 1.0;
            for (int k=1; k<=l_max; k++)
                auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            

            // Step 2.1. Radial forward
            for (int mu=0; mu<n_radial_basis; mu++) {
                for (int xi=0; xi<chebyshev_size; xi++) {
                    int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                              + mu*chebyshev_size + xi;
                    dod_vals[mu] += coeffs[idx] * p_RadialBasis->vals()[xi];
                    for (int aa=0; aa<3; aa++) {
                        dloss_combination_dod[mu] += 2*f_weight/(3*inum)
                                                     * (force_ml[center_idx][aa] - force_dft[center_idx][aa])
                                                     * coeffs[idx] * p_RadialBasis->ders2r()[xi] * neigh_vec[aa] / distance_ij;
                        dloss_combination_dod[mu] -= 2*f_weight/(3*inum)
                                                     * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa])
                                                     * coeffs[idx] * p_RadialBasis->ders2r()[xi] * neigh_vec[aa] / distance_ij;
                    }
                }
            }

            // Step 2.2. Angular forward: basic
            for (int mu=0; mu<n_angular_basis; mu++) {
                for (int l=1; l<=l_max; l++) {
                    for (int mp=0; mp<2*l+1; mp++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                      + (mu+n_radial_basis)*chebyshev_size + xi;
                            int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);

                            CoordType A = p_RadialBasis->vals()[xi];
                            CoordType B = 0.0;
                            Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                            CoordType C = 1/auto_dist_powers_[l];
                            CoordType A_ders[3] = {0.0};
                            CoordType B_ders[3] = {0.0};
                            CoordType C_ders[3] = {0.0};
                            Blm<CoordType>::find_blm_der2xyz(B_ders, l, mp, neigh_vec, distance_ij);
                            for (int aa=0; aa<3; aa++) {
                                A_ders[aa] = p_RadialBasis->ders2r()[xi] * neigh_vec[aa] / distance_ij;
                                C_ders[aa] = -l / (auto_dist_powers_[l] * distance_ij) * (neigh_vec[aa] / distance_ij);
                            }
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

        // Step 2.3. Angular forward: times
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    int idx_Clm = NepIndex::get_Clm_index(l, mp);
                    int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);
                    dod_vals[n_radial_basis+idx_qinl] += (CoordType)C3B[idx_Clm] * std::pow(mom_vals[idx_Sinlm], 2);

                    dloss_combination_dod[n_radial_basis + idx_qinl] += dloss_combination_mom[idx_Sinlm]
                                                                        * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
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

                    e_sites_der2mom[idx_Sinlm] = e_sites_der2dod[n_radial_basis + idx_qinl]
                                                 * 2 * (CoordType)C3B[idx_Clm] * mom_vals[idx_Sinlm];
                }
            }
        }

        // Step 4.1 loss_der2coeffs
        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms+jj];
            type_outer = types[neigh_idx];
            for (int aa=0; aa<3; aa++)
                neigh_vec[aa] = rcs[ii*umax_num_neigh_atoms + jj][aa];
            distance_ij = std::sqrt(std::pow(neigh_vec[0], 2)
                                    + std::pow(neigh_vec[1], 2)
                                    + std::pow(neigh_vec[2], 2));
            if (distance_ij > rmax)
                continue;
            p_RadialBasis->build(distance_ij);
            auto_dist_powers_[0] = 1.0;
            for (int k=1; k<=l_max; k++)
                auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            
            // 4.1.1. loss_der2coeffs -- Radial
            for (int mu=0; mu<n_radial_basis; mu++) {
                for (int xi=0; xi<chebyshev_size; xi++) {
                    int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                              + mu*chebyshev_size + xi;
                    
                    CoordType tmpe_loss_der2coeff = 0.0;
                    tmpe_loss_der2coeff = 2*e_weight/inum*(etot_ml - etot_dft)
                                          * e_sites_der2dod[mu]
                                          * p_RadialBasis->vals()[xi];
                    
                    CoordType tmpf_loss_der2coeff = 0.0;
                    for (int aa=0; aa<3; aa++) {
                        CoordType tmp_prefix = 0.0;
                        CoordType tmp_deriv = p_RadialBasis->ders2r()[xi] * neigh_vec[aa] / distance_ij;

                        tmp_prefix += 2*f_weight/(3*inum)
                                      * (force_ml[center_idx][aa] - force_dft[center_idx][aa]);
                        tmp_prefix -= 2*f_weight/(3*inum)
                                      * (force_ml[neigh_idx][aa] - force_dft[neigh_idx][aa]);
                        tmpf_loss_der2coeff += tmp_prefix * tmp_deriv;
                    }
                    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                    #pragma omp atomic
                    #endif
                    loss_der2coeffs[idx] += (tmpe_loss_der2coeff
                                             + tmpf_loss_der2coeff * e_sites_der2dod[mu]);
                }
            }

            // 4.1.2. loss_der2coeffs -- Angular
            for (int mu=0; mu<n_angular_basis; mu++) {
                for (int l=1; l<=l_max; l++) {
                    for (int mp=0; mp<2*l+1; mp++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                      + (mu+n_radial_basis)*chebyshev_size + xi;
                            int idx_Clm = NepIndex::get_Clm_index(l, mp);
                            int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                            int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);

                            CoordType A = p_RadialBasis->vals()[xi];
                            CoordType B = 0.0;
                            Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                            CoordType C = 1/auto_dist_powers_[l];

                            CoordType A_ders[3] = {0.0};
                            CoordType B_ders[3] = {0.0};
                            CoordType C_ders[3] = {0.0};
                            for (int aa=0; aa<3; aa++) {
                                A_ders[aa] = p_RadialBasis->ders2r()[xi] * neigh_vec[aa] / distance_ij;
                                C_ders[aa] = -l / (auto_dist_powers_[l]*distance_ij)
                                             * (neigh_vec[aa] / distance_ij);
                            }
                            Blm<CoordType>::find_blm_der2xyz(B_ders, l, mp, neigh_vec, distance_ij);

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
                            #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                            #pragma omp atomic
                            #endif
                            loss_der2coeffs[idx] += (tmpe_loss_der2coeff
                                                     + tmpf_loss_der2coeff * e_sites_der2mom[idx_Sinlm]);
                        }
                    }
                }
            }
        }

        // Step 4.2. der2w0
        for (int p=0; p<num_neurons; p++) {
            CoordType hidden_val = 0.0;
            CoordType activated_hidden_der = 0.0;
            CoordType activated_hidden_der2der = 0.0;
            for (int k=0; k<num_descriptors; k++)
                hidden_val += type_central_w0[p*num_descriptors+k] * dod_vals[k] / q_scaler[k];
            TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
            TanhActivationFunc<CoordType>::find_der2der(activated_hidden_der2der, hidden_val);
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
                
                #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                #pragma omp atomic
                #endif
                loss_der2w0[type_central*num_neurons*num_descriptors + p*num_descriptors + k] += tmpe_loss_der2w0
                                                                                                 + tmpf_loss_der2w0;
            }
        }

        // Step 4.3. der2w1
        for (int p=0; p<num_neurons; p++) {
            CoordType tmpe_loss_der2w1 = 0.0;
            CoordType tmpf_loss_der2w1 = 0.0;
            CoordType hidden_val = 0.0;
            CoordType activated_hidden_val = 0.0;
            CoordType activated_hidden_der = 0.0;
            for (int k=0; k<num_descriptors; k++)
                hidden_val += type_central_w0[p*num_descriptors+k] * dod_vals[k] / q_scaler[k];
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

            #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
            #pragma omp atomic
            #endif
            loss_der2w1[type_central*num_neurons + p] += (tmpe_loss_der2w1
                                                          + tmpf_loss_der2w1);
        }

        // Step 4.4. der2type_bias
        #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
        #pragma omp atomic
        #endif
        loss_der2type_bias[type_central] += 2*e_weight/inum*(etot_ml-etot_dft);
    }

    // Step . Free
    free(mom_vals);
    free(dod_vals);
    free(e_sites_der2mom);
    free(e_sites_der2dod);
    free(dloss_combination_mom);
    free(dloss_combination_dod);
    delete p_RadialBasis;
    free(auto_dist_powers_);
    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    }
    #endif
}


};  // namespace: nep
};  // namespace : ai2pot


#endif