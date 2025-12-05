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

#ifndef AI2POT_NEP_NEP_H
#define AI2POT_NEP_NEP_H

#include <omp.h>
#include <stdlib.h>

#include "./basis.h"
#include "./nep_utilities.h"


namespace ai2pot {
namespace nep {


template <typename CoordType>
class Nep {
public:
    static void find_ef(
        CoordType &etot,
        CoordType (*force)[3],
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

};  // class : Nep


template <typename CoordType>
void Nep<CoordType>::find_ef(
    CoordType &etot,
    CoordType (*force)[3],
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
    int num_coeffs = ntypes * ntypes * (n_radial_basis + n_angular_basis) * chebyshev_size;
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_Sinlm = NepIndex::get_num_Sinlm(n_angular_basis, l_max);

    // Step 2. 
    etot = 0;
    memset(force, 0, sizeof(CoordType) * (inum+nghost) * 3);

    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp parallel private(mom_vals, dod_vals, e_sites_der2mom, e_sites_der2dod)
    {
    #endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * num_Sinlm);
    dod_vals = (CoordType*)malloc(sizeof(CoordType) * num_descriptors);
    e_sites_der2mom = (CoordType*)malloc(sizeof(CoordType) * num_Sinlm);
    e_sites_der2dod = (CoordType*)malloc(sizeof(CoordType) * num_descriptors);
    CoordType *auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * (l_max+1));

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    RQ_Chebyshev<CoordType> *p_RadialBasis;
    p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);

    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++)
    {
        center_idx = ilist[ii];
        type_central = types[center_idx];
        CoordType *type_central_w0 = &w0[type_central*num_neurons*num_descriptors];
        CoordType *type_central_w1 = &w1[type_central*num_neurons];
        memset(mom_vals, 0, sizeof(CoordType) * num_Sinlm);
        memset(dod_vals, 0, sizeof(CoordType) * num_descriptors);
        memset(e_sites_der2mom, 0, sizeof(CoordType) * num_Sinlm);
        memset(e_sites_der2dod, 0, sizeof(CoordType) * num_descriptors);
        CoordType e_site = 0;

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
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
            
            // Step 2.1. Radial forward
            for (int mu=0; mu<n_radial_basis; mu++) {
                for (int xi=0; xi<chebyshev_size; xi++) {
                    int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                              + mu*chebyshev_size + xi;
                    dod_vals[mu] += coeffs[idx] * p_RadialBasis->vals()[xi];
                }
            }

            // Step 2.2. Angular forward: basic
            for (int mu=0; mu<n_angular_basis; mu++) {
                for (int l=1; l<=l_max; l++) {
                    for (int mp=0; mp<2*l+1; mp++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                      + (n_radial_basis+mu)*chebyshev_size + xi;
                            int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);

                            CoordType A = p_RadialBasis->vals()[xi];
                            CoordType B = 0.0;
                            Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                            CoordType C = 1/auto_dist_powers_[l];
                            mom_vals[idx_Sinlm] += coeffs[idx] * A * B * C;
                        }
                    }
                }
            }
        }

        // Step 2.3. A forward: times
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    int idx_Clm = NepIndex::get_Clm_index(l, mp);
                    dod_vals[n_radial_basis+idx_qinl] += (CoordType)C3B[idx_Clm] * std::pow(mom_vals[idx_Sinlm], 2);
                }
            }
        }

        // Step 2.4. Ei
        e_site = type_bias[type_central];
        for (int p=0; p<num_neurons; p++) {
            CoordType hidden_val = 0.0;
            CoordType activated_hidden_val = 0.0;
            for (int k=0; k<num_descriptors; k++) {
                hidden_val += type_central_w0[p*num_descriptors + k] * dod_vals[k];
            }
            TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
            e_site += type_central_w1[p] * activated_hidden_val;
        }
        #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
        #pragma omp atomic
        #endif
        etot += e_site;

        // 3. backward (Force)
        // 3.1. RA basic
        for (int p=0; p<num_neurons; p++) {
            CoordType hidden_val = 0.0;
            CoordType activated_hidden_der = 0.0;
            for (int k=0; k<num_descriptors; k++)
                hidden_val += type_central_w0[p*num_descriptors + k] * dod_vals[k];
            TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
            
            for (int k=0; k<num_descriptors; k++)
                e_sites_der2dod[k] += type_central_w1[p] 
                                      * activated_hidden_der 
                                      * type_central_w0[p*num_descriptors + k];
        }

        // 3.2. Angular times
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

        // 3.3. 
        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            neigh_vec[0] = rcs[ii*umax_num_neigh_atoms + jj][0];
            neigh_vec[1] = rcs[ii*umax_num_neigh_atoms + jj][1];
            neigh_vec[2] = rcs[ii*umax_num_neigh_atoms + jj][2];
            distance_ij = std::sqrt(std::pow(neigh_vec[0], 2)
                                    + std::pow(neigh_vec[1], 2)
                                    + std::pow(neigh_vec[2], 2));
            if (distance_ij > rmax)
                continue;
            p_RadialBasis->build(distance_ij);
            
            auto_dist_powers_[0] = 1.0;
            for (int k=1; k<=l_max; k++)
                auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;

            for (int aa=0; aa<3; aa++) {
                CoordType e_site_ders_ija = 0.0;
                // 3.3.1. Raidal contribution
                for (int mu=0; mu<n_radial_basis; mu++) {
                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                  + mu*chebyshev_size + xi;
        
                        e_site_ders_ija += e_sites_der2dod[mu]
                                           * coeffs[idx]
                                           * p_RadialBasis->ders2r()[xi] 
                                           * neigh_vec[aa] / distance_ij;
                    }
                }

                // 3.3.2. Angular contribution
                for (int mu=0; mu<n_angular_basis; mu++) {
                    for (int l=1; l<=l_max; l++) {
                        for (int mp=0; mp<2*l+1; mp++) {
                            int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                            for (int xi=0; xi<chebyshev_size; xi++) {
                                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                          + (n_radial_basis+mu)*chebyshev_size + xi;
                                
                                CoordType A = p_RadialBasis->vals()[xi];
                                CoordType B = 0.0;
                                Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                                CoordType C = 1.0/auto_dist_powers_[l];
                                
                                CoordType A_ders[3] = {0.0};
                                CoordType B_ders[3] = {0.0};
                                CoordType C_ders[3] = {0.0};
                                A_ders[aa] = p_RadialBasis->ders2r()[xi] * neigh_vec[aa] / distance_ij;
                                Blm<CoordType>::find_blm_der2xyz(B_ders, l, mp, neigh_vec, distance_ij);
                                C_ders[aa] = -l / (auto_dist_powers_[l]*distance_ij) 
                                             * (neigh_vec[aa] / distance_ij);
                                
                                e_site_ders_ija += e_sites_der2mom[idx_Sinlm]
                                                   * coeffs[idx] *
                                                   (A_ders[aa] * B * C
                                                    + A * B_ders[aa] * C
                                                    + A * B * C_ders[aa]);
                            }
                        }
                    }
                }

                #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                #pragma omp atomic
                #endif
                force[center_idx][aa] += e_site_ders_ija;
                #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                #pragma omp atomic
                #endif
                force[neigh_idx][aa] -= e_site_ders_ija;
            }
        }
    }

    // Step . Free
    free(mom_vals);
    free(dod_vals);
    free(e_sites_der2mom);
    free(e_sites_der2dod);
    free(auto_dist_powers_);
    delete p_RadialBasis;
    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    }
    #endif
}


};  // namespace : nep
};  // namespace : ai2pot


#endif
