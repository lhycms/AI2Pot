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

#ifndef AI2POT_NEP_NEP_BASIS_H
#define AI2POT_NEP_NEP_BASIS_H

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "./basis.h"
#include "./nep_utilities.h"


namespace ai2pot {
namespace nep {

template <typename CoordType>
class MomsDodsValDer {
public:
    static void find_val_der(
        CoordType *mom_vals,
        CoordType *dod_vals,
        CoordType (*mom_ders)[3],
        CoordType (*dod_ders)[3],
        CoordType chebyshev_size,
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
        int umax_num_neigh_atoms,
        CoordType rmax_radial,
        CoordType rmax_angular);
};  // class : NepBasis


template <typename CoordType>
void MomsDodsValDer<CoordType>::find_val_der(
    CoordType *mom_vals,
    CoordType *dod_vals,
    CoordType (*mom_ders)[3],
    CoordType (*dod_ders)[3],
    CoordType chebyshev_size,
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
    int umax_num_neigh_atoms,
    CoordType rmax_radial,
    CoordType rmax_angular)
{
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_Sinlm = NepIndex::get_num_Sinlm(n_angular_basis, l_max);
    memset(mom_vals, 0, sizeof(CoordType) * num_Sinlm);
    memset(dod_vals, 0, sizeof(CoordType) * num_descriptors);
    memset(mom_ders, 0, sizeof(CoordType) * num_Sinlm * umax_num_neigh_atoms * 3);
    memset(dod_ders, 0, sizeof(CoordType) * n_radial_basis * umax_num_neigh_atoms * 3);
    int num_coeffs = ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size;

    // Step 1.
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    int type_central;
    int type_outer;

    CoordType *auto_dist_powers_;
    auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * (l_max+1));
    RQ_Chebyshev<CoordType> *p_RadialBasis_radial = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax_radial);
    RQ_Chebyshev<CoordType> *p_RadialBasis_angular = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax_angular);

    // Step 2.
    type_central = types[silist];
    for (int jj=0; jj<snumneigh; jj++) {
        type_outer = types[sfirstneigh[jj]];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( neigh_vec[0]*neigh_vec[0]
                                 + neigh_vec[1]*neigh_vec[1]
                                 + neigh_vec[2]*neigh_vec[2] );
        if (distance_ij > rmax_radial)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        p_RadialBasis_radial->build(distance_ij);
        p_RadialBasis_angular->build(distance_ij);

        auto_dist_powers_[0] = 1.0;
        for (int k=1; k<=l_max; k++)
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
        
        // Step 2.1. Radial forward
        if (distance_ij <= rmax_radial)
        for (int mu=0; mu<n_radial_basis; mu++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                          + mu*chebyshev_size + xi;
                dod_vals[mu] += coeffs[idx] * p_RadialBasis_radial->vals()[xi];

                CoordType A_ders[3] = {0.0};
                A_ders[0] = p_RadialBasis_radial->ders2r()[xi] * neigh_vec[0] * distance_ij_inv;
                A_ders[1] = p_RadialBasis_radial->ders2r()[xi] * neigh_vec[1] * distance_ij_inv;
                A_ders[2] = p_RadialBasis_radial->ders2r()[xi] * neigh_vec[2] * distance_ij_inv;

                for (int aa=0; aa<3; aa++)
                    dod_ders[mu*umax_num_neigh_atoms + jj][aa] += coeffs[idx] * A_ders[aa];
            }
        }

        // Step 2.2. Angular forward: basic
        if (distance_ij <= rmax_angular)
        for (int mu=0; mu<n_angular_basis; mu++) {
            for (int l=1; l<=l_max; l++) {
                CoordType C_ders[3] = {0.0};
                CoordType tmp_prefix_C_der = -l / (auto_dist_powers_[l]*distance_ij)
                                             * distance_ij_inv;
                C_ders[0] = tmp_prefix_C_der * neigh_vec[0];
                C_ders[1] = tmp_prefix_C_der * neigh_vec[1];
                C_ders[2] = tmp_prefix_C_der * neigh_vec[2];

                for (int mp=0; mp<2*l+1; mp++) {
                    int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                    CoordType B = 0.0;
                    Blm<CoordType>::find_blm_val(&B, l, mp, neigh_vec, distance_ij);
                    CoordType C = 1/auto_dist_powers_[l];
                    CoordType B_ders[3] = {0.0};
                    Blm<CoordType>::find_blm_der2xyz(B_ders, l, mp, neigh_vec, distance_ij);

                    for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (type_central*ntypes+type_outer)*(n_radial_basis+n_angular_basis)*chebyshev_size
                                  + (n_radial_basis+mu)*chebyshev_size + xi;

                        CoordType A = p_RadialBasis_angular->vals()[xi];
                        CoordType A_ders[3] = {0.0};
                        A_ders[0] = p_RadialBasis_angular->ders2r()[xi] * neigh_vec[0] * distance_ij_inv;
                        A_ders[1] = p_RadialBasis_angular->ders2r()[xi] * neigh_vec[1] * distance_ij_inv;
                        A_ders[2] = p_RadialBasis_angular->ders2r()[xi] * neigh_vec[2] * distance_ij_inv;
                        mom_vals[idx_Sinlm] += coeffs[idx] * A * B * C;

                        for (int aa=0; aa<3; aa++)
                            mom_ders[idx_Sinlm*umax_num_neigh_atoms + jj][aa] += coeffs[idx]
                                            * (A_ders[aa]*B*C + A*B_ders[aa]*C + A*B*C_ders[aa]);
                    }
                }
            }
        }
    }

    // Step 2.3. Angular Forward: times
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            int idx_qinl = NepIndex::get_qinl_index(l_max, mu, l);

            for (int mp=0; mp<2*l+1; mp++) {
                int idx_Sinlm = NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                int idx_Clm = NepIndex::get_Clm_index(l, mp);
                if (mp == 0)
                    dod_vals[n_radial_basis+idx_qinl] += (CoordType)C3B[idx_Clm]
                                                         * mom_vals[idx_Sinlm]
                                                         * mom_vals[idx_Sinlm];
                else
                    dod_vals[n_radial_basis+idx_qinl] += 2 * (CoordType)C3B[idx_Clm]
                                                         * mom_vals[idx_Sinlm]
                                                         * mom_vals[idx_Sinlm];
            }
        }
    }

    free(auto_dist_powers_);
    delete p_RadialBasis_radial;
    delete p_RadialBasis_angular;
}

};  // namespace : nep
};  // namespace : ai2pot


#endif
