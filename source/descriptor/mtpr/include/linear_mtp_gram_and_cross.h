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

#ifndef AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_H
#define AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "./basis.h"

namespace ai2pot {
namespace mtpr {


template <typename CoordType>
class LinearMtpGramAndCross {
public:
    static void accumulate_structure_efv_components(
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
        int natoms_pad,
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
};  // class : LinearMtpGramAndCross


template <typename CoordType>
void LinearMtpGramAndCross<CoordType>::accumulate_structure_efv_components(
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
    int natoms_pad,
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
    // Step 1.
    int num_parameters = alpha_scalar_moments + ntypes;
    memset(energy_components, 0, sizeof(CoordType)*num_parameters);
    memset(force_components, 0, sizeof(CoordType)*natoms_pad*3*num_parameters);
    memset(virial_components, 0, sizeof(CoordType)*3*3*num_parameters);
    CoordType *mom_vals;
    CoordType *moms_der2xyz;
    
    int max_alpha_index_basic = 0;
    for (int ii=0; ii<alpha_index_basic_count; ii++) {
        int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][3];
        if (now_alpha_index_basic > max_alpha_index_basic)
            max_alpha_index_basic = now_alpha_index_basic;
    }
    max_alpha_index_basic++;

    // Step 2.
    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp parallel private(mom_vals, moms_der2xyz)
    {
    #endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    moms_der2xyz = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    CoordType *auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic);
    CoordType (*auto_coords_powers_)[3] = (CoordType (*)[3])malloc(sizeof(CoordType) * max_alpha_index_basic * 3);
    RQ_Chebyshev<CoordType> *p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);


    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++) {
        memset(mom_vals, 0, sizeof(CoordType)*alpha_moments_count);
        center_idx = ilist[ii];
        type_central = types[center_idx];

        // Step 2.1. energy_components
        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            for (int aa=0; aa<3; aa++)
                neigh_vec[aa] = rcs[ii*umax_num_neigh_atoms + jj][aa];
            distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                     + std::pow(neigh_vec[1], 2)
                                     + std::pow(neigh_vec[2], 2) );
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
                    CoordType A = p_RadialBasis->vals()[xi];
                    CoordType B = mult0;
                    CoordType C = powk;

                    mom_vals[i] += coeffs[idx] * A * B * C;
                }
            }
        }

        for (int i=0; i<alpha_index_times_count; i++) {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];
            mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
        }

        // energy_components
        #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
        #pragma omp atomic
        #endif
        energy_components[alpha_scalar_moments+type_central] += 1.0;
        for (int k=0; k<alpha_scalar_moments; k++) {
            #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
            #pragma omp atomic
            #endif
            energy_components[k] += mom_vals[alpha_moment_mapping[k]];
        }


        // 2.2. force_components && virial_components
        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            for (int aa=0; aa<3; aa++)
                neigh_vec[aa] = rcs[ii*umax_num_neigh_atoms + jj][aa];
            distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                     + std::pow(neigh_vec[1], 2)
                                     + std::pow(neigh_vec[2], 2) );
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
                    auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * neigh_vec[aa];
            }

            for (int aa=0; aa<3; aa++) {
                memset(moms_der2xyz, 0, sizeof(CoordType)*alpha_moments_count);
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
                        CoordType A = p_RadialBasis->vals()[xi];
                        CoordType B = mult0;
                        CoordType C = powk;
                        CoordType A_der = p_RadialBasis->ders2r()[xi] * neigh_vec[aa] * distance_ij_inv;
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
                    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                    #pragma omp atomic
                    #endif
                    force_components[center_idx*3*alpha_scalar_moments + aa*alpha_scalar_moments + k] += moms_der2xyz[alpha_moment_mapping[k]];
                    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                    #pragma omp atomic
                    #endif
                    force_components[neigh_idx*3*alpha_scalar_moments + aa*alpha_scalar_moments + k] -= moms_der2xyz[alpha_moment_mapping[k]];
                }

                // virial_components
                for (int bb=0; bb<3; bb++) {
                    for (int k=0; k<alpha_scalar_moments; k++) {
                        #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                        #pragma omp atomic
                        #endif
                        virial_components[(aa*3+bb)*alpha_scalar_moments + k] -= moms_der2xyz[alpha_moment_mapping[k]] * neigh_vec[bb];
                    }
                }
            }
        }
    }

    // Step . Free
    free(mom_vals);
    free(moms_der2xyz);
    free(auto_dist_powers_);
    free(auto_coords_powers_);
    delete p_RadialBasis;

    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    }
    #endif
}

};  // namespace : mtpr
};  // namespace : ai2pot

#endif
