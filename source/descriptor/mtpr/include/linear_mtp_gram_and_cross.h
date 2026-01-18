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
    static void find_structure_efv_components(
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

    static void accumulate_lin_matrix_lin_vector(
        CoordType *lin_matrix,
        CoordType *lin_vector,
        CoordType e_weight,
        CoordType f_weight,
        CoordType v_weight,
        CoordType *betot_dft,
        CoordType (*bforce_dft)[3],
        CoordType *bvirial_dft,
        CoordType *energy_components,
        CoordType *force_components,
        CoordType *virial_components,
        int alpha_scalar_moments,
        int nmus,
        int batch_size,
        int natoms_pad,
        int *binum,
        int *bilist,
        int ntypes);
    
    static void find_lin_matrix_lin_vector_launcher(
        CoordType *lin_matrix,
        CoordType *lin_vector,
        CoordType e_weight,
        CoordType f_weight,
        CoordType v_weight,
        CoordType *betot_dft,
        CoordType (*bforce_dft)[3],
        CoordType *bvirial_dft,
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
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin);
};  // class : LinearMtpGramAndCross


template <typename CoordType>
void LinearMtpGramAndCross<CoordType>::find_structure_efv_components(
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
                    force_components[center_idx*3*num_parameters + aa*num_parameters + k] += moms_der2xyz[alpha_moment_mapping[k]];
                    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                    #pragma omp atomic
                    #endif
                    force_components[neigh_idx*3*num_parameters + aa*num_parameters + k] -= moms_der2xyz[alpha_moment_mapping[k]];
                }

                // virial_components
                for (int bb=0; bb<3; bb++) {
                    for (int k=0; k<alpha_scalar_moments; k++) {
                        #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                        #pragma omp atomic
                        #endif
                        virial_components[(aa*3+bb)*num_parameters + k] -= moms_der2xyz[alpha_moment_mapping[k]] * neigh_vec[bb];
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


template <typename CoordType>
void LinearMtpGramAndCross<CoordType>::accumulate_lin_matrix_lin_vector(
    CoordType *lin_matrix,
    CoordType *lin_vector,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_dft,
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_dft,
    CoordType *energy_components,
    CoordType *force_components,
    CoordType *virial_components,
    int alpha_scalar_moments,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int ntypes)
{
    int num_parameters = alpha_scalar_moments + ntypes;

    for (int bb=0; bb<batch_size; bb++) {
        // 1. energy term
        for (int k1=0; k1<num_parameters; k1++) {
            for (int k2=k1; k2<num_parameters; k2++) {
                lin_matrix[k1*num_parameters + k2] += e_weight 
                                                      * energy_components[k1]
                                                      * energy_components[k2];
            }
            lin_vector[k1] += e_weight * betot_dft[bb] * energy_components[k1];
        }

        // 2. force term
        for (int ii=0; ii<binum[bb]; ii++) {
            int center_idx = bilist[bb*natoms_pad + ii];
            for (int aa=0; aa<3; aa++) {
                for (int k1=0; k1<num_parameters; k1++) {
                    for (int k2=k1; k2<num_parameters; k2++) {
                        lin_matrix[k1*num_parameters + k2] += f_weight 
                                                              * force_components[(center_idx*3+aa)*num_parameters + k1]
                                                              * force_components[(center_idx*3+aa)*num_parameters + k2];
                    }
                    lin_vector[k1] += f_weight 
                                      * bforce_dft[bb*natoms_pad + center_idx][aa]
                                      * force_components[(center_idx*3+aa)*num_parameters + k1];
                }
            }
        }

        // 3. virial term
        for (int alpha=0; alpha<3; alpha++) {
            for (int beta=0; beta<3; beta++) {
                for (int k1=0; k1<num_parameters; k1++) {
                    for (int k2=k1; k2<num_parameters; k2++) {
                        lin_matrix[k1*num_parameters + k2] += v_weight
                                                              * virial_components[(alpha*3+beta)*num_parameters + k1]
                                                              * virial_components[(alpha*3+beta)*num_parameters + k2];
                    }
                    lin_vector[k1] += v_weight
                                      * bvirial_dft[bb*3*3 + alpha*3 + beta]
                                      * virial_components[(alpha*3+beta)*num_parameters + k1];
                }
            }
        }
    }

    for (int k1=0; k1<num_parameters; k1++)
        for (int k2=k1+1; k2<num_parameters; k2++)
            lin_matrix[k2*num_parameters + k1] = lin_matrix[k1*num_parameters + k2];
}



template <typename CoordType>
void LinearMtpGramAndCross<CoordType>::find_lin_matrix_lin_vector_launcher(
    CoordType *lin_matrix,
    CoordType *lin_vector,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_dft,
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_dft,
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
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    // Step . 
    int num_parameters = alpha_scalar_moments + ntypes;
    memset(lin_matrix, 0, sizeof(CoordType)*num_parameters*num_parameters);
    memset(lin_vector, 0, sizeof(CoordType)*num_parameters);
    
    CoordType *energy_components;
    CoordType *force_components;
    CoordType *virial_components;
    energy_components = (CoordType*)malloc(sizeof(CoordType)*num_parameters);
    force_components = (CoordType*)malloc(sizeof(CoordType)*natoms_pad*3*num_parameters);
    virial_components = (CoordType*)malloc(sizeof(CoordType)*3*3*num_parameters);

    // Step 1. 
    for (int bb=0; bb<batch_size; bb++) {
        memset(energy_components, 0, sizeof(CoordType)*num_parameters);
        memset(force_components, 0, sizeof(CoordType)*natoms_pad*3*num_parameters);
        memset(virial_components, 0, sizeof(CoordType)*3*3*num_parameters);

        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb*natoms_pad];

        find_structure_efv_components(
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
            natoms_pad,
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            ntypes,
            type_map,
            umax_num_neigh_atoms,
            nghost,
            rmax,
            rmin);

        // Step 2.
        accumulate_lin_matrix_lin_vector(
            lin_matrix,
            lin_vector,
            e_weight,
            f_weight,
            v_weight,
            betot_dft,
            bforce_dft,
            bvirial_dft,
            energy_components,
            force_components,
            virial_components,
            alpha_scalar_moments,
            nmus,
            batch_size,
            natoms_pad,
            binum,
            bilist,
            ntypes);
    }

    // Step . Free
    free(energy_components);
    free(force_components);
    free(virial_components);
}


};  // namespace : mtpr
};  // namespace : ai2pot

#endif
