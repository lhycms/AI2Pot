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

#ifndef AI2POT_MTP_BASIS_GRAD_H
#define AI2POT_MTP_BASIS_GRAD_H
#include <stdlib.h>
#include <cstring>
#include <cassert>

#include "./basis.h"


namespace ai2pot {
namespace mtpr {


template <typename CoordType>
class MtpBasisGrad {
public:
    static void find_val_der(
        CoordType (*mbg_val)[3],
        CoordType *mbg_der2coeffs,
        bool calculate_der,
        int chebyshev_size,
        CoordType *coeffs,
        const int alpha_moments_count,
        const int alpha_index_basic_count,
        const int (*alpha_index_basic)[4],
        const int alpha_index_times_count,
        const int (*alpha_index_times)[4],
        const int alpha_scalar_moments,
        const int *alpha_moment_mapping,
        const int max_num_mus4mom,
        const int *num_mus4moms,
        const int *mus4moms_ptr,
        int nmus,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        CoordType (*relative_coords)[3],
        int *types,
        int ntypes,
        int umax_num_neigh_atoms,
        CoordType rmax,
        CoordType rmin);

    static void find_der_backward();
};


template <typename CoordType>
void MtpBasisGrad<CoordType>::find_val_der(
    CoordType (*mbg_val)[3],
    CoordType *mbg_der2coeffs,
    bool calculate_der,
    int chebyshev_size,
    CoordType *coeffs,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    const int max_num_mus4mom,
    const int *num_mus4moms,
    const int *mus4moms_ptr,
    int nmus,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*relative_coords)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    CoordType rmax,
    CoordType rmin)
{
    // Step 1. 
    //memset(mbg_val, 0, sizeof(CoordType) * inum * alpha_moments_count * umax_num_neigh_atoms * 3);
    //memset(mbg_der2coeffs, 0, sizeof(CoordType) * inum * alpha_moments_count * umax_num_neigh_atoms * 3 * ntypes * ntypes * nmus * chebyshev_size);

    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    CoordType *mom_der2coeffs;
    CoordType *mom_ders_der2coeffs;
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);
    mom_der2coeffs = (CoordType *)malloc(sizeof(CoordType) * alpha_moments_count * ntypes * ntypes * nmus * chebyshev_size);
    mom_ders_der2coeffs = (CoordType *)malloc(sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3 * ntypes * ntypes * nmus * chebyshev_size);
    //memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
    //memset(mom_ders, 0, sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);
    //memset(mom_der2coeffs, 0, sizeof(CoordType) * alpha_moments_count * ntypes * ntypes * nmus * chebyshev_size);
    //memset(mom_ders_der2coeffs, 0, sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3 * ntypes * ntypes * nmus * chebyshev_size);

    int max_alpha_index_basic = 0;
    for (int ii=0; ii<alpha_index_basic_count; ii++) {
        int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][2];
        if (now_alpha_index_basic > max_alpha_index_basic)
            max_alpha_index_basic = now_alpha_index_basic;
    }
    max_alpha_index_basic++;

    CoordType *auto_dist_powers_;
    CoordType (*auto_coords_powers_)[3];
    auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic);
    auto_coords_powers_ = (CoordType (*)[3])malloc(sizeof(CoordType) * max_alpha_index_basic * 3);
    memset(auto_dist_powers_, 0, sizeof(CoordType) * max_alpha_index_basic);
    memset(auto_coords_powers_, 0, sizeof(CoordType) * max_alpha_index_basic * 3);
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    int type_central;
    int type_outer;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    RQ_Chebyshev<CoordType> *p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);
    // Step 2.
    for (int ii=0; ii<inum; ii++)
    {
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);
        memset(mom_der2coeffs, 0, sizeof(CoordType) * alpha_moments_count * num_coeffs);
        memset(mom_ders_der2coeffs, 0, sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3 * num_coeffs);
        type_central = types[ilist[ii]];

        for (int jj=0; jj<numneigh[ii]; jj++) 
        {
            type_outer = types[firstneigh[ii*umax_num_neigh_atoms + jj]];
            for (int a=0; a<3; a++)
                NeighbVect[a] = relative_coords[ii*umax_num_neigh_atoms + jj][a];
            distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                   + std::pow(NeighbVect[1], 2)
                                   + std::pow(NeighbVect[2], 2) );
            if (distance_ij > rmax)
                continue;
            distance_ij_inv = 1 / distance_ij;
            p_RadialBasis->build(distance_ij);

            auto_dist_powers_[0] = 1.0;
            for (int a=0; a<3; a++)
                auto_coords_powers_[0][a] = 1.0;
            for (int k=1; k<max_alpha_index_basic; k++) {
                auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
                for (int a=0; a<3; a++)
                    auto_coords_powers_[k][a] = auto_coords_powers_[k-1][a] * NeighbVect[a];
            }

            for (int i=0; i<alpha_index_basic_count; i++)
            {
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
                    if (alpha_index_basic[i][1] != 0)
                        B_ders[0] = alpha_index_basic[i][1] * auto_coords_powers_[alpha_index_basic[i][1] - 1][0] 
                                    * pow1 
                                    * pow2;
                    if (alpha_index_basic[i][2] != 0)
                        B_ders[1] = alpha_index_basic[i][2] * auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
                                    * pow0
                                    * pow2;
                    if (alpha_index_basic[i][3] != 0)
                        B_ders[2] = alpha_index_basic[i][3] * auto_coords_powers_[alpha_index_basic[i][3] - 1][2]
                                    * pow0
                                    * pow1;
                    C_ders[0] = -k * powk * distance_ij_inv * NeighbVect[0] * distance_ij_inv;
                    C_ders[1] = -k * powk * distance_ij_inv * NeighbVect[1] * distance_ij_inv;
                    C_ders[2] = -k * powk * distance_ij_inv * NeighbVect[2] * distance_ij_inv;

                    mom_vals[i] += coeffs[idx] * A * B * C;
                    mom_der2coeffs[i*num_coeffs + idx] += A * B * C;
                    CoordType tmp0 = A_ders[0]*B*C + A*B_ders[0]*C + A*B*C_ders[0];
                    CoordType tmp1 = A_ders[1]*B*C + A*B_ders[1]*C + A*B*C_ders[1];
                    CoordType tmp2 = A_ders[2]*B*C + A*B_ders[2]*C + A*B*C_ders[2];
                    mom_ders[i*umax_num_neigh_atoms + jj][0] += coeffs[idx] * tmp0;
                    mom_ders[i*umax_num_neigh_atoms + jj][1] += coeffs[idx] * tmp1;
                    mom_ders[i*umax_num_neigh_atoms + jj][2] += coeffs[idx] * tmp2;
                    mom_ders_der2coeffs[i*umax_num_neigh_atoms*3*num_coeffs + jj*3*num_coeffs + 0*num_coeffs + idx] += tmp0;
                    mom_ders_der2coeffs[i*umax_num_neigh_atoms*3*num_coeffs + jj*3*num_coeffs + 1*num_coeffs + idx] += tmp1;
                    mom_ders_der2coeffs[i*umax_num_neigh_atoms*3*num_coeffs + jj*3*num_coeffs + 2*num_coeffs + idx] += tmp2;
                }
            }
        }

        for (int i=0; i<alpha_index_times_count; i++)
        {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = mom_vals[alpha_index_times[i][2]];

            mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;

            for (int tmp_type_central=0; tmp_type_central<ntypes; tmp_type_central++) {
                for (int tmp_type_outer=0; tmp_type_outer<ntypes; tmp_type_outer++) {
                    for (int q=0; q<num_mus4moms[alpha_index_times[i][0]]; q++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            int idx0 = (tmp_type_central*ntypes + tmp_type_outer)*nmus*chebyshev_size
                                       + mus4moms_ptr[alpha_index_times[i][0]*max_num_mus4mom + q]*chebyshev_size
                                       + xi;
                            
                            mom_der2coeffs[alpha_index_times[i][3]*num_coeffs + idx0] += val2
                                * mom_der2coeffs[alpha_index_times[i][0]*num_coeffs + idx0]
                                * val1;
                        }
                    }

                    for (int q=0; q<num_mus4moms[alpha_index_times[i][1]]; q++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            int idx1 = (tmp_type_central*ntypes + tmp_type_outer)*nmus*chebyshev_size
                                       + mus4moms_ptr[alpha_index_times[i][1]*max_num_mus4mom + q]*chebyshev_size
                                       + xi;
                            
                            mom_der2coeffs[alpha_index_times[i][3]*num_coeffs + idx1] += val2
                                * val0
                                * mom_der2coeffs[alpha_index_times[i][1]*num_coeffs + idx1];
                        }
                    }
                }
            }

            for (int jj=0; jj<numneigh[ii]; jj++)
            {
                for (int a=0; a<3; a++)
                    NeighbVect[a] = relative_coords[ii*umax_num_neigh_atoms + jj][a];
                distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                       + std::pow(NeighbVect[1], 2)
                                       + std::pow(NeighbVect[2], 2) );
                if (distance_ij > rmax)
                    continue;

                mom_ders[alpha_index_times[i][3]*umax_num_neigh_atoms + jj][0] += val2 *
                    ( mom_ders[alpha_index_times[i][0]*umax_num_neigh_atoms + jj][0] * val1
                    + val0 * mom_ders[alpha_index_times[i][1]*umax_num_neigh_atoms + jj][0]);
                mom_ders[alpha_index_times[i][3]*umax_num_neigh_atoms + jj][1] += val2 *
                    ( mom_ders[alpha_index_times[i][0]*umax_num_neigh_atoms + jj][1] * val1
                    + val0 * mom_ders[alpha_index_times[i][1]*umax_num_neigh_atoms + jj][1]);
                mom_ders[alpha_index_times[i][3]*umax_num_neigh_atoms + jj][2] += val2 *
                    ( mom_ders[alpha_index_times[i][0]*umax_num_neigh_atoms + jj][2] * val1
                    + val0 * mom_ders[alpha_index_times[i][1]*umax_num_neigh_atoms + jj][2]);

                for (int tmp_type_central=0; tmp_type_central<ntypes; tmp_type_central++) {
                    for (int tmp_type_outer=0; tmp_type_outer<ntypes; tmp_type_outer++) {
                        for (int q=0; q<num_mus4moms[alpha_index_times[i][0]]; q++) {
                            for (int xi=0; xi<chebyshev_size; xi++) {
                                for (int a=0; a<3; a++) {
                                    int cidx0 = (tmp_type_central*ntypes + tmp_type_outer)*nmus*chebyshev_size
                                                + mus4moms_ptr[alpha_index_times[i][0]*max_num_mus4mom + q]*chebyshev_size
                                                + xi;
                                    int idx0 = ( alpha_index_times[i][0]*umax_num_neigh_atoms*3*num_coeffs 
                                               + jj*3*num_coeffs
                                               + a*num_coeffs
                                               + cidx0);
                                    int idx3 = ( alpha_index_times[i][3]*umax_num_neigh_atoms*3*num_coeffs
                                               + jj*3*num_coeffs
                                               + a*num_coeffs
                                               + cidx0);
                                    
                                    mom_ders_der2coeffs[idx3] += val2 * (
                                        mom_ders_der2coeffs[idx0] * val1
                                        + mom_der2coeffs[alpha_index_times[i][0]*num_coeffs + cidx0] * mom_ders[alpha_index_times[i][1]*umax_num_neigh_atoms + jj][a]);
                                }
                            }
                        }

                        for (int q=0; q<num_mus4moms[alpha_index_times[i][1]]; q++) {
                            for (int xi=0; xi<chebyshev_size; xi++) {
                                for (int a=0; a<3; a++) {
                                    int cidx1 = (tmp_type_central*ntypes + tmp_type_outer)*nmus*chebyshev_size
                                                + mus4moms_ptr[alpha_index_times[i][1]*max_num_mus4mom + q]*chebyshev_size
                                                + xi;
                                    int idx1 = ( alpha_index_times[i][1]*umax_num_neigh_atoms*3*num_coeffs
                                               + jj*3*num_coeffs
                                               + a*num_coeffs
                                               + cidx1);
                                    int idx3 = ( alpha_index_times[i][3]*umax_num_neigh_atoms*3*num_coeffs
                                               + jj*3*num_coeffs
                                               + a*num_coeffs
                                               + cidx1);
                                    
                                    mom_ders_der2coeffs[idx3] += val2 * (
                                        val0 * mom_ders_der2coeffs[idx1]
                                        + mom_der2coeffs[alpha_index_times[i][1]*num_coeffs + cidx1] * mom_ders[alpha_index_times[i][0]*umax_num_neigh_atoms + jj][a]);
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int i=0; i<alpha_scalar_moments; i++) {
            for (int jj=0; jj<numneigh[ii]; jj++) {
                for (int a=0; a<3; a++)
                    NeighbVect[a] = relative_coords[ii*umax_num_neigh_atoms + jj][a];
                distance_ij = std::sqrt( std::pow(NeighbVect[0], 2) 
                                       + std::pow(NeighbVect[1], 2)
                                       + std::pow(NeighbVect[2], 2));
                if (distance_ij > rmax)
                    continue;

                for (int a=0; a<3; a++) {
                    mbg_val[ii*alpha_scalar_moments*umax_num_neigh_atoms*3 + i*umax_num_neigh_atoms*3 + jj*3 + a][a] = 
                        mom_ders[alpha_moment_mapping[i]*umax_num_neigh_atoms*3 + jj*3 + a][3];
                    for (int idx=0; idx<num_coeffs; idx++) {
                        mbg_der2coeffs[ii*alpha_scalar_moments*umax_num_neigh_atoms*3*num_coeffs + i*umax_num_neigh_atoms*3*num_coeffs + jj*3*num_coeffs + a*num_coeffs + idx] =
                            mom_ders_der2coeffs[alpha_moment_mapping[i]*umax_num_neigh_atoms*3*num_coeffs + jj*3*num_coeffs + a*num_coeffs + idx];
                    }
                }
            }
        }
    }

    // Step . Free
    free(mom_vals);
    free(mom_ders);
    free(mom_der2coeffs);
    free(auto_dist_powers_);
    free(auto_coords_powers_);
    delete p_RadialBasis;
}


};  // namespace : mtpr
};  // namespace : ai2pot


#endif
