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

namespace ai2pot {
namespace mtpr {

template <typename CoordType>
class MtpLoss {
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

    static void find_loss_der2efv(
        CoordType &loss_der2e,
        CoordType (*loss_der2f)[3],
        CoordType *loss_der2v,
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

    static void find_loss_der2eders_one(
        CoordType (*loss_der2eders)[3],
        int silist,
        int snumneigh,
        int *sfirstneigh,
        CoordType (*relative_coords)[3],
        int *types,
        int ntypes,
        int umax_num_neigh_atoms,
        CoordType rmax,
        CoordType rmin,
        CoordType *loss_der2f,
        CoordType *loss_der2v);

    static void accumulate_loss_der2coeffs_one(
        CoordType *loss_der2coeffs,
        CoordType *loss_der2linear_coeffs,
        CoordType *loss_der2type_bias,
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
        CoordType (*srelative_coords)[3],
        int *types,
        int ntypes,
        int umax_num_neigh_atoms,
        CoordType rmax,
        CoordType rmin,
        CoordType &loss_der2e,
        CoordType (*loss_der2eders)[3]);
};  // class : MtpLoss


template <typename CoordType>
void MtpLoss<CoordType>::find_loss(
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
    loss = 0;

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
    e_loss += e_weight / inum * std::pow((etot_ml - etot_dft), 2);

    loss = f_loss + v_loss + e_loss;
}


template <typename CoordType>
void MtpLoss<CoordType>::find_loss_der2efv(
    CoordType &loss_der2e,
    CoordType (*loss_der2f)[3],
    CoordType *loss_der2v,
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
    loss_der2e = 0;
    memset(loss_der2f, 0, sizeof(CoordType) * inum * 3);
    memset(loss_der2v, 0, sizeof(CoordType) * 9);

    // loss_der2e
    loss_der2e = e_weight / inum * 2.0 * (etot_ml - etot_dft);

    // loss_der2f
    for (int ii=0; ii<inum; ii++) {
        int center_idx = ilist[ii];
        for (int aa=0; aa<3; aa++) {
            loss_der2f[center_idx][aa] = f_weight / (3 * inum) * 2.0 * (force_ml[center_idx][aa] - force_dft[center_idx][aa]);
        }
    }

    // loss_der2v
    for (int aa=0; aa<3; aa++)
        for (int bb=0; bb<3; bb++)
            loss_der2v[aa*3 + bb] = v_weight / (9 * inum) * 2.0 * (virial_ml[aa][bb] - virial_dft[aa][bb]);
}


template <typename CoordType>
void MtpLoss<CoordType>::find_loss_der2eders_one(
    CoordType (*loss_der2eders)[3],
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srelative_coords)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    CoordType rmax,
    CoordType rmin,
    CoordType *loss_der2f,
    CoordType *loss_der2v)
{
    memset(loss_der2eders, 0, sizeof(CoordType) * umax_num_neigh_atoms * 3);
    CoordType NeighbVect[3] = {0., 0., 0.};
    CoordType distance_ij;
    
    int center_idx = silist;
    // Force -> e_ders
    for (int jj=0; jj<snumneigh; jj++) {
        int neigh_idx = sfirstneigh[jj];
        for (int aa=0; aa<3; aa++)
            NeighbVect[aa] = srelative_coords[jj*3 + aa];
        distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                               + std::pow(NeighbVect[1], 2)
                               + std::pow(NeighbVect[2], 2) );
        if (distance_ij > rmax)
            continue;

        for (int aa=0; aa<3; aa++) {
            loss_der2eders[jj][aa] += loss_der2f[center_idx];
            loss_der2eders[jj][aa] -= loss_der2f[neigh_idx];
        }
    }

    // Virials -> e_ders
    for (int jj=0; jj<snumneigh; jj++)
        for (int aa=0; aa<3; aa++)
            for (int bb=0; bb<3; bb++)
                loss_der2eders[jj][aa] += loss_der2v[aa][bb] * srelative_coords[jj*3 + bb];
}


template <typename CoordType>
void MtpLoss<CoordType>::accumulate_loss_der2coeffs_one(
    CoordType *loss_der2coeffs,
    CoordType *loss_der2linear_coeffs,
    CoordType *loss_der2type_bias,
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
    CoordType (*srelative_coords)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    CoordType rmax,
    CoordType rmin,
    CoordType &loss_der2e,
    CoordType (*loss_der2eders)[3])
{
    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    CoordType *e_site_der2mom;

    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
    e_site_der2mom = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
    memset(mom_ders, 0, sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
    memset(e_site_der2mom, 0, sizeof(CoordType) * alpha_moments_count);

    // Step 1.1.
    int max_alpha_index_basic = 0;
    for (int ii=0; ii<alpha_index_basic_count; ii++) {
        int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][3];
        if (now_alpha_index_basic > max_alpha_index_basic)
            max_alpha_index_basic = now_alpha_index_basic;
    }
    max_alpha_index_basic++;

    CoordType *auto_dist_powers_;
    CoordType (*auto_coords_powers_)[3];
    auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic);
    auto_coords_powers_ = (CoordType (*)[3])malloc(sizeof(CoordType) * max_alpha_index_basic * 3);

    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    int type_central;
    int type_outer;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    RQ_Chebyshev<CoordType> *p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);

    // Step 1.2.
    type_central = types[silist];
    for (int jj=0; jj<snumneigh; jj++) {
        type_outer = types[sfirstneigh[jj]];
        for (int a=0; a<3; a++)
            NeighbVect[a] = srelative_coords[jj][a];
        distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                               + std::pow(NeighbVect[1], 2)
                               + std::pow(NeighbVect[2], 2));
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        p_RadialBasis->build(distance_ij);

        auto_dist_powers_[0] = 1;
        for (int a=0; a<3; a++)
            auto_coords_powers_[0][a] = 1;
        for (int k=1; k<max_alpha_index_basic; k++) {
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            for (int a=0; a<3; a++)
                auto_coords_powers_[k][a] = auto_coords_powers_[k-1][a] * NeighbVect[a];
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
                                * auto_coords_powers_[alpha_index_basic[i][1]-1][0]
                                * pow1
                                * pow2;
                }
                if (alpha_index_basic[i][2] != 0) {
                    B_ders[1] = alpha_index_basic[i][2]
                                * pow0
                                * auto_coords_powers_[alpha_index_basic[i][2]-1][1]
                                * pow2;
                }
                if (alpha_index_basic[i][3] != 0) {
                    B_ders[2] = alpha_index_basic[i][3]
                                * pow0
                                * pow1
                                * auto_coords_powers_[alpha_index_basic[i][3]-1][2];
                }
                C_ders[0] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[0];
                C_ders[1] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[1];
                C_ders[2] = -k * powk * distance_ij_inv * distance_ij_inv * NeighbVect[2];
                mom_vals[i] += coeffs[idx] * A * B * C;
                for (int a=0; a<3; a++)
                    mom_ders[i*umax_num_neigh_atoms + jj][a] += coeffs[idx]
                            * (A_ders[a]*B*C + A*B_ders[a]*C + A*B*C_ders[a]);
            }
        }
    }

    for (int i=0; i<alpha_index_times_count; i++)
    {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];
        mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
    }

    // Backpropagation 2.
    // Step 2.1. 
    for (int i=0; i<alpha_scalar_moments; i++)
        e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i];

    // Step 2.2.
    for (int i=alpha_index_times_count-1; i>=0; i--) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];

        e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val1
                                                   * val2;
        e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val0
                                                   * val2;
    }

    // Step 2.3. 
    for (int i=0; i<alpha_index_basic_count; i++)
        for (int jj=0; jj<snumneigh; jj++)
            for (int a=0; a<3; a++)
                //
                loss_der2linear_coeffs_part_eders[i] += loss_der2eders[jj][a] * mom_ders[i*umax_num_neigh_atoms+jj][a];
    
    // loss_der2moms -- e_site part
    for (int i=0; i<alpha_index_basic_count; i++)
        //
        loss_der2moms[i] = loss_der2e * e_site_der2mom[i];
    
    for (int i=0; i<alpha_index_times_count; i++) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];

        loss_der2linear_coeffs_part_eders[alpha_index_times[i][3]] += 
            loss_der2linear_coeffs_part_eders[alpha_index_times[i][0]]
            * val1 * val2;
        loss_der2linear_coeffs_part_eders[alpha_index_times[i][3]] +=
            loss_der2linear_coeffs_part_eders[alpha_index_times[i][1]]
            * val0 * val2;
    }

    // loss_der2moms -- eders part
    for (int i=0; i<alpha_index_times_count; i++) {
        double val2 = alpha_index_times[i][2];

        loss_der2moms[alpha_index_times[i][0]] += loss
    }



    // Step . Free
    free(mom_vals);
    free(mom_ders);
    delete p_RadialBasis;
}


};  // namespace : mtpr
};  // namespace : ai2pot

#endif
