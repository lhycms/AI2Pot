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
        CoordType e_weight,
        CoordType f_weight,
        CoordType v_weight,
        CoordType etot_ml,
        CoordType etot_dft,
        CoordType (*force_ml)[3],
        CoordType (*force_dft)[3],
        CoordType *virial_ml,
        CoordType *virial_dft,
        int inum);

    static void find_loss_der2efv(
        CoordType &loss_der2e,
        CoordType (*loss_der2f)[3],
        CoordType *loss_der2v,
        CoordType e_weight,
        CoordType f_weight,
        CoordType v_weight,
        CoordType etot_ml,
        CoordType etot_dft,
        CoordType (*force_ml)[3],
        CoordType (*force_dft)[3],
        CoordType *virial_ml,
        CoordType *virial_dft,
        int inum);

    static void find_loss_der2eders_one(
        CoordType (*loss_der2eders)[3],
        CoordType *loss_der2f,
        CoordType *loss_der2v,
        int silist,
        int snumneigh,
        int *sfirstneigh,
        CoordType (*relative_coords)[3],
        int *types,
        int ntypes,
        int umax_num_neigh_atoms,
        CoordType rmax,
        CoordType rmin);

    static void find_loss_der2coeffs_one(
        CoordType *loss_der2coeffs,
        CoordType *loss_der2linear_coeffs,
        CoordType *loss_der2type_bias,
        CoordType *loss_der2e,
        CoordType (*loss_der2eders)[3],
        int silist,
        int snumneigh,
        int *sfirstneigh,
        CoordType (*relative_coords)[3],
        int *types,
        int ntypes,
        int umax_num_neigh_atoms,
        CoordType rmax,
        CoordType rmin);
};  // class : MtpLoss


template <typename CoordType>
void MtpLoss<CoordType>::find_loss(
    CoordType &loss,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    CoordType *virial_ml,
    CoordType *virial_dft,
    int inum)
{
    loss = 0;

    // Force term
    CoordType f_loss = 0;
    for (int ii=0; ii<inum; ii++)
        for (int aa=0; aa<3; aa++)
            f_loss += std::pow(force_ml[ii][aa] - force_dft[ii][aa], 2);
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
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*force_ml)[3],
    CoordType (*force_dft)[3],
    CoordType *virial_ml,
    CoordType *virial_dft,
    int inum)
{
    loss_der2e = 0;
    memset(loss_der2f, 0, sizeof(CoordType) * inum * 3);
    memset(loss_der2v, 0, sizeof(CoordType) * 9);

    // loss_der2e
    loss_der2e = e_weight / inum * 2.0 * (etot_ml - etot_dft);

    // loss_der2f
    for (int ii=0; ii<inum; ii++)
        for (int aa=0; aa<3; aa++)
            loss_der2f[ii][aa] = f_weight / (3 * inum) * 2.0 * (force_ml[ii][aa] - force_dft[ii][aa]);

    // loss_der2v
    for (int aa=0; aa<3; aa++)
        for (int bb=0; bb<3; bb++)
            loss_der2v[aa*3 + bb] = v_weight / (9 * inum) * 2.0 * (virial_ml[aa][bb] - virial_dft[aa][bb]);
}


template <typename CoordType>
void MtpLoss<CoordType>::find_loss_der2eders_one(
    CoordType (*loss_der2eders)[3],
    CoordType *loss_der2f,
    CoordType *loss_der2v,
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srelative_coords)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    CoordType rmax,
    CoordType rmin)
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

};  // namespace : mtpr
};  // namespace : ai2pot

#endif
