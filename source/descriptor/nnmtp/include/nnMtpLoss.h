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

#ifndef AI2POT_NNMTP_MTP_LOSS_H
#define AI2POT_NNMTP_MTP_LOSS_H
#include <stdlib.h>
#include <cstring>
#include <cmath>


namespace ai2pot {
namespace nnmtp {

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
void MtpLoss<CoordType>::find_ef_loss(
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


};  // namespace : nnmtp
};  // namespace : ai2pot

#endif
