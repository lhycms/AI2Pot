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

#ifndef AI2POT_FITUTILS_EFV_RMSE_H
#define AI2POT_FITUTILS_EFV_RMSE_H

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

namespace ai2pot {
namespace fitutils {

template <typename CoordType>
static void find_ef_rmse_cpu_launcher(
    CoordType &e_rmse,
    CoordType &f_rmse,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3])
{
    CoordType e_mse = 0.0;
    int num_e = 0;
    CoordType f_mse = 0.0;
    int num_f = 0;

    for (int bb=0; bb<batch_size; bb++) {
        CoordType tmp_e_diff = (betot_ml[bb] - betot_dft[bb]) / binum[bb];
        e_mse += tmp_e_diff * tmp_e_diff;
        num_e += 1;

        for (int ii=0; ii<binum[bb]; ii++) {
            int center_idx = bilist[bb*natoms_pad + ii];
            for (int aa=0; aa<3; aa++) {
                CoordType tmp_f_diff = bforce_ml[bb*natoms_pad + center_idx][aa] - bforce_dft[bb*natoms_pad + center_idx][aa];
                f_mse += tmp_f_diff * tmp_f_diff;
            }
            num_f += 3;
        }
    }

    e_rmse = std::sqrt(e_mse / num_e);
    f_rmse = std::sqrt(f_mse / num_f);
}


template <typename CoordType>
static void find_efv_rmse_cpu_launcher(
    CoordType &e_rmse,
    CoordType &f_rmse,
    CoordType &v_rmse,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
    CoordType *bvirial_dft)
{
    CoordType e_mse = 0.0;
    int num_e = 0;
    CoordType f_mse = 0.0;
    int num_f = 0;
    CoordType v_mse = 0.0;
    int num_v = 0;

    for (int bb=0; bb<batch_size; bb++) {
        CoordType tmp_e_diff = (betot_ml[bb] - betot_dft[bb]) / binum[bb];
        e_mse += tmp_e_diff * tmp_e_diff;
        num_e += 1;

        for (int ii=0; ii<binum[bb]; ii++) {
            int center_idx = bilist[bb*natoms_pad + ii];
            for (int aa=0; aa<3; aa++) {
                CoordType tmp_f_diff = bforce_ml[bb*natoms_pad + center_idx][aa] - bforce_dft[bb*natoms_pad + center_idx][aa];
                f_mse += tmp_f_diff * tmp_f_diff;
            }
            num_f += 3;
        }

        for (int a=0; a<3; a++) {
            for (int b=0; b<3; b++) {
                CoordType tmp_v_diff = (bvirial_ml[bb*9 + a*3+b] - bvirial_dft[bb*9 + a*3+b]) / binum[bb];
                v_mse += tmp_v_diff * tmp_v_diff;
            }
        }
        num_v += 9;
    }

    e_rmse = std::sqrt(e_mse / num_e);
    f_rmse = std::sqrt(f_mse / num_f);
    v_rmse = std::sqrt(v_mse / num_v);
}

};  // namespace : fitutils
};  // namespaec : ai2pot

#endif