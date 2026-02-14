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

#ifndef AI2POT_NEP_NEP_LOSS_CPU_LAUNCHER_H
#define AI2POT_NEP_NEP_LOSS_CPU_LAUNCHER_H

#include <stdio.h>
#include <stdlib.h>

#include "./nep_utilities.h"
#include "./nepLoss.h"


namespace ai2pot {
namespace nep {


template <typename CoordType>
static void find_ef_loss_cpu_launcher(
    CoordType *bloss,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3])
{
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *loss_ptr = bloss + bb;
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        CoordType *etot_ml_ptr = betot_ml + bb;
        CoordType *etot_dft_ptr = betot_dft + bb;
        CoordType (*force_ml)[3] = &bforce_ml[bb*natoms_pad];
        CoordType (*force_dft)[3] = &bforce_dft[bb*natoms_pad];

        NepLoss<CoordType>::find_ef_loss(
            *loss_ptr,
            inum,
            ilist,
            e_weight,
            f_weight,
            *etot_ml_ptr,
            *etot_dft_ptr,
            force_ml,
            force_dft);
    }
}



template <typename CoordType>
static void find_ef_loss_backward_cpu_launcher(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2w0,
    CoordType *bloss_der2b0,
    CoordType *bloss_der2w1,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *b0,
    CoordType *w1,
    CoordType *type_bias,
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
    CoordType rmax_radial,
    CoordType rmax_angular,
    CoordType *q_scaler)
{
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    int num_coeffs = ntypes * ntypes * (n_radial_basis + n_angular_basis) * chebyshev_size;
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *loss_der2coeffs = &bloss_der2coeffs[bb*num_coeffs];
        CoordType *loss_der2w0 = &bloss_der2w0[bb*ntypes*num_neurons*num_descriptors];
        CoordType *loss_der2b0 = &bloss_der2b0[bb*ntypes*num_neurons];
        CoordType *loss_der2w1 = &bloss_der2w1[bb*ntypes*num_neurons];
        CoordType *loss_der2type_bias = &bloss_der2type_bias[bb*ntypes];

        CoordType *etot_ml_ptr = betot_ml + bb;
        CoordType *etot_dft_ptr = betot_dft + bb;
        CoordType (*force_ml)[3] = &bforce_ml[bb*natoms_pad];
        CoordType (*force_dft)[3] = &bforce_dft[bb*natoms_pad];
        
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb*natoms_pad];

        NepLoss<CoordType>::find_ef_loss_backward(
            loss_der2coeffs,
            loss_der2w0,
            loss_der2b0,
            loss_der2w1,
            loss_der2type_bias,
            e_weight,
            f_weight,
            *etot_ml_ptr,
            *etot_dft_ptr,
            force_ml,
            force_dft,
            chebyshev_size,
            n_radial_basis,
            n_angular_basis,
            l_max,
            num_neurons,
            coeffs,
            w0,
            b0,
            w1,
            type_bias,
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
            rmax_radial,
            rmax_angular,
            q_scaler);
    }
}


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
        e_mse += (betot_ml[bb] - betot_dft[bb]) * (betot_ml[bb] - betot_dft[bb]);
        num_e += 1;

        for (int ii=0; ii<binum[bb]; ii++) {
            for (int aa=0; aa<3; aa++) {
                f_mse += (bforce_ml[bb*natoms_pad + ii][aa] - bforce_dft[bb*natoms_pad + ii][aa])
                         * (bforce_ml[bb*natoms_pad + ii][aa] - bforce_dft[bb*natoms_pad + ii][aa]);
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
        e_mse += (betot_ml[bb] - betot_dft[bb]) * (betot_ml[bb] - betot_dft[bb]);
        num_e += 1;

        for (int ii=0; ii<binum[bb]; ii++) {
            for (int aa=0; aa<3; aa++) {
                f_mse += (bforce_ml[bb*natoms_pad + ii][aa] - bforce_dft[bb*natoms_pad + ii][aa])
                         * (bforce_ml[bb*natoms_pad + ii][aa] - bforce_dft[bb*natoms_pad + ii][aa]);
            }
            num_f += 3;
        }

        for (int a=0; a<3; a++) {
            for (int b=0; b<3; b++) {
                v_mse += (bvirial_ml[bb*9 + a*3+b] - bvirial_dft[bb*9 + a*3+b])
                         * (bvirial_ml[bb*9 + a*3+b] - bvirial_dft[bb*9 + a*3+b]);
            }
        }
        num_v += 9;
    }

    e_rmse = std::sqrt(e_mse / num_e);
    f_rmse = std::sqrt(f_mse / num_f);
    v_rmse = std::sqrt(v_mse / num_v);
}


};  // namespace : nep
};  // namespace : ai2pot

#endif