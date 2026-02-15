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

#ifndef AI2POT_MTPR_LINEAR_MTP_LOSS_CPU_LAUNCHER
#define AI2POT_MTPR_LINEAR_MTP_LOSS_CPU_LAUNCHER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./linearMtp.h"


namespace ai2pot {
namespace mtpr {

template <typename CoordType>
static void find_loss_cpu_launcher(
    CoordType *bloss,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
    CoordType *bvirial_dft);

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
    CoordType (*bforce_dft)[3]);

template <typename CoordType>
static void find_loss_backward_cpu_launcher(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2linear_coeffs,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
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
    CoordType rmin,
    CoordType *q_scaler);

template <typename CoordType>
static void find_ef_loss_backward_cpu_launcher(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2linear_coeffs,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
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
    CoordType rmin,
    CoordType *q_scaler);









template <typename CoordType>
static void find_loss_cpu_launcher(
    CoordType *bloss,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
    CoordType *bvirial_dft)
{
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *loss_ptr = bloss + bb;
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        CoordType *etot_ml_ptr = betot_ml + bb;
        CoordType *etot_dft_ptr = betot_dft + bb;
        CoordType (*force_ml)[3] = &bforce_ml[bb*natoms_pad];
        CoordType (*force_dft)[3] = &bforce_dft[bb*natoms_pad];
        CoordType *virial_ml = &bvirial_ml[bb*9];
        CoordType *virial_dft = &bvirial_dft[bb*9];

        LinearMtpLoss<CoordType>::find_loss(
            *loss_ptr,
            inum,
            ilist,
            e_weight,
            f_weight,
            v_weight,
            *etot_ml_ptr,
            *etot_dft_ptr,
            force_ml,
            force_dft,
            virial_ml,
            virial_dft);
    }
}


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

        LinearMtpLoss<CoordType>::find_ef_loss(
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
static void find_loss_backward_cpu_launcher(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2linear_coeffs,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
    CoordType *bvirial_ml,
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
    CoordType rmin,
    CoordType *q_scaler)
{
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    for (int bb=0; bb<batch_size; bb++) {
        CoordType *loss_der2coeffs = &bloss_der2coeffs[bb*num_coeffs];
        CoordType *loss_der2linear_coeffs = &bloss_der2linear_coeffs[bb*alpha_scalar_moments];
        CoordType *loss_der2type_bias = &bloss_der2type_bias[bb*ntypes];

        CoordType *etot_ml_ptr = betot_ml + bb;
        CoordType *etot_dft_ptr = betot_dft + bb;
        CoordType (*force_ml)[3] = &bforce_ml[bb*natoms_pad];
        CoordType (*force_dft)[3] = &bforce_dft[bb*natoms_pad];
        CoordType *virial_ml = &bvirial_ml[bb*9];
        CoordType *virial_dft = &bvirial_dft[bb*9];
        
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb*natoms_pad];

        LinearMtpLoss<CoordType>::find_loss_backward(
            loss_der2coeffs,
            loss_der2linear_coeffs,
            loss_der2type_bias,
            e_weight,
            f_weight,
            v_weight,
            *etot_ml_ptr,
            *etot_dft_ptr,
            force_ml,
            force_dft,
            virial_ml,
            virial_dft,
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
            rmin,
            q_scaler);
    }
}


template <typename CoordType>
static void find_ef_loss_backward_cpu_launcher(
    CoordType *bloss_der2coeffs,
    CoordType *bloss_der2linear_coeffs,
    CoordType *bloss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType *betot_ml,
    CoordType *betot_dft,
    CoordType (*bforce_ml)[3],
    CoordType (*bforce_dft)[3],
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
    CoordType rmin,
    CoordType *q_scaler)
{
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    for (int bb=0; bb<batch_size; bb++) {
        CoordType *loss_der2coeffs = &bloss_der2coeffs[bb*num_coeffs];
        CoordType *loss_der2linear_coeffs = &bloss_der2linear_coeffs[bb*alpha_scalar_moments];
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

        LinearMtpLoss<CoordType>::find_ef_loss_backward(
            loss_der2coeffs,
            loss_der2linear_coeffs,
            loss_der2type_bias,
            e_weight,
            f_weight,
            *etot_ml_ptr,
            *etot_dft_ptr,
            force_ml,
            force_dft,
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
            rmin,
            q_scaler);
    }
}



};  // namespace : mtpr
};  // namespace : ai2pot

#endif
