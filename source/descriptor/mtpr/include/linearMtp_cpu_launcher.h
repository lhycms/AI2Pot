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

#ifndef AI2POT_MTPR_LINEAR_MTP_CPU_LAUNCHER_H
#define AI2POT_MTPR_LINEAR_MTP_CPU_LAUNCHER_H


#include "./linearMtp.h"


namespace ai2pot {
namespace mtpr {


template <typename CoordType>
static void find_efv_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
    CoordType *bvirial,
    int chebyshev_size,
    CoordType scaling,
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
static void find_ef_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
    int chebyshev_size,
    CoordType scaling,
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
static void find_descriptors_cpu_launcher(
    CoordType *bdescriptors,
    int chebyshev_size,
    CoordType scaling,
    CoordType *coeffs,
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





template <typename CoordType>
void find_efv_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
    CoordType *bvirial,
    int chebyshev_size,
    CoordType scaling,
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
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *etot_ptr = betot + bb;
        CoordType (*force)[3] = &bforce[bb*(natoms_pad + nghost)];
        CoordType *virial = &bvirial[bb*9];
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb*(natoms_pad+nghost)];

        LinearMtp<CoordType>::find_efv(
            *etot_ptr,
            force,
            virial,
            chebyshev_size,
            scaling,
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
void find_ef_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
    int chebyshev_size,
    CoordType scaling,
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
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *etot_ptr = betot + bb;
        CoordType (*force)[3] = &bforce[bb*(natoms_pad + nghost)];
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb*(natoms_pad+nghost)];

        LinearMtp<CoordType>::find_ef(
            *etot_ptr,
            force,
            chebyshev_size,
            scaling,
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
void find_descriptors_cpu_launcher(
    CoordType *bdescriptors,
    int chebyshev_size,
    CoordType scaling,
    CoordType *coeffs,
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
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *descriptors = &bdescriptors[bb*natoms_pad*alpha_scalar_moments];
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb*natoms_pad];

        LinearMtp<CoordType>::find_descriptors(
            descriptors,
            chebyshev_size,
            scaling,
            coeffs,
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
            rmin);
    }
}


};  // namespace : mtpr
};  // namespace : ai2pot


#endif