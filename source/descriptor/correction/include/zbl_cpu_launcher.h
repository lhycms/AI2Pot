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

#ifndef AI2POT_ZBL_CPU_LAUNCHER
#define AI2POT_ZBL_CPU_LAUNCHER

#include "./zbl.h"


namespace ai2pot {
namespace correction {

template <typename CoordType>
void find_zbl_efv_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
    CoordType *bvirial,
    int *Zis,
    int *Zjs,
    CoordType rmax,
    CoordType rmin,
    CoordType *cks,
    CoordType *dks,
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
    int nghost);


template <typename CoordType>
void find_zbl_ef_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
    int *Zis,
    int *Zjs,
    CoordType rmax,
    CoordType rmin,
    CoordType *cks,
    CoordType *dks,
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
    int nghost);


template <typename CoordType>
void find_zbl_efv_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
    CoordType *bvirial,
    int *Zis,
    int *Zjs,
    CoordType rmax,
    CoordType rmin,
    CoordType *cks,
    CoordType *dks,
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
    int nghost)
{
    GroupZBL<CoordType> group_zbl(
                ntypes,
                Zis,
                Zjs,
                rmax,
                rmin,
                cks,
                dks);
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *etot_ptr = &betot[bb];
        CoordType (*force)[3] = &bforce[bb * (natoms_pad+nghost)];
        CoordType *virial = &bvirial[bb * 9];
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb * (natoms_pad+nghost)];

        group_zbl.correct_efv(
            *etot_ptr,
            force,
            virial,
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            ntypes,
            type_map,
            umax_num_neigh_atoms,
            nghost);
    }
}


template <typename CoordType>
void find_zbl_ef_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
    int *Zis,
    int *Zjs,
    CoordType rmax,
    CoordType rmin,
    CoordType *cks,
    CoordType *dks,
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
    int nghost)
{
    GroupZBL<CoordType> group_zbl(
                ntypes,
                Zis,
                Zjs,
                rmax,
                rmin,
                cks,
                dks);
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *etot_ptr = &betot[bb];
        CoordType (*force)[3] = &bforce[bb * (natoms_pad+nghost)];
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb * (natoms_pad+nghost)];

        group_zbl.correct_ef(
            *etot_ptr,
            force,
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            ntypes,
            type_map,
            umax_num_neigh_atoms,
            nghost);
    }
}

};  // namespace : correction
};  // namespace : ai2pot

#endif
