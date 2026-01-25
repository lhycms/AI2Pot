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

#ifndef AI2POT_NEP_NEP_CPU_LAUNCHER_H
#define AI2POT_NEP_NEP_CPU_LAUNCHER_H


#include "./nep.h"

namespace ai2pot {
namespace nep {


template <typename CoordType>
static void find_ef_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
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
    CoordType *q_scaler);


template <typename CoordType>
static void find_descriptors_cpu_launcher(
    CoordType *bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *coeffs,
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
    CoordType rmax_angular);




template <typename CoordType>
void find_ef_cpu_launcher(
    CoordType *betot,
    CoordType (*bforce)[3],
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
    for (int bb=0; bb<batch_size; bb++) {
        CoordType *etot_ptr = betot + bb;
        CoordType (*force)[3] = &bforce[bb*(natoms_pad+nghost)];
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb*(natoms_pad+nghost)];

        Nep<CoordType>::find_ef(
            *etot_ptr,
            force,
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
void find_descriptors_cpu_launcher(
    CoordType *bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *coeffs,
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
    CoordType rmax_angular)
{
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);

    for (int bb=0; bb<batch_size; bb++) {
        CoordType *descriptors = &bdescriptors[bb*natoms_pad*num_descriptors];
        int inum = binum[bb];
        int *ilist = &bilist[bb*natoms_pad];
        int *numneigh = &bnumneigh[bb*natoms_pad];
        int *firstneigh = &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms];
        CoordType (*rcs)[3] = &brcs[bb*natoms_pad*umax_num_neigh_atoms];
        int *types = &btypes[bb*natoms_pad];

        Nep<CoordType>::find_descriptors(
            descriptors,
            chebyshev_size,
            n_radial_basis,
            n_angular_basis,
            l_max,
            coeffs,
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
            rmax_angular);
    }
}


};  // namespace : nep
};  // namespace : ai2pot

#endif