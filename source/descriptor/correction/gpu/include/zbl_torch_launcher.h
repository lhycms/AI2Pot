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

#ifndef AI2POT_ZBL_TORCH_LAUNCHER_H
#define AI2POT_ZBL_TORCH_LAUNHCER_H

namespace ai2pot {
namespace correction {

template <typename CoordType>
void correct_zbl_efv_torch_launcher(
    CoordType *d_betot_ptr,
    CoordType *d_bforce,
    CoordType *d_bvirial,
    CoordType rmax,
    CoordType rmin,
    CoordType *d_cks,
    CoordType *d_dks,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    CoordType (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost);


template <typename CoordType>
void correct_zbl_ef_torch_launcher(
    CoordType *d_betot_ptr,
    CoordType *d_bforce,
    CoordType rmax,
    CoordType rmin,
    CoordType *d_cks,
    CoordType *d_dks,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    CoordType (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost);

};  // namespace : correction
};  // namespace : ai2pot

#endif
