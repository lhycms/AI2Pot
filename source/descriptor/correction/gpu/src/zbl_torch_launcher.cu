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
#include "../include/zbl_torch_launcher.cuh"


template void ai2pot::correction::correct_zbl_efv_torch_launcher<float>(
    float *d_etot_ptr,
    float *d_force,
    float *d_virial,
    float rmax,
    float rmin,
    float *d_cks,
    float *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms);

template void ai2pot::correction::correct_zbl_efv_torch_launcher<double>(
    double *d_etot_ptr,
    double *d_force,
    double *d_virial,
    double rmax,
    double rmin,
    double *d_cks,
    double *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms);


template void ai2pot::correction::correct_zbl_ef_torch_launcher<float>(
    float *d_etot_ptr,
    float *d_force,
    float rmax,
    float rmin,
    float *d_cks,
    float *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms
);


template void ai2pot::correction::correct_zbl_ef_torch_launcher<double>(
    double *d_etot_ptr,
    double *d_force,
    double rmax,
    double rmin,
    double *d_cks,
    double *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms
);
