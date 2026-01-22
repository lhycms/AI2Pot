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

#include "../include/nep_torch_launcher.cuh"

// 1. find_efv


// 2. find_ef
template void ai2pot::nep::find_ef_torch_launcher<float>(
    float *d_betot_ptr,
    float (*d_bforce)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    float *d_coeffs,
    float *d_w0,
    float *d_b0,
    float *d_w1,
    float *d_type_bias,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin,
    float *d_q_scaler);

template void ai2pot::nep::find_ef_torch_launcher<double>(
    double *d_betot_ptr,
    double (*d_bforce)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    double *d_coeffs,
    double *d_w0,
    double *d_b0,
    double *d_w1,
    double *d_type_bias,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin,
    double *d_q_scaler);


// 3. find_descriptors
template void ai2pot::nep::find_descriptor_torch_launcher<float>(
    float *d_bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    float *d_coeffs,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin);

template void ai2pot::nep::find_descriptor_torch_launcher<double>(
    double *d_bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    double *d_coeffs,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin);