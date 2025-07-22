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

#include "../include/nnMtp_torch_launcher.cuh"


// 1. find_efv
template void ai2pot::nnmtp::find_efv_torch_launcher<float>(
    float *d_etot_ptr,
    float (*d_force)[3],
    float *d_virial,
    int chebyshev_size,
    int num_neurons,
    float *d_coeffs,
    float *d_w0,
    float *d_w1,
    float *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin);


template void ai2pot::nnmtp::find_efv_torch_launcher<double>(
    double *d_etot_ptr,
    double (*d_force)[3],
    double *d_virial,
    int chebyshev_size,
    int num_neurons,
    double *d_coeffs,
    double *d_w0,
    double *d_w1,
    double *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin);



// 2. find_ef
template void ai2pot::nnmtp::find_ef_torch_launcher<float>(
    float *d_etot_ptr,
    float (*d_force)[3],
    int chebyshev_size,
    int num_neurons,
    float *d_coeffs,
    float *d_w0,
    float *d_w1,
    float *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin);

template void ai2pot::nnmtp::find_ef_torch_launcher<double>(
    double *d_etot_ptr,
    double (*d_force)[3],
    int chebyshev_size,
    int num_neurons,
    double *d_coeffs,
    double *d_w0,
    double *d_w1,
    double *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin);
