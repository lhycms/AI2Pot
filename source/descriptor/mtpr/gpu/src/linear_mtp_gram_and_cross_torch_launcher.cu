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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/mtp_utilities.cuh"
#include "../include/linear_mtp_gram_and_cross_torch_launcher.cuh"


namespace ai2pot {
namespace mtpr {

// 1. find_efv_components()
template void find_efv_components_torch_launcher<float>(
    float *d_benergy_components,
    float *d_bforce_components,
    float *d_bvirial_components,
    int chebyshev_size,
    float *d_coeffs,
    float *d_linear_coeffs,
    float *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin,
    float *d_q_scaler);

template void find_efv_components_torch_launcher<double>(
    double *d_benergy_components,
    double *d_bforce_components,
    double *d_bvirial_components,
    int chebyshev_size,
    double *d_coeffs,
    double *d_linear_coeffs,
    double *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin,
    double *d_q_scaler);

// 2. find_lin_matrix_lin_vector()
template void find_lin_matrix_lin_vector_torch_launcher<float>(
    float *d_lin_matrix,
    float *d_lin_vector,
    float e_weight,
    float f_weight,
    float v_weight,
    float *d_betot_residual,
    float (*d_bforce_residual)[3],
    float *d_bvirial_residual,
    float *d_benergy_components,
    float *d_bforce_components,
    float *d_bvirial_components,
    int alpha_scalar_moments,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int ntypes);

template void find_lin_matrix_lin_vector_torch_launcher<double>(
    double *d_lin_matrix,
    double *d_lin_vector,
    double e_weight,
    double f_weight,
    double v_weight,
    double *d_betot_residual,
    double (*d_bforce_residual)[3],
    double *d_bvirial_residual,
    double *d_benergy_components,
    double *d_bforce_components,
    double *d_bvirial_components,
    int alpha_scalar_moments,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int ntypes);


// 3. mirror_lin_matrix()
template void mirror_lin_matrix_torch_launcher<float>(
    float *d_lin_matrix,
    int num_parameters);

template void mirror_lin_matrix_torch_launcher<double>(
    double *d_lin_matrix,
    int num_parameters);


};  // namespace : mtpr
};  // namespace : ai2pot
