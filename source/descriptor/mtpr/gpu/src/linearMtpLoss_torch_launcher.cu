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

#include "../include/linearMtpLoss.cuh"
#include "../include/linearMtpLoss_torch_launcher.cuh"

// 1. find_loss_torch_launcher()
template void ai2pot::mtpr::find_loss_torch_launcher<float>(
    float *d_loss_ptr,
    int inum,
    int *d_ilist,
    float e_weight,
    float f_weight,
    float v_weight,
    float etot_ml,
    float etot_dft,
    float (*d_force_ml)[3],
    float (*d_force_dft)[3],
    float *d_virial_ml,
    float *d_virial_dft);

template void ai2pot::mtpr::find_loss_torch_launcher<double>(
    double *d_loss_ptr,
    int inum,
    int *d_ilist,
    double e_weight,
    double f_weight,
    double v_weight,
    double etot_ml,
    double etot_dft,
    double (*d_force_ml)[3],
    double (*d_force_dft)[3],
    double *d_virial_ml,
    double *d_virial_dft);


// 2. find_ef_loss_torch_launcher()
template void ai2pot::mtpr::find_ef_loss_torch_launcher<float>(
    float *d_loss_ptr,
    int inum,
    int *d_ilist,
    float e_weight,
    float f_weight,
    float etot_ml,
    float etot_dft,
    float (*d_force_ml)[3],
    float (*d_force_dft)[3]);


template void ai2pot::mtpr::find_ef_loss_torch_launcher<double>(
    double *d_loss_ptr,
    int inum,
    int *d_ilist,
    double e_weight,
    double f_weight,
    double etot_ml,
    double etot_dft,
    double (*d_force_ml)[3],
    double (*d_force_dft)[3]);


// 3. find_loss_backward_torch_launcher()
template void ai2pot::mtpr::find_loss_backward_torch_launcher<float>(
    float *d_loss_der2coeffs,
    float *d_loss_der2linear_coeffs,
    float *d_loss_der2type_bias,
    float e_weight,
    float f_weight,
    float v_weight,
    float etot_ml,
    float etot_dft,
    float (*d_force_ml)[3],
    float (*d_force_dft)[3],
    float *d_virial_ml,
    float *d_virial_dft,
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


template void ai2pot::mtpr::find_loss_backward_torch_launcher<double>(
    double *d_loss_der2coeffs,
    double *d_loss_der2linear_coeffs,
    double *d_loss_der2type_bias,
    double e_weight,
    double f_weight,
    double v_weight,
    double etot_ml,
    double etot_dft,
    double (*d_force_ml)[3],
    double (*d_force_dft)[3],
    double *d_virial_ml,
    double *d_virial_dft,
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


// 4. find_ef_loss_backward_torch_launcher()
template void ai2pot::mtpr::find_ef_loss_backward_torch_launcher<float>(
    float *d_loss_der2coeffs,
    float *d_loss_der2linear_coeffs,
    float *d_loss_der2type_bias,
    float e_weight,
    float f_weight,
    float etot_ml,
    float etot_dft,
    float (*d_force_ml)[3],
    float (*d_force_dft)[3],
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

template void ai2pot::mtpr::find_ef_loss_backward_torch_launcher<double>(
    double *d_loss_der2coeffs,
    double *d_loss_der2linear_coeffs,
    double *d_loss_der2type_bias,
    double e_weight,
    double f_weight,
    double etot_ml,
    double etot_dft,
    double (*d_force_ml)[3],
    double (*d_force_dft)[3],
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
