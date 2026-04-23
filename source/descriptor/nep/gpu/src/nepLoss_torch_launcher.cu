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


#include "../include/nepLoss_torch_launcher.cuh"


// 1. find_ef_loss
template void ai2pot::nep::find_ef_loss_torch_launcher<float>(
    float *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    float e_weight,
    float f_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3]);

template void ai2pot::nep::find_ef_loss_torch_launcher<double>(
    double *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    double e_weight,
    double f_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3]);


// 2. find_loss
template void ai2pot::nep::find_loss_torch_launcher<float>(
    float *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    float e_weight,
    float f_weight,
    float v_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3],
    float *d_bvirial_ml,
    float *d_bvirial_dft);

template void ai2pot::nep::find_loss_torch_launcher<double>(
    double *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    double e_weight,
    double f_weight,
    double v_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3],
    double *d_bvirial_ml,
    double *d_bvirial_dft);


// 3. find_ef_loss_backward
template void ai2pot::nep::find_ef_loss_backward_torch_launcher<float>(
    float *d_bloss_der2coeffs,
    float *d_bloss_der2w0,
    float *d_bloss_der2b0,
    float *d_bloss_der2w1,
    float *d_bloss_der2type_bias,
    float e_weight,
    float f_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3],
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
    float rmax_radial,
    float rmax_angular,
    float *d_q_scaler);

template void ai2pot::nep::find_ef_loss_backward_torch_launcher<double>(
    double *d_bloss_der2coeffs,
    double *d_bloss_der2w0,
    double *d_bloss_der2b0,
    double *d_bloss_der2w1,
    double *d_bloss_der2type_bias,
    double e_weight,
    double f_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3],
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
    double rmax_radial,
    double rmax_angular,
    double *d_q_scaler);

// 4. find_loss_backward
template void ai2pot::nep::find_loss_backward_torch_launcher<float>(
    float *d_bloss_der2coeffs,
    float *d_bloss_der2w0,
    float *d_bloss_der2b0,
    float *d_bloss_der2w1,
    float *d_bloss_der2type_bias,
    float e_weight,
    float f_weight,
    float v_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3],
    float *d_bvirial_ml,
    float *d_bvirial_dft,
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
    float rmax_radial,
    float rmax_angular,
    float *d_q_scaler);

template void ai2pot::nep::find_loss_backward_torch_launcher<double>(
    double *d_bloss_der2coeffs,
    double *d_bloss_der2w0,
    double *d_bloss_der2b0,
    double *d_bloss_der2w1,
    double *d_bloss_der2type_bias,
    double e_weight,
    double f_weight,
    double v_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3],
    double *d_bvirial_ml,
    double *d_bvirial_dft,
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
    double rmax_radial,
    double rmax_angular,
    double *d_q_scaler);

// 5.1. find_num_real_atoms_in_batch
void ai2pot::nep::find_num_real_atoms_in_batch_torch_launcher(
    int *d_num_real_atoms_in_batch_ptr,
    int batch_size,
    int *d_binum);


// 5.2. find_e_se
template void ai2pot::nep::find_e_se_torch_launcher(
    float *d_e_se_ptr,
    int batch_size,
    int *d_binum,
    float *d_betot_ml,
    float *d_betot_dft);

template void ai2pot::nep::find_e_se_torch_launcher(
    double *d_e_se_ptr,
    int batch_size,
    int *d_binum,
    double *d_betot_ml,
    double *d_betot_dft);


// 5.3. find_f_se
template void ai2pot::nep::find_f_se_torch_launcher(
    float *d_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3]);

template void ai2pot::nep::find_f_se_torch_launcher(
    double *d_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3]);


// 5.4. find_v_se
template void ai2pot::nep::find_v_se_torch_launcher(
    float *d_v_se_ptr,
    int batch_size,
    float *d_bvirial_ml,
    float *d_bvirial_dft);

template void ai2pot::nep::find_v_se_torch_launcher(
    double *d_v_se_ptr,
    int batch_size,
    double *d_bvirial_ml,
    double *d_bvirial_dft);
