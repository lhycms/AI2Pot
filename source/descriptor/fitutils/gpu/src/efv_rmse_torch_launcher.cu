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

#include "../include/efv_rmse.cuh"
#include "../include/efv_rmse_torch_launcher.cuh"


// 1. find_num_real_atoms_in_batch
void ai2pot::fitutils::find_num_real_atoms_in_batch_torch_launcher(
    int *d_num_real_atoms_in_batch_ptr,
    int batch_size,
    int *d_binum);


// 2. find_e_se
template void ai2pot::fitutils::find_e_se_torch_launcher(
    float *d_e_se_ptr,
    int batch_size,
    int *d_binum,
    float *d_betot_ml,
    float *d_betot_dft);

template void ai2pot::fitutils::find_e_se_torch_launcher(
    double *d_e_se_ptr,
    int batch_size,
    int *d_binum,
    double *d_betot_ml,
    double *d_betot_dft);


// 3. find_f_se
template void ai2pot::fitutils::find_f_se_torch_launcher(
    float *d_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3]);

template void ai2pot::fitutils::find_f_se_torch_launcher(
    double *d_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3]);


// 4. find_v_se
template void ai2pot::fitutils::find_v_se_torch_launcher(
    float *d_v_se_ptr,
    int batch_size,
    int *d_binum,
    float *d_bvirial_ml,
    float *d_bvirial_dft);

template void ai2pot::fitutils::find_v_se_torch_launcher(
    double *d_v_se_ptr,
    int batch_size,
    int *d_binum,
    double *d_bvirial_ml,
    double *d_bvirial_dft);