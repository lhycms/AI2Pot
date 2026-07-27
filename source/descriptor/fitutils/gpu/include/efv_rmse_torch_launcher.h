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

#ifndef AI2POT_EFV_RMSE_TORCH_LAUNCHER_H
#define AI2POT_EFV_RMSE_TORCH_LAUNCHER_H

namespace ai2pot {
namespace fitutils {

// 5.1. find_num_real_atoms_in_batch
void find_num_real_atoms_in_batch_torch_launcher(
    int *d_num_real_atoms_in_batch_ptr,
    int batch_size,
    int *d_binum);

// 5.2. find_e_se
template <typename CoordType>
void find_e_se_torch_launcher(
    CoordType *d_e_se_ptr,
    int batch_size,
    int *d_binum,
    CoordType *d_betot_ml,
    CoordType *d_betot_dft);

// 5.3. find_f_se
template <typename CoordType>
void find_f_se_torch_launcher(
    CoordType *d_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    CoordType (*d_bforce_ml)[3],
    CoordType (*d_bforce_dft)[3]);

// 5.4. find_v_se
template <typename CoordType>
void find_v_se_torch_launcher(
    CoordType *d_v_se_ptr,
    int batch_size,
    int *d_binum,
    CoordType *d_bvirial_ml,
    CoordType *d_bvirial_dft);


};  // namespace : fitutils
};  // namespace : ai2pot

#endif