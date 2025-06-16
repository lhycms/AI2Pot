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

#ifndef AI2POT_LINEAR_MTP_LOSS_TORCH_LAUNCHER_H
#define AI2POT_LINEAR_MTP_LOSS_TORCH_LAUNCHER_H


namespace ai2pot {
namespace mtpr {

template <typename CoordType>
void find_loss_torch_launcher(
    CoordType *d_loss_ptr,
    int inum,
    int *d_ilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*d_force_ml)[3],
    CoordType (*d_force_dft)[3],
    CoordType *d_virial_ml,
    CoordType *d_virial_dft);


template <typename CoordType>
void find_ef_loss_torch_launcher(
    CoordType *d_loss_ptr,
    int inum,
    int *d_ilist,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*d_force_ml)[3],
    CoordType (*d_force_dft)[3]);


template <typename CoordType>
void find_loss_backward_torch_launcher(
    CoordType *d_loss_der2coeffs,
    CoordType *d_loss_der2linear_coeffs,
    CoordType *d_loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*d_force_ml)[3],
    CoordType (*d_force_dft)[3],
    CoordType *d_virial_ml,
    CoordType *d_virial_dft,
    int chebyshev_size,
    CoordType *d_coeffs,
    CoordType *d_linear_coeffs,
    CoordType *d_type_bias,
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
    CoordType (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);


template <typename CoordType>
void find_ef_loss_backward_torch_launcher(
    CoordType *d_loss_der2coeffs,
    CoordType *d_loss_der2linear_coeffs,
    CoordType *d_loss_der2type_bias,
    CoordType e_weight,
    CoordType f_weight,
    CoordType etot_ml,
    CoordType etot_dft,
    CoordType (*d_force_ml)[3],
    CoordType (*d_force_dft)[3],
    int chebyshev_size,
    CoordType *d_coeffs,
    CoordType *d_linear_coeffs,
    CoordType *d_type_bias,
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
    CoordType (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin);


};  // namespace : mtpr
};  // namespace : ai2pot


#endif