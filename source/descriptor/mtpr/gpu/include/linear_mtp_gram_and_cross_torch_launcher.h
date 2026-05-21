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

#ifndef AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_TORCH_LAUNCHER_H
#define AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_TORCH_LAUNCHER_H

namespace ai2pot {
namespace mtpr {

template <typename CoordType>
void find_efv_components_torch_launcher(
    CoordType *d_benergy_components,
    CoordType *d_bforce_components,
    CoordType *d_bvirial_components,
    int chebyshev_size,
    CoordType scaling,
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    CoordType (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *d_q_scaler);


template <typename CoordType>
void find_lin_matrix_lin_vector_torch_launcher(
    CoordType *d_lin_matrix,
    CoordType *d_lin_vector,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType *d_betot_residual,
    CoordType (*d_bforce_residual)[3],
    CoordType *d_bvirial_residual,
    CoordType *d_benergy_components,
    CoordType *d_bforce_components,
    CoordType *d_bvirial_components,
    int alpha_scalar_moments,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int ntypes);


template <typename CoordType>
void mirror_lin_matrix_torch_launcher(
    CoordType *d_lin_matrix,
    int num_parameters);


};  // namespace : mtpr
};  // namespace : ai2pot

#endif
