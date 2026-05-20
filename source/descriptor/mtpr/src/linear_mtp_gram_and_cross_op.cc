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

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "../include/linear_mtp_gram_and_cross_op.h"
#include "../include/linear_mtp_gram_and_cross.h"
#include "../../correction/include/zbl.h"


#if defined(USE_CUDA) or defined(__INTELLISENSE__)
#include "../gpu/include/linear_mtp_gram_and_cross_torch_launcher.h"
#include "../../correction/gpu/include/zbl_torch_launcher.h"

namespace ai2pot {
namespace mtpr {

extern template void find_lin_matrix_lin_vector_torch_launcher<float>(
    float *d_lin_matrix,
    float *d_lin_vector,
    float e_weight,
    float f_weight,
    float v_weight,
    float *d_betot_residual,
    float (*d_bforce_residual)[3],
    float *d_bvirial_residual,
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

extern template void find_lin_matrix_lin_vector_torch_launcher<double>(
    double *d_lin_matrix,
    double *d_lin_vector,
    double e_weight,
    double f_weight,
    double v_weight,
    double *d_betot_residual,
    double (*d_bforce_residual)[3],
    double *d_bvirial_residual,
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
};  // namespace : mtpr
};  // namespace : ai2pot


namespace ai2pot {
namespace correction {
extern template void ai2pot::correction::correct_zbl_efv_torch_launcher<float>(
    float *d_betot_ptr,
    float *d_bforce,
    float *d_bvirial,
    float rmax,
    float rmin,
    float *d_cks,
    float *d_dks,
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
    int nghost);

extern template void ai2pot::correction::correct_zbl_efv_torch_launcher<double>(
    double *d_betot_ptr,
    double *d_bforce,
    double *d_bvirial,
    double rmax,
    double rmin,
    double *d_cks,
    double *d_dks,
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
    int nghost);
};  // namespace : mtpr
};  // namespace : correction



namespace ai2pot {
namespace mtpr {

torch::autograd::variable_list LinMatrixLinVectorFunction::forward(
    torch::autograd::AutogradContext *ctx,
    double e_weight,
    double f_weight,
    double v_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    const at::Tensor& bvirial_dft_tensor,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& linear_coeffs_tensor,
    const at::Tensor& type_bias_tensor,
    const at::Tensor& alpha_index_basic_tensor,
    const at::Tensor& alpha_index_times_tensor,
    const at::Tensor& alpha_moment_mapping_tensor,
    int nmus,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    const at::Tensor &q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor &zbl_cks_tensor,
    const at::Tensor &zbl_dks_tensor)
{
    //
    int batch_size = bfirstneigh_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();

    int ntypes = type_map_tensor.size(0);
    int *type_map = (int*)type_map_tensor.data_ptr<int>();
    int *binum = (int*)binum_tensor.data_ptr<int>();
    int *bilist = (int*)bilist_tensor.data_ptr<int>();
    int *bnumneigh = (int*)bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = (int*)bfirstneigh_tensor.data_ptr<int>();
    int *btypes = (int*)btypes_tensor.data_ptr<int>();

    int num_parameters = alpha_scalar_moments + ntypes;

    //
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(brcs_tensor.scalar_type())
                                        .device(brcs_tensor.device());
    
    //
    at::Tensor lin_matrix_tensor = at::zeros({num_parameters, num_parameters}, float_options);
    at::Tensor lin_vector_tensor = at::zeros({num_parameters}, float_options);
    at::Tensor betot_zbl_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_zbl_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);
    at::Tensor bvirial_zbl_tensor = at::zeros({batch_size, 9}, float_options);
    
    //
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        // TODO

        if (brcs_tensor.device() == c10::kCPU) {

        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // TODO
            #endif
        }
    } else {
        // TODO

        if (brcs_tensor.device() == c10::kCPU) {

        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // TODO
            #endif
        }
    }

    //

    return {lin_matrix_tensor,
            lin_vector_tensor}
}

};  // namespace : mtpr
};  // namespace : ai2pot

#endif