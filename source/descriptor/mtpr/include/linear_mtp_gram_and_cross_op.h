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

#ifndef AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_OP_H
#define AI2POT_MTPR_LINEAR_MTP_GRAM_AND_CROSS_OP_H

#include <torch/torch.h>
#include <ATen/ATen.h>

namespace ai2pot {
namespace mtpr {

class LinMatrixLinVectorFunction : public torch::autograd::Function<LinMatrixLinVectorFunction>
{
public:
    static torch::autograd::variable_list forward(
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
        int alpha_moments_count,
        const at::Tensor& alpha_index_basic_tensor,
        const at::Tensor& alpha_index_times_tensor,
        const at::Tensor& alpha_moment_mapping_tensor,
        int nmus,
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& bnumeigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        const at::Tensor& brcs_tensor,
        const at::Tensor& btypes_tensor,
        const at::Tensor& type_map_tensor,
        int nghost,
        double rmax,
        double rmin,
        const at::Tensor& q_scaler_tensor,
        double zbl_rmax,
        double zbl_rmin,
        const at::Tensor& zbl_cks_tensor,
        const at::Tensor& zbl_dks_tensor);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : LinMatrixLinVectorFunction


torch::autograd::variable_list LinMatrixLinVectorOp(
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
    int alpha_moments_count,
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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor);

};  // namespace : mtpr
};  // namespace : ai2pot


#endif