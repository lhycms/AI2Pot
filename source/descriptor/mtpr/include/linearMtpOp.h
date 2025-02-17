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

#ifndef AI2POT_MTPR_LINEAR_MTP_OP_H
#define AI2POT_MTPR_LINEAR_MTP_OP_H

#include <ATen/ATen.h>
#include <torch/torch.h>

namespace ai2pot {
namespace mtpr {

class LinearMtpToLossFunction : public torch::autograd::Function<LinearMtpToLossFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        double e_weight,
        double f_weight,
        double v_weight,
        at::Tensor betot_dft_tensor,
        at::Tensor bforce_dft_tensor,
        at::Tensor bvirial_dft_tensor,
        int chebyshev_size,
        at::Tensor coeffs_tensor,
        at::Tensor linear_coeffs_tensor,
        at::Tensor type_bias_tensor,
        int alpha_moments_count,
        at::Tensor alpha_index_basic,
        at::Tensor alpha_index_times,
        at::Tensor alpha_moment_mapping,
        int nmus,
        at::Tensor binum_tensor,
        at::Tensor bilist_tensor,
        at::Tensor bnumneigh_tensor,
        at::Tensor bfirstneigh_tensor,
        at::Tensor brcs_tensor,
        at::Tensor btypes_tensor,
        int ntypes,
        double rmax,
        double rmin);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : LinearMtpToLoss


torch::autograd::variable_list LinearMtpToLossOp(
    
);


};  // namespace : mtpr 
};  // namespace : ai2pot

#endif