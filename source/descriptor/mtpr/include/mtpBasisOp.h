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

#ifndef AI2POT_MTPR_MTP_BASIS_OP_H
#define AI2POT_MTPR_MTP_BASIS_OP_H
#include <torch/torch.h>

namespace ai2pot {
namespace mtpr {

class MtpBasisFunction : public torch::autograd::Function<MtpBasisFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &alpha_index_basic_tensor,
        const at::Tensor &alpha_index_times_tensor,
        const at::Tensor &alpha_moment_mapping_tensor,
        const at::Tensor &num_mus4moms_tensor,
        const at::Tensor &mus4moms_tensor,
        const at::Tensor &nmus_tensor,
        int ntypes,
        int chebyshev_size,
        double scaling,
        const at::Tensor &coeffs_tensor,
        const at::Tensor &binum_tensor,
        const at::Tensor &bilist_tensor,
        const at::Tensor &bnumneigh_tensor,
        const at::Tensor &bfirstneigh_tensor,
        const at::Tensor &brcs_tensor,
        const at::Tensor &btypes_tensor,
        int umax_num_neighs,
        double rmax,
        double rmin);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : MtpBasisFunction


torch::autograd::variable_list MtpBasisOp(
    const at::Tensor &alpha_index_basic_tensor,
    const at::Tensor &alpha_index_times_tensor,
    const at::Tensor &alpha_moment_mapping_tensor,
    const at::Tensor &num_mus4moms_tensor,
    const at::Tensor &mus4moms_tensor,
    const at::Tensor &nmus_tensor,
    int ntypes,
    int chebyshev_size,
    double scaling,
    const at::Tensor &coeffs_tensor,
    const at::Tensor &binum_tensor,
    const at::Tensor &bilist_tensor,
    const at::Tensor &bnumneigh_tensor,
    const at::Tensor &bfirstneigh_tensor,
    const at::Tensor &brcs_tensor,
    const at::Tensor &btypes_tensor,
    int umax_num_neighs,
    double rmax,
    double rmin);

};  // namespace : mtpr
};  // namespace : ai2pot


#endif