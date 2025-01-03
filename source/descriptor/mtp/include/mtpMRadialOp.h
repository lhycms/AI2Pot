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

#ifndef AI2POT_MTPM_RADIAL_OP
#define AI2POT_MTPM_RADIAL_OP
#include <torch/torch.h>
#include "./mtpMRadial.h"

namespace ai2pot {
namespace mtp {

class MtpQFunction : public torch::autograd::Function<MtpQFunction>
{
public:
    // Returns: shape = [nneigh, size]
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        int64_t size,
        const at::Tensor& rcuts_tensor,        // rs_tensor = [rcut, rcut_smooth]
        const at::Tensor& rcs_tensor);         // .shape = [nneigh, 3];
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs);
};  // class MtpQFunction


torch::autograd::variable_list MtpQOp(
    int64_t size,
    const at::Tensor& rcuts_tensor,
    const at::Tensor& distances_tensor);

};  // namespace : mtp
};  // namespace : ai2pot

#endif
