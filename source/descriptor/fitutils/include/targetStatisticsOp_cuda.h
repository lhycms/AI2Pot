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

#ifndef AI2POT_FITUTILS_TARGET_STATISTICS_OP_CUDA_H
#define AI2POT_FITUTILS_TARGET_STATISTICS_OP_CUDA_H

#include <ATen/ATen.h>
#include <torch/torch.h>

namespace ai2pot {
namespace fitutils {

class TargetStatisticsFunctionCUDA : public torch::autograd::Function<TargetStatisticsFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor& binum_tensor,
        const at::Tensor& betot_dft_tensor,
        const at::Tensor& bforce_dft_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : TargetStatisticsFunctionCUDA


torch::autograd::variable_list TargetStatisticsOpCUDA(
    const at::Tensor& binum_tensor,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor);


};  // namespace : fitutils
};  // namespace : ai2pot

#endif
