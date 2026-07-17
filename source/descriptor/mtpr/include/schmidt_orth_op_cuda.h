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

#ifndef AI2POT_MTPR_SCHMIDT_ORTH_OP_CUDA_H
#define AI2POT_MTPR_SCHMIDT_ORTH_OP_CUDA_H

#include <torch/torch.h>
#include <ATen/ATen.h>

namespace ai2pot {
namespace mtpr {

class CoeffsSchmidtOrthFunctionCUDA : public torch::autograd::Function<CoeffsSchmidtOrthFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int ntypes,
        int nmus,
        int chebyshev_size,
        const at::Tensor& coeffs_tensor);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : CoeffsSchmidtOrthFunctionCUDA


torch::autograd::variable_list CoeffsSchmidtOrthOpCUDA(
    int ntypes,
    int nmus,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor);

};  // namespace : mtpr
};  // namespace : ai2pot

#endif