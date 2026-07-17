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

#include "../include/schmidt_orth_op_cpu.h"
#include "../include/schmidt_orth.h"


namespace ai2pot {
namespace mtpr {

torch::autograd::variable_list CoeffsSchmidtOrthFunctionCPU::forward(
    torch::autograd::AutogradContext *ctx,
    int ntypes,
    int nmus,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor)
{
    // 1.
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(coeffs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(coeffs_tensor.scalar_type())
                                        .device(coeffs_tensor.device());

    // 3. 
    at::Tensor orth_coeffs_tensor = at::zeros({num_coeffs}, float_options);

    // 4.
    if (coeffs_tensor.scalar_type() == torch::kFloat32) {
        float *orth_coeffs = orth_coeffs_tensor.data_ptr<float>();
        float *coeffs = coeffs_tensor.data_ptr<float>();

        CoeffsSchmidtOrth<float>::find_orthogonal_basis_set(
            orth_coeffs,
            ntypes,
            nmus,
            chebyshev_size,
            coeffs);
    } else {
        double *orth_coeffs = orth_coeffs_tensor.data_ptr<double>();
        double *coeffs = coeffs_tensor.data_ptr<double>();

        CoeffsSchmidtOrth<double>::find_orthogonal_basis_set(
            orth_coeffs,
            ntypes,
            nmus,
            chebyshev_size,
            coeffs);
    }

    // 5.
    return {orth_coeffs_tensor};
}


torch::autograd::variable_list CoeffsSchmidtOrthFunctionCPU::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    return {
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
}


torch::autograd::variable_list CoeffsSchmidtOrthOpCPU(
    int ntypes,
    int nmus,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor)
{
    return CoeffsSchmidtOrthFunctionCPU::apply(
        ntypes,
        nmus,
        chebyshev_size,
        coeffs_tensor);
}

};  // namespace : mtpr
};  // namespace : ai2pot