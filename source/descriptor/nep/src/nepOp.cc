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

#include "../include/nep.h"
#include "../include/nepOp.h"


namespace ai2pot {
namespace nep {

torch::autograd::variable_list NepToEFLossFunction::forward(
    torch::autograd::AutogradContext *ctx,
    double e_weight,
    double f_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& w1_tensor,
    const at::Tensor& type_bias_tensor,
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
    const at::Tensor& q_scaler_tensor)
{
    // 1. 
    int batch_size = bfirstneigh_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int *type_map = (int*)type_map_tensor.data_ptr<int>();
    int num_neurons = w1_tensor.size(0);
    int num_descriptors = w1_tensor.size(1);

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(brcs_tensor.scalar_type())
                                        .device(brcs_tensor.device());

    // 3. 
    at::Tensor bloss_tensor;
    at::Tensor betot_tensor;
    at::Tensor bforce_tensor;

    // 4. 
    if (brcs_tensor.scalar_type.dtype() == torch::kFloat32) {
        if (brcs_tensor.device() == c10::kCPU) {

        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    } else {
        if (brcs_tensor.device() == c10::kCPU) {

        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ... 
            #endif
        }
    }

    float_options = c10::TensorOptions()
                        .dtype(torch::kFloat64)
                        .device(brcs_tensor.device());
    ctx->save_for_backward({
        torch::tensor(e_weight, float_options),
        torch::tensor(f_weight, float_options),
        betot_dft_tensor,
        bforce_dft_tensor,
        torch::tensor(chebyshev_size, int_options),
        torch::tensor(n_radial_basis, int_options),
        torch::tensor(n_angular_basis, int_options),
        torch::tensor(l_max, int_options),
        coeffs_tensor,
        w0_tensor,
        w1_tensor,
        type_bias_tensor,
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        type_map_tensor,
        torch::tensor(nghost, int_options),
        torch::tensor(rmax, float_options),
        torch::tensor(rmin, float_options)},
        q_scaler_tensor);
    
    return {bloss_tensor};
}


torch::autograd::variable_list NepToEFLossFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    // 1. 

    // 2. 

    // 3. 

    // 4. 
}



torch::autograd::variable_list NepToEFLossOp(
    double e_weight,
    double f_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& w1_tensor,
    const at::Tensor& type_bias_tensor,
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
    const at::Tensor& q_scaler_tensor)
{
    return NepToEFLossFunction::apply(
        e_weight,
        f_weight,
        betot_dft_tensor,
        bforce_dft_tensor,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        coeffs_tensor,
        w0_tensor,
        w1_tensor,
        type_bias_tensor,
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        type_map_tensor,
        nghost,
        rmax,
        rmin,
        q_scaler_tensor);
}


};  // namespace : nep
};  // namespace : ai2pot
