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
#include <stdio.h>

#include "../include/targetStatistics.h"
#include "../include/targetStatisticsOp.h"

namespace ai2pot {
namespace fitutils {

torch::autograd::variable_list TargetStatisticsFunction::forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor& binum_tensor,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor)
{
    // 1. 
    int batch_size = binum_tensor.size(0);
    int natoms_pad = bforce_dft_tensor.size(1);
    int *binum = binum_tensor.data_ptr<int>();

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(binum_tensor.device());
    c10::TensorOptions float_options;

    // 3. 
    at::Tensor natoms_in_batch_tensor;
    at::Tensor ei_mean_tensor;
    at::Tensor ei_M2_tensor;
    at::Tensor force_M2_tensor;

    // 4.
    if (betot_dft_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(betot_dft_tensor.device());
        natoms_in_batch_tensor = torch::tensor(0, int_options);
        ei_mean_tensor = torch::tensor(0, float_options);
        ei_M2_tensor = torch::tensor(0, float_options);
        force_M2_tensor = torch::tensor(0, float_options);

        if (betot_dft_tensor.device() == c10::kCPU) {
            int *natoms_in_batch_ptr = natoms_in_batch_tensor.data_ptr<int>();
            float *ei_mean_ptr = ei_mean_tensor.data_ptr<float>();
            float *ei_M2_ptr = ei_M2_tensor.data_ptr<float>();
            float *force_M2_ptr = force_M2_tensor.data_ptr<float>();
            float *betot_dft = betot_dft_tensor.data_ptr<float>();
            float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();

            TargetStatistics<float>::find_ef_statistics(
                natoms_in_batch_ptr,
                ei_mean_ptr,
                ei_M2_ptr,
                force_M2_ptr,
                batch_size,
                natoms_pad,
                binum,
                betot_dft,
                bforce_dft);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(betot_dft_tensor.device());
        natoms_in_batch_tensor = torch::tensor(0, int_options);
        ei_mean_tensor = torch::tensor(0, float_options);
        ei_M2_tensor = torch::tensor(0, float_options);
        force_M2_tensor = torch::tensor(0, float_options);

        if (betot_dft_tensor.device() == c10::kCPU) {
            int *natoms_in_batch_ptr = natoms_in_batch_tensor.data_ptr<int>();
            double *ei_mean_ptr = ei_mean_tensor.data_ptr<double>();
            double *ei_M2_ptr = ei_M2_tensor.data_ptr<double>();
            double *force_M2_ptr = force_M2_tensor.data_ptr<double>();
            double *betot_dft = betot_dft_tensor.data_ptr<double>();
            double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();

            TargetStatistics<double>::find_ef_statistics(
                natoms_in_batch_ptr,
                ei_mean_ptr,
                ei_M2_ptr,
                force_M2_ptr,
                batch_size,
                natoms_pad,
                binum,
                betot_dft,
                bforce_dft);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    }

    ctx->save_for_backward({
        binum_tensor,
        betot_dft_tensor,
        bforce_dft_tensor});

    return {
        natoms_in_batch_tensor,
        ei_mean_tensor,
        ei_M2_tensor,
        force_M2_tensor};
}


torch::autograd::variable_list TargetStatisticsFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    return {
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
}


torch::autograd::variable_list TargetStatisticsOp(
    const at::Tensor& binum_tensor,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor)
{
    return TargetStatisticsFunction::apply(
        binum_tensor,
        betot_dft_tensor,
        bforce_dft_tensor);
}


};  // namespace : fitutils
};  // namespace : ai2pot