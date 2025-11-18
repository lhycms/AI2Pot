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

#include "../include/descriptorsStatisticsOp.h"
#include "../include/descriptorsStatistics.h"


#if defined(USE_CUDA) or defined(__INTELLISENSE__)
#include "../gpu/include/descriptorsStatistics_torch_launcher.h"

namespace ai2pot {
namespace fitutils {

// 1. find_all_type_descriptors_statistics_torch_launcher()
extern template void find_all_type_descriptors_statistics_torch_launcher<float>(
    int *d_natoms_in_batch_ptr,
    float *d_descriptors_mean,
    float *d_descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *d_binum,
    float *d_bdescriptors);

extern template void find_all_type_descriptors_statistics_torch_launcher<double>(
    int *d_natoms_in_batch_ptr,
    double *d_descriptors_mean,
    double *d_descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *d_binum,
    double *d_bdescriptors);


// 2. find_each_type_descriptors_statistics_torch_launcher()

};  // namespace : fitutils
};  // namespace : ai2pot
#endif


namespace ai2pot {
namespace fitutils {

torch::autograd::variable_list AllTypeDescriptorsStatisticsFunction::forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor& binum_tensor,
    const at::Tensor& bdescriptors_tensor)
{
    // 1. 
    int batch_size = binum_tensor.size(0);
    int natoms_pad = bdescriptors_tensor.size(1);
    int descriptor_dim = bdescriptors_tensor.size(2);
    int *binum = binum_tensor.data_ptr<int>();

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(bdescriptors_tensor.device());
    c10::TensorOptions float_options;

    // 3. 
    at::Tensor natoms_in_batch_tensor;
    at::Tensor descriptors_mean_tensor;
    at::Tensor descriptors_M2_tensor;

    // 4. 
    if (bdescriptors_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(bdescriptors_tensor.device());
        natoms_in_batch_tensor = torch::tensor(0, int_options);
        descriptors_mean_tensor = torch::zeros({descriptor_dim}, float_options);
        descriptors_M2_tensor = torch::zeros({descriptor_dim}, float_options);

        int *natoms_in_batch_ptr = natoms_in_batch_tensor.data_ptr<int>();
        float *descriptors_mean = descriptors_mean_tensor.data_ptr<float>();
        float *descriptors_M2 = descriptors_M2_tensor.data_ptr<float>();
        float *bdescriptors = bdescriptors_tensor.data_ptr<float>();

        if (bdescriptors_tensor.device() == c10::kCPU) {
            AllTypeDescriptorsStatistics<float>::find_descriptors_statistics(
                natoms_in_batch_ptr,
                descriptors_mean,
                descriptors_M2,
                batch_size,
                natoms_pad,
                descriptor_dim,
                binum,
                bdescriptors);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            ai2pot::fitutils::find_all_type_descriptors_statistics_torch_launcher(
                natoms_in_batch_ptr,
                descriptors_mean,
                descriptors_M2,
                batch_size,
                natoms_pad,
                descriptor_dim,
                binum,
                bdescriptors);
            #endif
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(bdescriptors_tensor.device());
        natoms_in_batch_tensor = torch::tensor(0, int_options);
        descriptors_mean_tensor = torch::zeros({descriptor_dim}, float_options);
        descriptors_M2_tensor = torch::zeros({descriptor_dim}, float_options);

        int *natoms_in_batch_ptr = natoms_in_batch_tensor.data_ptr<int>();
        double *descriptors_mean = descriptors_mean_tensor.data_ptr<double>();
        double *descriptors_M2 = descriptors_M2_tensor.data_ptr<double>();
        double *bdescriptors = bdescriptors_tensor.data_ptr<double>();

        if (bdescriptors_tensor.device() == c10::kCPU) {
            AllTypeDescriptorsStatistics<double>::find_descriptors_statistics(
                natoms_in_batch_ptr,
                descriptors_mean,
                descriptors_M2,
                batch_size,
                natoms_pad,
                descriptor_dim,
                binum,
                bdescriptors);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            ai2pot::fitutils::find_all_type_descriptors_statistics_torch_launcher(
                natoms_in_batch_ptr,
                descriptors_mean,
                descriptors_M2,
                batch_size,
                natoms_pad,
                descriptor_dim,
                binum,
                bdescriptors);
            #endif
        }
    }

    ctx->save_for_backward({
        binum_tensor,
        bdescriptors_tensor});

    return {natoms_in_batch_tensor,
            descriptors_mean_tensor,
            descriptors_M2_tensor};
}


torch::autograd::variable_list AllTypeDescriptorsStatisticsFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    return {at::Tensor(),
            at::Tensor()};
}


torch::autograd::variable_list EachTypeDescriptorsStatisticsFunction::forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& btypes_tensor,
    int ntypes,
    const at::Tensor& bdescriptors_tensor)
{
    // 1. 
    int batch_size = bdescriptors_tensor.size(0);
    int natoms_pad = bdescriptors_tensor.size(1);
    int descriptor_dim = bdescriptors_tensor.size(2);
    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    // 2.
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(bdescriptors_tensor.device());
    c10::TensorOptions float_options;

    // 3. 
    at::Tensor natoms_in_batch_tensor;
    at::Tensor descriptors_mean_tensor;
    at::Tensor descriptors_M2_tensor;

    // 4.
    if (bdescriptors_tensor.scalar_type()== torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(bdescriptors_tensor.device());
        
        natoms_in_batch_tensor = at::zeros({ntypes}, int_options);
        descriptors_mean_tensor = at::zeros({ntypes, descriptor_dim}, float_options);
        descriptors_M2_tensor = at::zeros({ntypes, descriptor_dim}, float_options);

        int *natoms_in_batch = natoms_in_batch_tensor.data_ptr<int>();
        float *descriptors_mean = descriptors_mean_tensor.data_ptr<float>();
        float *descriptors_M2 = descriptors_M2_tensor.data_ptr<float>();
        float *bdescriptors = bdescriptors_tensor.data_ptr<float>();

        if (bdescriptors_tensor.device() == c10::kCPU) {
            ai2pot::fitutils::EachTypeDescriptorsStatistics<float>::find_descriptors_statistics(
                natoms_in_batch,
                descriptors_mean,
                descriptors_M2,
                batch_size,
                natoms_pad,
                descriptor_dim,
                binum,
                bilist,
                btypes,
                ntypes,
                bdescriptors);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(bdescriptors_tensor.device());

        natoms_in_batch_tensor = at::zeros({ntypes}, int_options);
        descriptors_mean_tensor = at::zeros({ntypes, descriptor_dim}, float_options);
        descriptors_M2_tensor = at::zeros({ntypes, descriptor_dim}, float_options);

        int *natoms_in_batch = natoms_in_batch_tensor.data_ptr<int>();
        double *descriptors_mean = descriptors_mean_tensor.data_ptr<double>();
        double *descriptors_M2 = descriptors_M2_tensor.data_ptr<double>();
        double *bdescriptors = bdescriptors_tensor.data_ptr<double>();

        if (bdescriptors_tensor.device() == c10::kCPU) {
            ai2pot::fitutils::EachTypeDescriptorsStatistics<double>::find_descriptors_statistics(
                natoms_in_batch,
                descriptors_mean,
                descriptors_M2,
                batch_size,
                natoms_pad,
                descriptor_dim,
                binum,
                bilist,
                btypes,
                ntypes,
                bdescriptors);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    }

    // 5. 
    ctx->save_for_backward(
        {binum_tensor,
         bilist_tensor,
         btypes_tensor,
         torch::tensor(ntypes, int_options),
         bdescriptors_tensor});
         
    return {natoms_in_batch_tensor,
            descriptors_mean_tensor,
            descriptors_M2_tensor};
}


torch::autograd::variable_list EachTypeDescriptorsStatisticsFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    return {
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
}


torch::autograd::variable_list AllTypeDescriptorsStatisticsOp(
    const at::Tensor& binum_tensor,
    const at::Tensor& bdescriptors_tensor)
{
    return ai2pot::fitutils::AllTypeDescriptorsStatisticsFunction::apply(
        binum_tensor,
        bdescriptors_tensor);
}


torch::autograd::variable_list EachTypeDescriptorsStatisticsOp(
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& btypes_tensor,
    int ntypes,
    const at::Tensor& bdescriptors_tensor)
{
    return EachTypeDescriptorsStatisticsFunction::apply(
        binum_tensor,
        bilist_tensor,
        btypes_tensor,
        ntypes,
        bdescriptors_tensor);
}

};  // namespace : fituitls
};  // namespace : ai2pot