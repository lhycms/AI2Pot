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

#include "../include/descriptorsStatisticsOp_cuda.h"
#include "../include/descriptorsStatistics.h"


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


// 3. find_all_type_descriptors_maxmin_torch_launcher()
extern template void find_all_type_descriptors_maxmin_torch_launcher<float>(
    float *descriptors_max,
    float *descriptors_min,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    float *bdescriptors);

extern template void find_all_type_descriptors_maxmin_torch_launcher<double>(
    double *descriptors_max,
    double *descriptors_min,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    double *bdescriptors);

};  // namespace : fitutils
};  // namespace : ai2pot


namespace ai2pot {
namespace fitutils {

torch::autograd::variable_list AllTypeDescriptorsStatisticsFunctionCUDA::forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor& binum_tensor,
    const at::Tensor& bdescriptors_tensor)
{
    // 1. 
    int batch_size = bdescriptors_tensor.size(0);
    int natoms_pad = bdescriptors_tensor.size(1);
    int descriptor_dim = bdescriptors_tensor.size(2);
    int *binum = binum_tensor.data_ptr<int>();

    // 2.
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(bdescriptors_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(bdescriptors_tensor.scalar_type())
                                        .device(bdescriptors_tensor.device());
    
    // 3.
    at::Tensor natoms_in_batch_tensor = torch::tensor(0, int_options);
    at::Tensor descriptors_mean_tensor = torch::zeros({descriptor_dim}, float_options);
    at::Tensor descriptors_M2_tensor = torch::zeros({descriptor_dim}, float_options);

    // 4. 
    if (bdescriptors_tensor.scalar_type() == torch::kFloat32) {
        int *natoms_in_batch_ptr = natoms_in_batch_tensor.data_ptr<int>();
        float *descriptors_mean = descriptors_mean_tensor.data_ptr<float>();
        float *descriptors_M2 = descriptors_M2_tensor.data_ptr<float>();
        float *bdescriptors = bdescriptors_tensor.data_ptr<float>();

        find_all_type_descriptors_statistics_torch_launcher(
                natoms_in_batch_ptr,
                descriptors_mean,
                descriptors_M2,
                batch_size,
                natoms_pad,
                descriptor_dim,
                binum,
                bdescriptors);
    } else {
        int *natoms_in_batch_ptr = natoms_in_batch_tensor.data_ptr<int>();
        double *descriptors_mean = descriptors_mean_tensor.data_ptr<double>();
        double *descriptors_M2 = descriptors_M2_tensor.data_ptr<double>();
        double *bdescriptors = bdescriptors_tensor.data_ptr<double>();

        find_all_type_descriptors_statistics_torch_launcher(
                natoms_in_batch_ptr,
                descriptors_mean,
                descriptors_M2,
                batch_size,
                natoms_pad,
                descriptor_dim,
                binum,
                bdescriptors);
    }
    
    return {natoms_in_batch_tensor,
            descriptors_mean_tensor,
            descriptors_M2_tensor};
}


torch::autograd::variable_list AllTypeDescriptorsStatisticsFunctionCUDA::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    return {at::Tensor(),
            at::Tensor()};
}


torch::autograd::variable_list EachTypeDescriptorsStatisticsFunctionCUDA::forward(
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
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(bdescriptors_tensor.scalar_type())
                                        .device(bdescriptors_tensor.device());
    
    // 3. 
    at::Tensor natoms_in_batch_tensor = at::zeros({ntypes}, int_options);
    at::Tensor descriptors_mean_tensor = at::zeros({ntypes, descriptor_dim}, float_options);
    at::Tensor descriptors_M2_tensor = at::zeros({ntypes, descriptor_dim}, float_options);

    // 4.
    if (bdescriptors_tensor.scalar_type() == torch::kFloat32) {
        // TODO
    } else {
        // TODO
    }

    return {natoms_in_batch_tensor,
            descriptors_mean_tensor,
            descriptors_M2_tensor};
}


torch::autograd::variable_list EachTypeDescriptorsStatisticsFunctionCUDA::backward(
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


torch::autograd::variable_list AllTypeDescriptorsMaxminFunctionCUDA::forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor& binum_tensor,
    const at::Tensor& bdescriptors_tensor)
{
    // 1.
    int batch_size = bdescriptors_tensor.size(0);
    int natoms_pad = bdescriptors_tensor.size(1);
    int descriptor_dim = bdescriptors_tensor.size(2);
    int *binum = binum_tensor.data_ptr<int>();

    // 2.
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(bdescriptors_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(bdescriptors_tensor.scalar_type())
                                        .device(bdescriptors_tensor.device());
    
    // 3.
    at::Tensor descriptors_max_tensor = at::zeros({descriptor_dim}, float_options);
    at::Tensor descriptors_min_tensor = at::zeros({descriptor_dim}, float_options);
    
    // 4. 
    if (bdescriptors_tensor.scalar_type() == torch::kFloat32) {
        float *bdescriptors = bdescriptors_tensor.data_ptr<float>();
        float *descriptors_max = descriptors_max_tensor.data_ptr<float>();
        float *descriptors_min = descriptors_min_tensor.data_ptr<float>();

        find_all_type_descriptors_maxmin_torch_launcher(
            descriptors_max,
            descriptors_min,
            batch_size,
            natoms_pad,
            descriptor_dim,
            binum,
            bdescriptors);
    } else {
        double *bdescriptors = bdescriptors_tensor.data_ptr<double>();
        double *descriptors_max = descriptors_max_tensor.data_ptr<double>();
        double *descriptors_min = descriptors_min_tensor.data_ptr<double>();

        find_all_type_descriptors_maxmin_torch_launcher(
            descriptors_max,
            descriptors_min,
            batch_size,
            natoms_pad,
            descriptor_dim,
            binum,
            bdescriptors);
    }

    return {descriptors_max_tensor,
            descriptors_min_tensor};
}


torch::autograd::variable_list AllTypeDescriptorsMaxminFunctionCUDA::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    return {at::Tensor(),
            at::Tensor()};
}


torch::autograd::variable_list AllTypeDescriptorsStatisticsOpCUDA(
    const at::Tensor& binum_tensor,
    const at::Tensor& bdescriptors_tensor)
{
    return AllTypeDescriptorsStatisticsFunctionCUDA::apply(
        binum_tensor,
        bdescriptors_tensor);
}


torch::autograd::variable_list EachTypeDescriptorsStatisticsOpCUDA(
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& btypes_tensor,
    int ntypes,
    const at::Tensor& bdescriptors_tensor)
{
    return EachTypeDescriptorsStatisticsFunctionCUDA::apply(
        binum_tensor,
        bilist_tensor,
        btypes_tensor,
        ntypes,
        bdescriptors_tensor);
}


torch::autograd::variable_list AllTypeDescriptorsMaxminOpCUDA(
    const at::Tensor& binum_tensor,
    const at::Tensor& bdescriptors_tensor)
{
    return AllTypeDescriptorsMaxminFunctionCUDA::apply(
        binum_tensor,
        bdescriptors_tensor);
}



};  // namespace : fitutils
};  // namespace : ai2pot