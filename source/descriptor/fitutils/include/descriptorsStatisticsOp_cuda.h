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

#ifndef AI2POT_FITUTILS_DESCRIPTORS_STATISTICS_OP_CUDA_H
#define AI2POT_FITUTILS_DESCRIPTORS_STATISTICS_OP_CUDA_H

#include <ATen/ATen.h>
#include <torch/torch.h>

namespace ai2pot {
namespace fitutils {

class AllTypeDescriptorsStatisticsFunctionCUDA : public torch::autograd::Function<AllTypeDescriptorsStatisticsFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor& binum_tensor,
        const at::Tensor& bdescriptors_tensor);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : AllTypeDescriptorsStatisticsFunctionCUDA


class EachTypeDescriptorsStatisticsFunctionCUDA : public torch::autograd::Function<EachTypeDescriptorsStatisticsFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& btypes_tensor,
        int ntypes,
        const at::Tensor& bdescriptors_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : EachTypeDescriptorsStatisticsFunctionCUDA


class AllTypeDescriptorsMaxminFunctionCUDA : public torch::autograd::Function<AllTypeDescriptorsMaxminFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor& binum_tensor,
        const at::Tensor& bdescriptors_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : AllTypeDescriptorsMaxminFunctionCUDA


torch::autograd::variable_list AllTypeDescriptorsStatisticsOpCUDA(
    const at::Tensor& binum_tensor,
    const at::Tensor& bdescriptors_tensor);


torch::autograd::variable_list EachTypeDescriptorsStatisticsOpCUDA(
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& btypes_tensor,
    int ntypes,
    const at::Tensor& bdescriptors_tensor);

torch::autograd::variable_list AllTypeDescriptorsMaxminOpCUDA(
    const at::Tensor& binum_tensor,
    const at::Tensor& bdescriptors_tensor);



};  // namespace : fitutils
};  // namespace : ai2pot


#endif