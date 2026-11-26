/*
    Copyright 2026 Hanyu Liu

    SPDX-License-Identifier: LGPL-3.0-only

    This file is part of AI2Pot.

    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, version 3 of the License.

    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with AI2Pot. If not, see <https://www.gnu.org/licenses/>.
*/

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include <string>


class AI2PotNodeCalculator
{
public:
    AI2PotNodeCalculator(
        std::string pt_path,
        torch::Device device);

    torch::autograd::variable_list predict_ef(
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& bnumneigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        const at::Tensor& brcs_tensor,
        const at::Tensor& btypes_tensor,
        const at::Tensor& bnghost_tensor);

    torch::autograd::variable_list predict_efv(
        const at::Tensor& binum_tensor,
        const at::Tensor& nilist_tensor,
        const at::Tensor& bnumneigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        const at::Tensor& brcs_tensor,
        const at::Tensor& btypes_tensor,
        const at::Tensor& bnghost_tensor);

    torch::autograd::variable_list predict_gamma_local(
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& bnumneigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        const at::Tensor& brcs_tensor,
        const at::Tensor& btypes_tensor,
        const at::Tensor& bnghost_tensor,
        const at::Tensor& maxvol_matrix_inv);
    
    torch::autograd::variable_list predict_gamma_gloal(
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& bnumneigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        const at::Tensor& brcs_tensor,
        const at::Tensor& btypes_tensor,
        const at::Tensor& bnghost_tensor,
        const at::Tensor& maxvol_matrix_inv);

    at::Tensor load_matrix_matrix_inv(std::string maxvaol_matrix_inv_path);

    
    torch::jit::script::Module model;
    std::string potential_type;
};  // class : AI2PotCalculator
