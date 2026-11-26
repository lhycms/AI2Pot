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

#include <torch/serialize.h>

#include "../include/node_calculator.h"


AI2PotNodeCalculator::AI2PotNodeCalculator(
    std::string pt_path,
    torch::Device device)
{
    // Step 1.
    this->model = torch::jit::load(pt_path);
    this->model.to(device);
    this->model.eval();

    // Step 2.
    if (model.hasattr("mtp_level") && model.hasattr("num_neurons"))
        this->potential_type = "AI2POT-NNMTP";
    else if (model.hasattr("mtp_level"))
        this->potential_type = "AI2POT-MTP";
    else if (model.hasattr("rmax_radial"))
        this->potential_type = "AI2POT-NEP";
    else
        this->potential_type = "AI2Pot-custom";
}


torch::autograd::variable_list AI2PotNodeCalculator::predict_ef(
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& bnghost_tensor)
{
    c10::intrusive_ptr<c10::ivalue::Tuple> output = this->model.get_method("predict_ef")(
        {
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            bnghost_tensor
        }).toTuple();
    
    at::Tensor betot_tensor = output->elements()[0].toTensor();
    at::Tensor bforce_tensor = output->elements()[1].toTensor();

    return {betot_tensor, bforce_tensor};
}


torch::autograd::variable_list AI2PotNodeCalculator::predict_efv(
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& bnghost_tensor)
{
    c10::intrusive_ptr<c10::ivalue::Tuple> output = this->model.get_method("predict_efv")(
        {
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            bnghost_tensor
        }).toTuple();
    
    at::Tensor betot_tensor = output->elements()[0].toTensor();
    at::Tensor bforce_tensor = output->elements()[1].toTensor();
    at::Tensor bvirial_tensor = output->elements()[2].toTensor();

    return {betot_tensor, bforce_tensor, bvirial_tensor};
}


torch::autograd::variable_list AI2PotNodeCalculator::predict_gamma_gloal(
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& bnghost_tensor,
    const at::Tensor& maxvol_matrix_inv)
{
    at::Tensor bgamma_structure_tensor = this->model.get_method("predict_gamma_global")(
        {
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            bnghost_tensor,
            maxvol_matrix_inv
        }).toTensor();
    
    return {bgamma_structure_tensor};
}


torch::autograd::variable_list AI2PotNodeCalculator::predict_gamma_local(
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& bnghost_tensor,
    const at::Tensor& maxvol_matrix_inv)
{
    c10::intrusive_ptr<c10::ivalue::Tuple> output = this->model.get_method("predict_gamma_local")(
        {
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            bnghost_tensor,
            maxvol_matrix_inv
        }).toTuple();
    
    at::Tensor bgamma_structures_tensor = output->elements()[0].toTensor();
    at::Tensor bgamma_atoms_tensor = output->elements()[1].toTensor();

    return {bgamma_structures_tensor, bgamma_atoms_tensor};
}


at::Tensor AI2PotNodeCalculator::load_matrix_matrix_inv(std::string maxvol_matrix_inv_path)
{
    std::ifstream input(maxvol_matrix_inv_path, std::ios::binary);

    std::vector<char> buffer(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    auto data = torch::pickle_load(buffer);
    return data.toTensor();
}
