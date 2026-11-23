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


private:
    torch::jit::script::Module model;
    std::string potential_type;
};  // class : AI2PotCalculator


AI2PotNodeCalculator::AI2PotNodeCalculator(
    std::string pt_path,
    torch::Device device)
{
    // Step 1.
    model = torch::jit::load(pt_path);
    model.to(device);
    model.eval();

    // Step 2.
    if (model.hasattr("mtp_level") && model.hasattr("num_neurons"))
        potential_type = "AI2POT-NNMTP";
    else if (model.hasattr("mtp_level"))
        potential_type = "AI2POT-MTP";
    else if (model.hasattr("rmax_radial"))
        potential_type = "AI2POT-NEP";
    else
        potential_type = "AI2Pot-custom";

}
