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
