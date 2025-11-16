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

#include <torch/torch.h>
#include <torch/extension.h>
#include <climits>
#include <cassert>

#include "../include/targetStatisticsOp.h"
#include "../include/descriptorsStatisticsOp.h"


TORCH_LIBRARY(fitutils, m) {
    m.def(
        "targetStatisticsOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& betot_dft_tensor,
           const at::Tensor& bforce_dft_tensor)
        {
            return ai2pot::fitutils::TargetStatisticsOp(
                binum_tensor,
                betot_dft_tensor,
                bforce_dft_tensor);
        }
    );

    m.def(
        "allTypeDescriptorsStatisticsOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& bdescriptors_tensor)
        {
            return ai2pot::fitutils::AllTypeDescriptorsStatisticsOp(
                binum_tensor,
                bdescriptors_tensor);
        }
    );
}
