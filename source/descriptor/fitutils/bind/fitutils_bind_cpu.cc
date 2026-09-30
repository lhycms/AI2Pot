/*
    Copyright 2025 Hanyu Liu

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
#include <torch/extension.h>
#include <climits>
#include <cassert>

#include "../include/targetStatisticsOp_cpu.h"
#include "../include/descriptorsStatisticsOp_cpu.h"


TORCH_LIBRARY(fitutils, m) {
    m.def(
        "targetStatisticsOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& betot_dft_tensor,
           const at::Tensor& bforce_dft_tensor)
        {
            return ai2pot::fitutils::TargetStatisticsOpCPU(
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
            return ai2pot::fitutils::AllTypeDescriptorsStatisticsOpCPU(
                binum_tensor,
                bdescriptors_tensor);
        }
    );

    m.def(
        "eachTypeDescriptorsStatisticsOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& bilist_tensor,
           const at::Tensor& btypes_tensor,
           int64_t ntypes,
           const at::Tensor& bdescriptors_tensor)
        {
            assert(ntypes < INT_MAX);

            return ai2pot::fitutils::EachTypeDescriptorsStatisticsOpCPU(
                binum_tensor,
                bilist_tensor,
                btypes_tensor,
                (int)ntypes,
                bdescriptors_tensor);
        }
    );

    m.def(
        "allTypeDescriptorsMaxminOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& bdescriptors_tensor)
        {
            return ai2pot::fitutils::AllTypeDescriptorsMaxminOpCPU(
                binum_tensor,
                bdescriptors_tensor);
        }
    );
}



TORCH_LIBRARY_IMPL(fitutils, CPU, m) {
    m.impl(
        "targetStatisticsOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& betot_dft_tensor,
           const at::Tensor& bforce_dft_tensor)
        {
            return ai2pot::fitutils::TargetStatisticsOpCPU(
                binum_tensor,
                betot_dft_tensor,
                bforce_dft_tensor);
        }
    );

    m.impl(
        "allTypeDescriptorsStatisticsOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& bdescriptors_tensor)
        {
            return ai2pot::fitutils::AllTypeDescriptorsStatisticsOpCPU(
                binum_tensor,
                bdescriptors_tensor);
        }
    );

    m.impl(
        "eachTypeDescriptorsStatisticsOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& bilist_tensor,
           const at::Tensor& btypes_tensor,
           int64_t ntypes,
           const at::Tensor& bdescriptors_tensor)
        {
            assert(ntypes < INT_MAX);

            return ai2pot::fitutils::EachTypeDescriptorsStatisticsOpCPU(
                binum_tensor,
                bilist_tensor,
                btypes_tensor,
                (int)ntypes,
                bdescriptors_tensor);
        }
    );

    m.impl(
        "allTypeDescriptorsMaxminOp",
        [](const at::Tensor& binum_tensor,
           const at::Tensor& bdescriptors_tensor)
        {
            return ai2pot::fitutils::AllTypeDescriptorsMaxminOpCPU(
                binum_tensor,
                bdescriptors_tensor);
        }
    );
}
