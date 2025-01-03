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

#include "../include/mtpParamOp.h"
#include "../include/mtpBasisOp.h"


TORCH_LIBRARY(mtpr, m) {
    m.def(
        "mtpParamOp",
        [](int64_t mtp_level)
        {
            assert(mtp_level <= INT_MAX);
            return ai2pot::mtpr::MtpParamOp((int)mtp_level);
        }
    );

    m.def(
        "mtpBasisOp",
        [](const at::Tensor &alpha_index_basic_tensor,
           const at::Tensor &alpha_index_times_tensor,
           const at::Tensor &alpha_moment_mapping_tensor,
           const at::Tensor &num_mus4moms_tensor,
           const at::Tensor &mus4moms_tensor,
           const at::Tensor &nmus_tensor,
           int64_t ntypes,
           int64_t chebyshev_size,
           const at::Tensor &coeffs_tensor,
           const at::Tensor &binum_tensor,
           const at::Tensor &bilist_tensor,
           const at::Tensor &bnumneigh_tensor,
           const at::Tensor &bfirstneigh_tensor,
           const at::Tensor &brcs_tensor,
           const at::Tensor &btypes_tensor,
           int64_t umax_num_neighs,
           double rmax,
           double rmin)
        {
            assert(ntypes <= INT_MAX);
            assert(chebyshev_size <= INT_MAX);
            assert(umax_num_neighs <= INT_MAX);

            return ai2pot::mtpr::MtpBasisOp(
                alpha_index_basic_tensor,
                alpha_index_times_tensor,
                alpha_moment_mapping_tensor,
                num_mus4moms_tensor,
                mus4moms_tensor,
                nmus_tensor,
                (int)ntypes,
                (int)chebyshev_size,
                coeffs_tensor,
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                (int)umax_num_neighs,
                rmax,
                rmin);
        }
    );
}