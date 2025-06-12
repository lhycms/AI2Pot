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
#include "../include/linearMtpOp.h"


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

    m.def(
        "linearMtpToLossOp",
        [](double e_weight,
           double f_weight,
           double v_weight,
           const at::Tensor& betot_dft_tensor,
           const at::Tensor& bforce_dft_tensor,
           const at::Tensor& bvirial_dft_tensor,
           int64_t chebyshev_size,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& linear_coeffs_tensor,
           const at::Tensor& type_bias_tensor,
           int64_t alpha_moments_count,
           const at::Tensor& alpha_index_basic_tensor,
           const at::Tensor& alpha_index_times_tensor,
           const at::Tensor& alpha_moment_mapping_tensor,
           int64_t nmus,
           const at::Tensor& binum_tensor,
           const at::Tensor& bilist_tensor,
           const at::Tensor& bnumneigh_tensor,
           const at::Tensor& bfirstneigh_tensor,
           const at::Tensor& brcs_tensor,
           const at::Tensor& btypes_tensor,
           const at::Tensor& type_map_tensor,
           int64_t nghost,
           double rmax,
           double rmin,
           double zbl_rmax,
           double zbl_rmin,
           const at::Tensor& zbl_cks_tensor,
           const at::Tensor& zbl_dks_tensor)
        {   
            assert(chebyshev_size < INT_MAX);
            assert(alpha_moments_count < INT_MAX);
            assert(nmus < INT_MAX);
            assert(nghost < INT_MAX);

            return ai2pot::mtpr::LinearMtpToLossOp(
                e_weight,
                f_weight,
                v_weight,
                betot_dft_tensor,
                bforce_dft_tensor,
                bvirial_dft_tensor,
                (int)chebyshev_size,
                coeffs_tensor,
                linear_coeffs_tensor,
                type_bias_tensor,
                (int)alpha_moments_count,
                alpha_index_basic_tensor,
                alpha_index_times_tensor,
                alpha_moment_mapping_tensor,
                (int)nmus,
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                type_map_tensor,
                (int)nghost,
                rmax,
                rmin,
                zbl_rmax,
                zbl_rmin,
                zbl_cks_tensor,
                zbl_dks_tensor);
        }
    );

    m.def(
        "linearMtpToEFLossOp",
        [](double e_weight,
           double f_weight,
           const at::Tensor& betot_dft_tensor,
           const at::Tensor& bforce_dft_tensor,
           int64_t chebyshev_size,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& linear_coeffs_tensor,
           const at::Tensor& type_bias_tensor,
           int64_t alpha_moments_count,
           const at::Tensor& alpha_index_basic_tensor,
           const at::Tensor& alpha_index_times_tensor,
           const at::Tensor& alpha_moment_mapping_tensor,
           int64_t nmus,
           const at::Tensor& binum_tensor,
           const at::Tensor& bilist_tensor,
           const at::Tensor& bnumneigh_tensor,
           const at::Tensor& bfirstneigh_tensor,
           const at::Tensor& brcs_tensor,
           const at::Tensor& btypes_tensor,
           const at::Tensor& type_map_tensor,
           int64_t nghost,
           double rmax,
           double rmin,
           double zbl_rmax,
           double zbl_rmin,
           const at::Tensor& zbl_cks_tensor,
           const at::Tensor& zbl_dks_tensor)
        {   
            assert(chebyshev_size < INT_MAX);
            assert(alpha_moments_count < INT_MAX);
            assert(nmus < INT_MAX);
            assert(nghost < INT_MAX);

            return ai2pot::mtpr::LinearMtpToEFLossOp(
                e_weight,
                f_weight,
                betot_dft_tensor,
                bforce_dft_tensor,
                (int)chebyshev_size,
                coeffs_tensor,
                linear_coeffs_tensor,
                type_bias_tensor,
                (int)alpha_moments_count,
                alpha_index_basic_tensor,
                alpha_index_times_tensor,
                alpha_moment_mapping_tensor,
                (int)nmus,
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                type_map_tensor,
                (int)nghost,
                rmax,
                rmin,
                zbl_rmax,
                zbl_rmin,
                zbl_cks_tensor,
                zbl_dks_tensor);
        }
    );

    m.def(
        "linearMtpToEFV",
        [](int64_t chebyshev_size,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& linear_coeffs_tensor,
           const at::Tensor& type_bias_tensor,
           int64_t alpha_moments_count,
           const at::Tensor& alpha_index_basic_tensor,
           const at::Tensor& alpha_index_times_tensor,
           const at::Tensor& alpha_moment_mapping_tensor,
           int64_t nmus,
           const at::Tensor& binum_tensor,
           const at::Tensor& bilist_tensor,
           const at::Tensor& bnumneigh_tensor,
           const at::Tensor& bfirstneigh_tensor,
           const at::Tensor& brcs_tensor,
           const at::Tensor& btypes_tensor,
           const at::Tensor& type_map_tensor,
           int64_t nghost,
           double rmax,
           double rmin,
           double zbl_rmax,
           double zbl_rmin,
           const at::Tensor& zbl_cks_tensor,
           const at::Tensor& zbl_dks_tensor)
        {
            assert(chebyshev_size < INT_MAX);
            assert(alpha_moments_count < INT_MAX);
            assert(nmus < INT_MAX);
            assert(nghost < INT_MAX);

            return ai2pot::mtpr::LinearMtpToEFVOp(
                (int)chebyshev_size,
                coeffs_tensor,
                linear_coeffs_tensor,
                type_bias_tensor,
                (int)alpha_moments_count,
                alpha_index_basic_tensor,
                alpha_index_times_tensor,
                alpha_moment_mapping_tensor,
                (int)nmus,
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                type_map_tensor,
                (int)nghost,
                rmax,
                rmin,
                zbl_rmax,
                zbl_rmin,
                zbl_cks_tensor,
                zbl_dks_tensor);
        }
    );
}
