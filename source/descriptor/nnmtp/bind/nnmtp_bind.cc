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
#include "../include/nnMtpOp.h"


TORCH_LIBRARY(nnmtp, m) {
    m.def(
        "mtpParamOp",
        [](int64_t mtp_level)
        {
            assert(mtp_level <= INT_MAX);
            return ai2pot::nnmtp::MtpParamOp((int)mtp_level);
        }
    );  // mtpParamOp

    m.def(
        "nnMtpToEFLossOp",
        [](double e_weight,
           double f_weight,
           const at::Tensor& betot_dft_tensor,
           const at::Tensor& bforce_dft_tensor,
           int64_t chebyshev_size,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& w0_tensor,
           const at::Tensor& w1_tensor,
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

            return ai2pot::nnmtp::NNMtpToEFLossOp(
                e_weight,
                f_weight,
                betot_dft_tensor,
                bforce_dft_tensor,
                (int)chebyshev_size,
                coeffs_tensor,
                w0_tensor,
                w1_tensor,
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
    );  // nnMtpToEFLossOp


    m.def(
        "nnMtpToLossOp",
        [](double e_weight,
           double f_weight,
           double v_weight,
           const at::Tensor& betot_dft_tensor,
           const at::Tensor& bforce_dft_tensor,
           const at::Tensor& bvirial_dft_tensor,
           int64_t chebyshev_size,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& w0_tensor,
           const at::Tensor& w1_tensor,
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

            return ai2pot::nnmtp::NNMtpToLossOp(
                e_weight,
                f_weight,
                v_weight,
                betot_dft_tensor,
                bforce_dft_tensor,
                bvirial_dft_tensor,
                (int)chebyshev_size,
                coeffs_tensor,
                w0_tensor,
                w1_tensor,
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
    );  // nnMtpToLossOp


    m.def(
        "nnMtpToEFOp",
        [](int64_t chebyshev_size,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& w0_tensor,
           const at::Tensor& w1_tensor,
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

            return ai2pot::nnmtp::NNMtpToEFOp(
                (int)chebyshev_size,
                coeffs_tensor,
                w0_tensor,
                w1_tensor,
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
    );  // nnMtpToEFOp


    m.def(
        "nnMtpToEFVOp",
        [](int64_t chebyshev_size,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& w0_tensor,
           const at::Tensor& w1_tensor,
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

            return ai2pot::nnmtp::NNMtpToEFVOp(
                (int)chebyshev_size,
                coeffs_tensor,
                w0_tensor,
                w1_tensor,
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
    );  // nnMtpToEFVOp


    m.def(
        "nnMtpToDescriptorsOp",
        [](int64_t chebyshev_size,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& w0_tensor,
           const at::Tensor& w1_tensor,
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
           double rmin)
        {
            assert(chebyshev_size < INT_MAX);
            assert(alpha_moments_count < INT_MAX);
            assert(nmus < INT_MAX);
            assert(nghost < INT_MAX);

            return ai2pot::nnmtp::NNMtpToDescriptorsOp(
                (int)chebyshev_size,
                coeffs_tensor,
                w0_tensor,
                w1_tensor,
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
                rmin);
        }
    );
};
