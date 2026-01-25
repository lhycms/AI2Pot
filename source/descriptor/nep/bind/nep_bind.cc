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

#include "../include/nepOp.h"


TORCH_LIBRARY(nep, m) {
    m.def(
        "nepToEFOp",
        [](int64_t chebyshev_size,
           int64_t n_radial_basis,
           int64_t n_angular_basis,
           int64_t l_max,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& w0_tensor,
           const at::Tensor& b0_tensor,
           const at::Tensor& w1_tensor,
           const at::Tensor& type_bias_tensor,
           const at::Tensor& binum_tensor,
           const at::Tensor& bilist_tensor,
           const at::Tensor& bnumneigh_tensor,
           const at::Tensor& bfirstneigh_tensor,
           const at::Tensor& brcs_tensor,
           const at::Tensor& btypes_tensor,
           const at::Tensor& type_map_tensor,
           int64_t nghost,
           double rmax_radial,
           double rmax_angular,
           const at::Tensor& q_scaler_tensor,
           double zbl_rmax,
           double zbl_rmin,
           const at::Tensor& zbl_cks_tensor,
           const at::Tensor& zbl_dks_tensor)
        {
            return ai2pot::nep::NepToEFOp(
                (int)chebyshev_size,
                (int)n_radial_basis,
                (int)n_angular_basis,
                (int)l_max,
                coeffs_tensor,
                w0_tensor,
                b0_tensor,
                w1_tensor,
                type_bias_tensor,
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                type_map_tensor,
                (int)nghost,
                rmax_radial,
                rmax_angular,
                q_scaler_tensor,
                zbl_rmax,
                zbl_rmin,
                zbl_cks_tensor,
                zbl_dks_tensor);
        }
    );

    m.def(
        "nepToEFLossOp",
        [](double e_weight,
           double f_weight,
           const at::Tensor& betot_dft_tensor,
           const at::Tensor& bforce_dft_tensor,
           int64_t chebyshev_size,
           int64_t n_radial_basis,
           int64_t n_angular_basis,
           int64_t l_max,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& w0_tensor,
           const at::Tensor& b0_tensor,
           const at::Tensor& w1_tensor,
           const at::Tensor& type_bias_tensor,
           const at::Tensor& binum_tensor,
           const at::Tensor& bilist_tensor,
           const at::Tensor& bnumneigh_tensor,
           const at::Tensor& bfirstneigh_tensor,
           const at::Tensor& brcs_tensor,
           const at::Tensor& btypes_tensor,
           const at::Tensor& type_map_tensor,
           int64_t nghost,
           double rmax_radial,
           double rmax_angular,
           const at::Tensor& q_scaler_tensor,
           double zbl_rmax,
           double zbl_rmin,
           const at::Tensor& zbl_cks_tensor,
           const at::Tensor& zbl_dks_tensor)
        {
            assert(chebyshev_size < INT_MAX);
            assert(n_radial_basis < INT_MAX);
            assert(n_angular_basis < INT_MAX);
            assert(l_max < INT_MAX);
            assert(nghost < INT_MAX);

            return ai2pot::nep::NepToEFLossOp(
                e_weight,
                f_weight,
                betot_dft_tensor,
                bforce_dft_tensor,
                (int)chebyshev_size,
                (int)n_radial_basis,
                (int)n_angular_basis,
                (int)l_max,
                coeffs_tensor,
                w0_tensor,
                b0_tensor,
                w1_tensor,
                type_bias_tensor,
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                type_map_tensor,
                (int)nghost,
                rmax_radial,
                rmax_angular,
                q_scaler_tensor,
                zbl_rmax,
                zbl_rmin,
                zbl_cks_tensor,
                zbl_dks_tensor);
        }
    );


    m.def(
        "nepToDescriptorsOp",
        [](int64_t chebyshev_size,
           int64_t n_radial_basis,
           int64_t n_angular_basis,
           int64_t l_max,
           const at::Tensor& coeffs_tensor,
           const at::Tensor& binum_tensor,
           const at::Tensor& bilist_tensor,
           const at::Tensor& bnumneigh_tensor,
           const at::Tensor& bfirstneigh_tensor,
           const at::Tensor& brcs_tensor,
           const at::Tensor& btypes_tensor,
           const at::Tensor& type_map_tensor,
           int64_t nghost,
           double rmax_radial,
           double rmax_angular)
        {
            return ai2pot::nep::NepToDescriptorsOp(
                (int)chebyshev_size,
                (int)n_radial_basis,
                (int)n_angular_basis,
                (int)l_max,
                coeffs_tensor,
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                type_map_tensor,
                (int)nghost,
                rmax_radial,
                rmax_angular);
        }
    );
};