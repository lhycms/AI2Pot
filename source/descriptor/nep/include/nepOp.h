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

#ifndef AI2POT_NEP_NEP_OP_H
#define AI2POT_NEP_NEP_OP_H

#include <ATen/ATen.h>
#include <torch/torch.h>


namespace ai2pot {
namespace nep {


class NepToEFFunction : public torch::autograd::Function<NepToEFFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int chebyshev_size,
        int n_radial_basis,
        int n_angular_basis,
        int l_max,
        const at::Tensor& coeffs_tensor,
        const at::Tensor& w0_tensor,
        const at::Tensor& w1_tensor,
        const at::Tensor& type_bias_tensor,
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& bnumneigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        const at::Tensor& brcs_tensor,
        const at::Tensor& btypes_tensor,
        const at::Tensor& type_map_tensor,
        int nghost,
        double rmax,
        double rmin,
        const at::Tensor& q_scaler_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : NepToEFFunction


class NepToEFLossFunction : public torch::autograd::Function<NepToEFLossFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        double e_weight,
        double f_weight,
        const at::Tensor& betot_dft_tensor,
        const at::Tensor& bforce_dft_tensor,
        int chebyshev_size,
        int n_radial_basis,
        int n_angular_basis,
        int l_max,
        const at::Tensor& coeffs_tensor,
        const at::Tensor& w0_tensor,
        const at::Tensor& w1_tensor,
        const at::Tensor& type_bias_tensor,
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& bnumneigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        const at::Tensor& brcs_tensor,
        const at::Tensor& btypes_tensor,
        const at::Tensor& type_map_tensor,
        int nghost,
        double rmax,
        double rmin,
        const at::Tensor& q_scaler_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : NepToEFLossFunction


torch::autograd::variable_list NepToEFOp(
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& w1_tensor,
    const at::Tensor& type_bias_tensor,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    const at::Tensor& q_scaler_tensor);


torch::autograd::variable_list NepToEFLossOp(
    double e_weight,
    double f_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& w1_tensor,
    const at::Tensor& type_bias_tensor,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    const at::Tensor& q_scaler_tensor);

};  // namespace : nep
};  // namespace : ai2pot


#endif
