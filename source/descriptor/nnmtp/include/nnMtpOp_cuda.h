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

#ifndef AI2POT_NNMTP_NNMTP_OP_CUDA_H
#define AI2POT_NNMTP_NNMTP_OP_CUDA_H

#include <ATen/ATen.h>
#include <torch/torch.h>


namespace ai2pot {
namespace nnmtp {

class NNMtpToEFLossFunctionCUDA : public torch::autograd::Function<NNMtpToEFLossFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        double e_weight,
        double f_weight,
        const at::Tensor& betot_dft_tensor,
        const at::Tensor& bforce_dft_tensor,
        int chebyshev_size,
        const at::Tensor& coeffs_tensor,
        const at::Tensor& w0_tensor,
        const at::Tensor& b0_tensor,
        const at::Tensor& w1_tensor,
        const at::Tensor& type_bias_tensor,
        int alpha_moments_count,
        const at::Tensor& alpha_index_basic_tensor,
        const at::Tensor& alpha_index_times_tensor,
        const at::Tensor& alpha_moment_mapping_tensor,
        int nmus,
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
        const at::Tensor& q_scaler_tensor,
        double zbl_rmax,
        double zbl_rmin,
        const at::Tensor& zbl_cks_tensor,
        const at::Tensor& zbl_dks_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : NNMtpToEFLossFunctionCUDA


class NNMtpToLossFunctionCUDA : public torch::autograd::Function<NNMtpToLossFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        double e_weight,
        double f_weight,
        double v_weight,
        const at::Tensor& betot_dft_tensor,
        const at::Tensor& bforce_dft_tensor,
        const at::Tensor& bvirial_dft_tensor,
        int chebyshev_size,
        const at::Tensor& coeffs_tensor,
        const at::Tensor& w0_tensor,
        const at::Tensor& b0_tensor,
        const at::Tensor& w1_tensor,
        const at::Tensor& type_bias_tensor,
        int alpha_moments_count,
        const at::Tensor& alpha_index_basic_tensor,
        const at::Tensor& alpha_index_times_tensor,
        const at::Tensor& alpha_moment_mapping_tensor,
        int nmus,
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
        const at::Tensor& q_scaler_tensor,
        double zbl_rmax,
        double zbl_rmin,
        const at::Tensor& zbl_cks_tensor,
        const at::Tensor& zbl_dks_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : NNMtpToLossFunctionCUDA


class NNMtpToEFFunctionCUDA : public torch::autograd::Function<NNMtpToEFFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int chebyshev_size,
        const at::Tensor& coeffs_tensor,
        const at::Tensor& w0_tensor,
        const at::Tensor& b0_tensor,
        const at::Tensor& w1_tensor,
        const at::Tensor& type_bias_tensor,
        int alpha_moments_count,
        const at::Tensor& alpha_index_basic_tensor,
        const at::Tensor& alpha_index_times_tensor,
        const at::Tensor& alpha_moment_mapping_tensor,
        int nmus,
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
        const at::Tensor& q_scaler_tensor,
        double zbl_rmax,
        double zbl_rmin,
        const at::Tensor& zbl_cks_tensor,
        const at::Tensor& zbl_dks_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : NNMtpToEFFunctionCUDA


class NNMtpToEFVFunctionCUDA : public torch::autograd::Function<NNMtpToEFVFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int chebyshev_size,
        const at::Tensor& coeffs_tensor,
        const at::Tensor& w0_tensor,
        const at::Tensor& b0_tensor,
        const at::Tensor& w1_tensor,
        const at::Tensor& type_bias_tensor,
        int alpha_moments_count,
        const at::Tensor& alpha_index_basic_tensor,
        const at::Tensor& alpha_index_times_tensor,
        const at::Tensor& alpha_moment_mapping_tensor,
        int nmus,
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
        const at::Tensor& q_scaler_tensor,
        double zbl_rmax,
        double zbl_rmin,
        const at::Tensor& zbl_cks_tensor,
        const at::Tensor& zbl_dks_tensor);
};  // class : NNMtpToEFVFunctionCUDA


class NNMtpToDescriptorsFunctionCUDA : public torch::autograd::Function<NNMtpToDescriptorsFunctionCUDA>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int chebyshev_size,
        const at::Tensor& coeffs_tensor,
        int alpha_moments_count,
        const at::Tensor& alpha_index_basic_tensor,
        const at::Tensor& alpha_index_times_tensor,
        const at::Tensor& alpha_moment_mapping_tensor,
        int nmus,
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& bnumneigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        const at::Tensor& brcs_tensor,
        const at::Tensor& btypes_tensor,
        int nghost,
        double rmax,
        double rmin);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : NNMtpToDescriptorsFunctionCUDA


torch::autograd::variable_list NNMtpToEFLossOpCUDA(
    double e_weight,
    double f_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& b0_tensor,
    const at::Tensor& w1_tensor,
    const at::Tensor& type_bias_tensor,
    int alpha_moments_count,
    const at::Tensor& alpha_index_basic_tensor,
    const at::Tensor& alpha_index_times_tensor,
    const at::Tensor& alpha_moment_mapping_tensor,
    int nmus,
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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor);

torch::autograd::variable_list NNMtpToLossOpCUDA(
    double e_weight,
    double f_weight,
    double v_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    const at::Tensor& bvirial_dft_tensor,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& b0_tensor,
    const at::Tensor& w1_tensor,
    const at::Tensor& type_bias_tensor,
    int alpha_moments_count,
    const at::Tensor& alpha_index_basic_tensor,
    const at::Tensor& alpha_index_times_tensor,
    const at::Tensor& alpha_moment_mapping_tensor,
    int nmus,
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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor);

torch::autograd::variable_list NNMtpToEFOpCUDA(
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& b0_tensor,
    const at::Tensor& w1_tensor,
    const at::Tensor& type_bias_tensor,
    int alpha_moments_count,
    const at::Tensor& alpha_index_basic_tensor,
    const at::Tensor& alpha_index_times_tensor,
    const at::Tensor& alpha_moment_mapping_tensor,
    int nmus,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btype_tensor,
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor);

torch::autograd::variable_list NNMtpToEFVOpCUDA(
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& b0_tensor,
    const at::Tensor& w1_tensor,
    const at::Tensor& type_bias_tensor,
    int alpha_moments_count,
    const at::Tensor& alpha_index_basic_tensor,
    const at::Tensor& alpha_index_times_tensor,
    const at::Tensor& alpha_moment_mapping_tensor,
    int nmus,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btype_tensor,
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor);

torch::autograd::variable_list NNMtpToDescriptorsOpCUDA(
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    int alpha_moments_count,
    const at::Tensor& alpha_index_basic_tensor,
    const at::Tensor& alpha_index_times_tensor,
    const at::Tensor& alpha_moment_mapping_tensor,
    int nmus,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin);


};  // namespace : nnmtp
};  // namespace : ai2pot


#endif