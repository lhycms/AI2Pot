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

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "../include/nnMtp.h"
#include "../include/nnMtpOp.h"


namespace ai2pot {
namespace nnmtp {

torch::autograd::variable_list NNMtpToEFLossFunction::forward(
    torch::autograd::AutogradContext *ctx,
    double e_weight,
    double f_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
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
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    // 1.
    int batch_size = (int)bfirstneigh_tensor.size(0);
    int num_atoms = (int)bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = (int)bfirstneigh_tensor.size(2);
    int ntypes = (int)type_map_tensor.size(0);
    int *type_map = type_map_tensor.data_ptr<int>();
    int num_neurons = (int)w1_tensor.size(0);
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    // 3. 
    at::Tensor bloss_tensor;

    // 4.
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(brcs_tensor.device());
        bloss_tensor = at::zeros({batch_size}, float_options);
        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *w0 = w0_tensor.data_ptr<float>();
        float *w1 = w1_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        for (int bb=0; bb<batch_size; bb++) 
        {
            float loss = bloss_tensor[bb].item<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
            float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<float>::find_ef_loss(
                loss,
                e_weight,
                f_weight,
                etot_dft,
                force_dft,
                chebyshev_size,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moments,
                alpha_moment_mapping,
                nmus,
                inum,
                ilist,
                numneigh,
                firstneigh,
                rcs,
                types,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                zbl_rmax,
                zbl_rmin,
                zbl_cks,
                zbl_dks);
        }
    } else {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat64)
                        .device(brcs_tensor.device());
        bloss_tensor = at::zeros({batch_size}, float_options);
        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *w0 = w0_tensor.data_ptr<double>();
        double *w1 = w1_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        for (int bb=0; bb<batch_size; bb++) 
        {
            double loss = bloss_tensor[bb].item<double>();
            double etot_dft = betot_dft_tensor[bb].item<double>();
            double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<double>::find_ef_loss(
                loss,
                e_weight,
                f_weight,
                etot_dft,
                force_dft,
                chebyshev_size,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moments,
                alpha_moment_mapping,
                nmus,
                inum,
                ilist,
                numneigh,
                firstneigh,
                rcs,
                types,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                zbl_rmax,
                zbl_rmin,
                zbl_cks,
                zbl_dks);
        }
    }


    float_options = c10::TensorOptions()
                    .dtype(torch::kFloat64)
                    .device(brcs_tensor.device());
    ctx->save_for_backward({
        torch::tensor(e_weight, float_options),
        torch::tensor(f_weight, float_options),
        betot_dft_tensor,
        bforce_dft_tensor,
        torch::tensor(chebyshev_size, int_options),
        coeffs_tensor,
        w0_tensor,
        w1_tensor,
        type_bias_tensor,
        torch::tensor(alpha_moments_count, int_options),
        alpha_index_basic_tensor,
        alpha_index_times_tensor,
        alpha_moment_mapping_tensor,
        torch::tensor(nmus, int_options),
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        type_map_tensor,
        torch::tensor(nghost, int_options),
        torch::tensor(rmax, float_options),
        torch::tensor(rmin, float_options),
        torch::tensor(zbl_rmax, float_options),
        torch::tensor(zbl_rmin, float_options),
        zbl_cks_tensor,
        zbl_dks_tensor});

    return {bloss_tensor};
}

torch::autograd::variable_list NNMtpToEFLossFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    
    double e_weight = ctx->get_saved_variables()[0].item<double>();
    double f_weight = ctx->get_saved_variables()[1].item<double>();


    // 1. 

    // 2. 

    // 3. 

    // 4. 


    return {
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
}

};  // namespace : nnmtp
};  // namespace : ai2pot