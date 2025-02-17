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

#include "../include/linearMtp.h"
#include "../include/linearMtpOp.h"


namespace ai2pot {
namespace mtpr {

torch::autograd::variable_list LinearMtpToLossFunction::forward(
    torch::autograd::AutogradContext *ctx,
    double e_weight,
    double f_weight,
    double v_weight,
    at::Tensor betot_dft_tensor,
    at::Tensor bforce_dft_tensor,
    at::Tensor bvirial_dft_tensor,
    int chebyshev_size,
    at::Tensor coeffs_tensor,
    at::Tensor linear_coeffs_tensor,
    at::Tensor type_bias_tensor,
    int alpha_moments_count,
    at::Tensor alpha_index_basic_tensor,
    at::Tensor alpha_index_times_tensor,
    at::Tensor alpha_moment_mapping_tensor,
    int nmus,
    at::Tensor binum_tensor,
    at::Tensor bilist_tensor,
    at::Tensor bnumneigh_tensor,
    at::Tensor bfirstneigh_tensor,
    at::Tensor brcs_tensor,
    at::Tensor btypes_tensor,
    int ntypes,
    int nghost,
    double rmax,
    double rmin)
{
    int nbatches = (int)betot_dft_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic.size(0);
    int alpha_index_times_count = (int)alpha_index_times.size(0);
    int alpha_scalar_moment = (int)alpha_moment_mapping.size(0);
    int umax_num_neighs = (int)bfirstneigh_tensor.size(2);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bloss_tensor;

    if (brcs_tensor.dtype == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(brcs_tensor.device());
        bloss_tensor = at::zeros({nbatches}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            float *loss = bloss_tensor[bb].data_ptr<float>();
            float *etot_dft = betot_dft_tensor[bb].data_ptr<float>();
            float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
            float *virial_dft = bvirial_dft_tensor[bb].data_ptr<float>();
            float *coeffs = coeffs_tensor.data_ptr<float>();
            float *linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
            float *type_bias = type_bias_tensor.data_ptr<float>();
            int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
            int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
            int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = btypes_tensor[bb].data_ptr<int>();

            LinearMtp<float>::find_loss(
                (*loss),
                e_weight,
                f_weight,
                v_weight,
                (*etot_dft),
                force_dft,
                virial_dft,
                chebyshev_size,
                coeffs,
                linear_coeffs,
                type_bias,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moment,
                alpha_moment_mapping,
                nmus,
                inum,
                ilist,
                numneigh,
                firstneigh,
                rcs,
                types,
                ntypes,
                umax_num_neighs,
                nghost,
                rmax,
                rmin);
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(brcs_tensor.device());
        bloss_tensor = at::zeros({nbatches}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            double *loss = bloss_tensor[bb].data_ptr<double>();
            double *etot_dft = betot_dft_tensor[bb].data_ptr<double>();
            double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
            double *virial_dft = bvirial_dft_tensor[bb].data_ptr<double>();
            double *coeffs = coeffs_tensor.data_ptr<double>();
            double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
            double *type_bias = type_bias_tensor.data_ptr<double>();
            int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
            int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
            int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = btypes_tensor[bb].data_ptr<int>();

            LinearMtp<double>::find_loss(
                (*loss),
                e_weight,
                f_weight,
                v_weight,
                (*etot_dft),
                force_dft,
                virial_dft,
                chebyshev_size,
                coeffs,
                linear_coeffs,
                type_bias,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moment,
                alpha_moment_mapping,
                nmus,
                inum,
                ilist,
                numneigh,
                firstneigh,
                rcs,
                types,
                ntypes,
                umax_num_neighs,
                nghost,
                rmax,
                rmin);
        }
    }


    ctx->save_for_backward({e_weight, 
                            f_weight, 
                            v_weight,
                            betot_dft_tensor, 
                            bforce_dft_tensor,
                            bvirial_dft_tensor,
                            chebyshev_size,
                            coeffs_tensor,
                            linear_coeffs_tensor,
                            type_bias_tensor,
                            alpha_moments_count,
                            alpha_index_basic_tensor,
                            alpha_index_times_tensor,
                            alpha_moment_mapping_tensor,
                            nmus,
                            binum_tensor,
                            bilist_tensor,
                            bnumneigh_tensor,
                            bfirstneigh_tensor,
                            brcs_tensor,
                            btypes_tensor,
                            ntypes,
                            nghost,
                            rmax,
                            rmin});

    return {bloss_tensor};
}


torch::autograd::variable_list LinearMtpToLossFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    
    double e_weight = ctx->save_for_backward()[0];
    double f_weight = ctx->save_for_backward()[1];
    double v_weight = ctx->save_for_backward()[2];
    at::Tensor betot_dft_tensor = ctx->save_for_backward()[3];
    at::Tensor bforce_dft_tensor = ctx->save_for_backward()[4];
    at::Tensor bvirial_dft_tensor = ctx->save_for_backward()[5];
    int chebyshev_size = ctx->save_for_backward()[6];
    at::Tensor coeffs_tensor = ctx->save_for_backward()[7];
    at::Tensor linear_coeffs_tensor = ctx->save_for_backward()[8];
    at::Tensor type_bias_tensor = ctx->save_for_backward()[9];
    int alpha_moments_count = ctx->save_for_backward()[10];
    at::Tensor alpha_index_basic_tensor = ctx->save_for_backward()[11];
    at::Tensor alpha_index_times_tensor = ctx->save_for_backward()[12];
    at::Tensor alpha_moment_mapping_tensor = ctx->save_for_backward()[13];
    int nmus = ctx->save_for_backward()[14];
    at::Tensor binum_tensor = ctx->save_for_backward()[15];
    at::Tensor bilist_tensor = ctx->save_for_backward()[16];
    at::Tensor bnumneigh_tensor = ctx->save_for_backward()[17];
    at::Tensor bfirstneigh_tensor = ctx->save_for_backward()[18];
    at::Tensor brcs_tensor = ctx->save_for_backward()[19];
    at::Tensor btypes_tensor = ctx->save_for_backward()[20];
    int ntypes = ctx->save_for_backward()[21];
    int nghost = ctx->save_for_backward()[22];
    double rmax = ctx->save_for_backward()[23];
    double rmin = ctx->save_for_backward()[24];
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    int num_linear_coeffs = linear_coeffs_tensor.size(0);

    int nbatches = (int)bgrad_output_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moment = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neighs = (int)bfirstneigh_tensor.size(2);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bloss_der2coeffs_tensor;
    at::Tensor bloss_der2linear_coeffs_tensor;
    at::Tensor bloss_der2type_bias_tensor;
    
    // To do
    if (bgrad_output_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);


    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);


    }

    return {};
}


};  // namespace : mtpr
};  // namespace : ai2pot
