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
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    const at::Tensor& bvirial_dft_tensor,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& linear_coeffs_tensor,
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
    int ntypes,
    int nghost,
    double rmax,
    double rmin)
{
    int nbatches = (int)betot_dft_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moment = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neighs = (int)bfirstneigh_tensor.size(2);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bloss_tensor;

    if (brcs_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(brcs_tensor.device());
        bloss_tensor = at::zeros({nbatches}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            float *loss = bloss_tensor[bb].data_ptr<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
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
                etot_dft,
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
            double etot_dft = betot_dft_tensor[bb].item<double>();
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
                etot_dft,
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

    float_options = c10::TensorOptions()
                        .dtype(torch::kFloat64)
                        .device(brcs_tensor.device());
    ctx->save_for_backward({at::tensor(e_weight, float_options), 
                            at::tensor(f_weight, float_options), 
                            at::tensor(v_weight, float_options),
                            betot_dft_tensor, 
                            bforce_dft_tensor,
                            bvirial_dft_tensor,
                            at::tensor(chebyshev_size, int_options),
                            coeffs_tensor,
                            linear_coeffs_tensor,
                            type_bias_tensor,
                            at::tensor(alpha_moments_count, int_options),
                            alpha_index_basic_tensor,
                            alpha_index_times_tensor,
                            alpha_moment_mapping_tensor,
                            at::tensor(nmus, int_options),
                            binum_tensor,
                            bilist_tensor,
                            bnumneigh_tensor,
                            bfirstneigh_tensor,
                            brcs_tensor,
                            btypes_tensor,
                            at::tensor(ntypes, int_options),
                            at::tensor(nghost, int_options),
                            at::tensor(rmax, float_options),
                            at::tensor(rmin, float_options)
                            });

    if (brcs_tensor.requires_grad())
        bloss_tensor.requires_grad_(true);
    return {bloss_tensor};
}


torch::autograd::variable_list LinearMtpToLossFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    
    double e_weight = ctx->get_saved_variables()[0].item<double>();
    double f_weight = ctx->get_saved_variables()[1].item<double>();
    double v_weight = ctx->get_saved_variables()[2].item<double>();
    at::Tensor betot_dft_tensor = ctx->get_saved_variables()[3];
    at::Tensor bforce_dft_tensor = ctx->get_saved_variables()[4];
    at::Tensor bvirial_dft_tensor = ctx->get_saved_variables()[5];
    int chebyshev_size = ctx->get_saved_variables()[6].item<int>();
    at::Tensor coeffs_tensor = ctx->get_saved_variables()[7];
    at::Tensor linear_coeffs_tensor = ctx->get_saved_variables()[8];
    at::Tensor type_bias_tensor = ctx->get_saved_variables()[9];
    int alpha_moments_count = ctx->get_saved_variables()[10].item<int>();
    at::Tensor alpha_index_basic_tensor = ctx->get_saved_variables()[11];
    at::Tensor alpha_index_times_tensor = ctx->get_saved_variables()[12];
    at::Tensor alpha_moment_mapping_tensor = ctx->get_saved_variables()[13];
    int nmus = ctx->get_saved_variables()[14].item<int>();
    at::Tensor binum_tensor = ctx->get_saved_variables()[15];
    at::Tensor bilist_tensor = ctx->get_saved_variables()[16];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[17];
    at::Tensor bfirstneigh_tensor = ctx->get_saved_variables()[18];
    at::Tensor brcs_tensor = ctx->get_saved_variables()[19];
    at::Tensor btypes_tensor = ctx->get_saved_variables()[20];
    int ntypes = ctx->get_saved_variables()[21].item<int>();
    int nghost = ctx->get_saved_variables()[22].item<int>();
    double rmax = ctx->get_saved_variables()[23].item<double>();
    double rmin = ctx->get_saved_variables()[24].item<double>();
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    int num_linear_coeffs = (int)linear_coeffs_tensor.size(0);

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
    
    if (bgrad_output_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            float *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<float>();
            float *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<float>();
            float *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
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


            LinearMtp<float>::find_loss_backward(
                loss_der2coeffs,
                loss_der2linear_coeffs,
                loss_der2type_bias,
                e_weight,
                f_weight,
                v_weight,
                etot_dft,
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
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            double *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<double>();
            double *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<double>();
            double *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<double>();
            double etot_dft = betot_dft_tensor[bb].item<double>();
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

            LinearMtp<double>::find_loss_backward(
                loss_der2coeffs,
                loss_der2linear_coeffs,
                loss_der2type_bias,
                e_weight,
                f_weight,
                v_weight,
                etot_dft,
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

    bloss_der2coeffs_tensor.requires_grad_(true);
    bloss_der2linear_coeffs_tensor.requires_grad_(true);
    bloss_der2type_bias_tensor.requires_grad_(true);

    return {at::Tensor(), 
            at::Tensor(),
            at::Tensor(),
            at::Tensor(), 
            at::Tensor(), 
            at::Tensor(),
            at::Tensor(),
            bloss_der2coeffs_tensor,
            bloss_der2linear_coeffs_tensor,
            bloss_der2type_bias_tensor,
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
            at::Tensor()
            };
}


torch::autograd::variable_list LinearMtpToEFLossFunction::forward(
    torch::autograd::AutogradContext *ctx,
    double e_weight,
    double f_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& linear_coeffs_tensor,
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
    int ntypes,
    int nghost,
    double rmax,
    double rmin)
{
    int nbatches = (int)betot_dft_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moment = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neighs = (int)bfirstneigh_tensor.size(2);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bloss_tensor;

    if (brcs_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(brcs_tensor.device());
        bloss_tensor = at::zeros({nbatches}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            float *loss = bloss_tensor[bb].data_ptr<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
            float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
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

            LinearMtp<float>::find_ef_loss(
                (*loss),
                e_weight,
                f_weight,
                etot_dft,
                force_dft,
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
            double etot_dft = betot_dft_tensor[bb].item<double>();
            double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
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

            LinearMtp<double>::find_ef_loss(
                (*loss),
                e_weight,
                f_weight,
                v_weight,
                etot_dft,
                force_dft,
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

    float_options = c10::TensorOptions()
                        .dtype(torch::kFloat64)
                        .device(brcs_tensor.device());
    ctx->save_for_backward({at::tensor(e_weight, float_options), 
                            at::tensor(f_weight, float_options), 
                            betot_dft_tensor, 
                            bforce_dft_tensor,
                            at::tensor(chebyshev_size, int_options),
                            coeffs_tensor,
                            linear_coeffs_tensor,
                            type_bias_tensor,
                            at::tensor(alpha_moments_count, int_options),
                            alpha_index_basic_tensor,
                            alpha_index_times_tensor,
                            alpha_moment_mapping_tensor,
                            at::tensor(nmus, int_options),
                            binum_tensor,
                            bilist_tensor,
                            bnumneigh_tensor,
                            bfirstneigh_tensor,
                            brcs_tensor,
                            btypes_tensor,
                            at::tensor(ntypes, int_options),
                            at::tensor(nghost, int_options),
                            at::tensor(rmax, float_options),
                            at::tensor(rmin, float_options)
                            });

    if (brcs_tensor.requires_grad())
        bloss_tensor.requires_grad_(true);
    return {bloss_tensor};
}


torch::autograd::variable_list LinearMtpToEFLossFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    
    double e_weight = ctx->get_saved_variables()[0].item<double>();
    double f_weight = ctx->get_saved_variables()[1].item<double>();
    at::Tensor betot_dft_tensor = ctx->get_saved_variables()[2];
    at::Tensor bforce_dft_tensor = ctx->get_saved_variables()[3];
    int chebyshev_size = ctx->get_saved_variables()[4].item<int>();
    at::Tensor coeffs_tensor = ctx->get_saved_variables()[5];
    at::Tensor linear_coeffs_tensor = ctx->get_saved_variables()[6];
    at::Tensor type_bias_tensor = ctx->get_saved_variables()[7];
    int alpha_moments_count = ctx->get_saved_variables()[8].item<int>();
    at::Tensor alpha_index_basic_tensor = ctx->get_saved_variables()[9];
    at::Tensor alpha_index_times_tensor = ctx->get_saved_variables()[10];
    at::Tensor alpha_moment_mapping_tensor = ctx->get_saved_variables()[11];
    int nmus = ctx->get_saved_variables()[12].item<int>();
    at::Tensor binum_tensor = ctx->get_saved_variables()[13];
    at::Tensor bilist_tensor = ctx->get_saved_variables()[14];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[15];
    at::Tensor bfirstneigh_tensor = ctx->get_saved_variables()[16];
    at::Tensor brcs_tensor = ctx->get_saved_variables()[17];
    at::Tensor btypes_tensor = ctx->get_saved_variables()[18];
    int ntypes = ctx->get_saved_variables()[19].item<int>();
    int nghost = ctx->get_saved_variables()[20].item<int>();
    double rmax = ctx->get_saved_variables()[21].item<double>();
    double rmin = ctx->get_saved_variables()[22].item<double>();
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    int num_linear_coeffs = (int)linear_coeffs_tensor.size(0);

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
    
    if (bgrad_output_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            float *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<float>();
            float *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<float>();
            float *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
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


            LinearMtp<float>::find_ef_loss_backward(
                loss_der2coeffs,
                loss_der2linear_coeffs,
                loss_der2type_bias,
                e_weight,
                f_weight,
                etot_dft,
                force_dft,
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
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            double *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<double>();
            double *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<double>();
            double *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<double>();
            double etot_dft = betot_dft_tensor[bb].item<double>();
            double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
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

            LinearMtp<double>::find_ef_loss_backward(
                loss_der2coeffs,
                loss_der2linear_coeffs,
                loss_der2type_bias,
                e_weight,
                f_weight,
                etot_dft,
                force_dft,
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

    bloss_der2coeffs_tensor.requires_grad_(true);
    bloss_der2linear_coeffs_tensor.requires_grad_(true);
    bloss_der2type_bias_tensor.requires_grad_(true);

    return {at::Tensor(),
            at::Tensor(), 
            at::Tensor(), 
            at::Tensor(),
            at::Tensor(),
            bloss_der2coeffs_tensor,
            bloss_der2linear_coeffs_tensor,
            bloss_der2type_bias_tensor,
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
            at::Tensor()
            };
}


torch::autograd::variable_list LinearMtpToLossOp(
    double e_weight,
    double f_weight,
    double v_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    const at::Tensor& bvirial_dft_tensor,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& linear_coeffs_tensor,
    const at::Tensor& type_bias_tensor,
    int alpha_moments_count,
    const at::Tensor& alpha_index_basic,
    const at::Tensor& alpha_index_times,
    const at::Tensor& alpha_moment_mapping,
    int nmus,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    int ntypes,
    int nghost,
    double rmax,
    double rmin)
{
    return LinearMtpToLossFunction::apply(
        e_weight,
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
        alpha_index_basic,
        alpha_index_times,
        alpha_moment_mapping,
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
        rmin);
}


torch::autograd::variable_list LinearMtpToEFLossOp(
    double e_weight,
    double f_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    int chebyshev_size,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& linear_coeffs_tensor,
    const at::Tensor& type_bias_tensor,
    int alpha_moments_count,
    const at::Tensor& alpha_index_basic,
    const at::Tensor& alpha_index_times,
    const at::Tensor& alpha_moment_mapping,
    int nmus,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    int ntypes,
    int nghost,
    double rmax,
    double rmin)
{
    return LinearMtpToEFLossFunction::apply(
        e_weight,
        f_weight,
        betot_dft_tensor,
        bforce_dft_tensor,
        chebyshev_size,
        coeffs_tensor,
        linear_coeffs_tensor,
        type_bias_tensor,
        alpha_moments_count,
        alpha_index_basic,
        alpha_index_times,
        alpha_moment_mapping,
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
        rmin);
}

};  // namespace : mtpr
};  // namespace : ai2pot
