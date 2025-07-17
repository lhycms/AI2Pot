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
    int num_neurons = (int)(w1_tensor.size(0) / ntypes);
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
            float *loss = (float*)bloss_tensor[bb].data_ptr<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
            float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<float>::find_ef_loss(
                (*loss),
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
            double *loss = (double*)bloss_tensor[bb].data_ptr<double>();
            double etot_dft = betot_dft_tensor[bb].item<double>();
            double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<double>::find_ef_loss(
                (*loss),
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
    at::Tensor betot_dft_tensor = ctx->get_saved_variables()[2];
    at::Tensor bforce_dft_tensor = ctx->get_saved_variables()[3];
    int chebyshev_size = ctx->get_saved_variables()[4].item<int>();
    at::Tensor coeffs_tensor = ctx->get_saved_variables()[5];
    at::Tensor w0_tensor = ctx->get_saved_variables()[6];
    at::Tensor w1_tensor = ctx->get_saved_variables()[7];
    at::Tensor type_bias_tensor = ctx->get_saved_variables()[8];
    int alpha_moments_count = ctx->get_saved_variables()[9].item<int>();
    at::Tensor alpha_index_basic_tensor = ctx->get_saved_variables()[10];
    at::Tensor alpha_index_times_tensor = ctx->get_saved_variables()[11];
    at::Tensor alpha_moment_mapping_tensor = ctx->get_saved_variables()[12];
    int nmus = ctx->get_saved_variables()[13].item<int>();
    at::Tensor binum_tensor = ctx->get_saved_variables()[14];
    at::Tensor bilist_tensor = ctx->get_saved_variables()[15];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[16];
    at::Tensor bfirstneigh_tensor = ctx->get_saved_variables()[17];
    at::Tensor brcs_tensor = ctx->get_saved_variables()[18];
    at::Tensor btypes_tensor = ctx->get_saved_variables()[19];
    at::Tensor type_map_tensor = ctx->get_saved_variables()[20];
    int nghost = ctx->get_saved_variables()[21].item<int>();
    double rmax = ctx->get_saved_variables()[22].item<double>();
    double rmin = ctx->get_saved_variables()[23].item<double>();
    double zbl_rmax = ctx->get_saved_variables()[24].item<double>();
    double zbl_rmin = ctx->get_saved_variables()[25].item<double>();
    at::Tensor zbl_cks_tensor = ctx->get_saved_variables()[26];
    at::Tensor zbl_dks_tensor = ctx->get_saved_variables()[27];

    // 1. 
    int batch_size = (int)bfirstneigh_tensor.size(0);
    int num_atoms = (int)bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = (int)bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int *type_map = (int*)type_map_tensor.data_ptr<int>();
    int num_neurons = (int)(w1_tensor.size(0) / ntypes);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    // 3. 
    at::Tensor bloss_der2coeffs_tensor;
    at::Tensor bloss_der2w0_tensor;
    at::Tensor bloss_der2w1_tensor;
    at::Tensor bloss_der2type_bias_tensor;

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(brcs_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({batch_size, num_coeffs}, float_options);
        bloss_der2w0_tensor = at::zeros({batch_size, num_neurons*alpha_scalar_moments}, float_options);
        bloss_der2w1_tensor = at::zeros({batch_size, num_neurons}, float_options);
        bloss_der2type_bias_tensor = at::zeros({batch_size, ntypes}, float_options);
        float *coeffs = (float*)coeffs_tensor.data_ptr<float>();
        float *w0 = (float*)w0_tensor.data_ptr<float>();
        float *w1 = (float*)w1_tensor.data_ptr<float>();
        float *type_bias = (float*)type_bias_tensor.data_ptr<float>();
        float *zbl_cks = (float*)zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = (float*)zbl_dks_tensor.data_ptr<float>();

        for (int bb=0; bb<batch_size; bb++)
        {
            float *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<float>();
            float *loss_der2w0 = bloss_der2w0_tensor[bb].data_ptr<float>();
            float *loss_der2w1 = bloss_der2w1_tensor[bb].data_ptr<float>();
            float *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
            float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<float>::find_ef_loss_backward(
                loss_der2coeffs,
                loss_der2w0,
                loss_der2w1,
                loss_der2type_bias,
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
        bloss_der2coeffs_tensor = at::zeros({batch_size, num_coeffs}, float_options);
        bloss_der2w0_tensor = at::zeros({batch_size, num_neurons*alpha_scalar_moments}, float_options);
        bloss_der2w1_tensor = at::zeros({batch_size, num_neurons}, float_options);
        bloss_der2type_bias_tensor = at::zeros({batch_size, ntypes}, float_options);
        double *coeffs = (double*)coeffs_tensor.data_ptr<double>();
        double *w0 = (double*)w0_tensor.data_ptr<double>();
        double *w1 = (double*)w1_tensor.data_ptr<double>();
        double *type_bias = (double*)type_bias_tensor.data_ptr<double>();
        double *zbl_cks = (double*)zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = (double*)zbl_dks_tensor.data_ptr<double>();

        for (int bb=0; bb<batch_size; bb++)
        {
            double *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<double>();
            double *loss_der2w0 = bloss_der2w0_tensor[bb].data_ptr<double>();
            double *loss_der2w1 = bloss_der2w1_tensor[bb].data_ptr<double>();
            double *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<double>();
            double etot_dft = betot_dft_tensor[bb].item<double>();
            double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<double>::find_ef_loss_backward(
                loss_der2coeffs,
                loss_der2w0,
                loss_der2w1,
                loss_der2type_bias,
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

    return {
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        torch::matmul(bgrad_output_tensor, bloss_der2coeffs_tensor),
        torch::matmul(bgrad_output_tensor, bloss_der2w0_tensor),
        torch::matmul(bgrad_output_tensor, bloss_der2w1_tensor),
        torch::matmul(bgrad_output_tensor, bloss_der2type_bias_tensor),
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


torch::autograd::variable_list NNMtpToLossFunction::forward(
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
    int num_neurons = (int)(w1_tensor.size(0) / ntypes);
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
            float *loss = (float*)bloss_tensor[bb].data_ptr<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
            float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
            float *virial_dft = (float*)bvirial_dft_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<float>::find_loss(
                (*loss),
                e_weight,
                f_weight,
                v_weight,
                etot_dft,
                force_dft,
                virial_dft,
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
            double *loss = (double*)bloss_tensor[bb].data_ptr<double>();
            double etot_dft = betot_dft_tensor[bb].item<double>();
            double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
            double *virial_dft = (double*)bvirial_dft_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<double>::find_loss(
                (*loss),
                e_weight,
                f_weight,
                v_weight,
                etot_dft,
                force_dft,
                virial_dft,
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
        torch::tensor(v_weight, float_options),
        betot_dft_tensor,
        bforce_dft_tensor,
        bvirial_dft_tensor,
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


torch::autograd::variable_list NNMtpToLossFunction::backward(
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
    at::Tensor w0_tensor = ctx->get_saved_variables()[8];
    at::Tensor w1_tensor = ctx->get_saved_variables()[9];
    at::Tensor type_bias_tensor = ctx->get_saved_variables()[10];
    int alpha_moments_count = ctx->get_saved_variables()[11].item<int>();
    at::Tensor alpha_index_basic_tensor = ctx->get_saved_variables()[12];
    at::Tensor alpha_index_times_tensor = ctx->get_saved_variables()[13];
    at::Tensor alpha_moment_mapping_tensor = ctx->get_saved_variables()[14];
    int nmus = ctx->get_saved_variables()[15].item<int>();
    at::Tensor binum_tensor = ctx->get_saved_variables()[16];
    at::Tensor bilist_tensor = ctx->get_saved_variables()[17];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[18];
    at::Tensor bfirstneigh_tensor = ctx->get_saved_variables()[19];
    at::Tensor brcs_tensor = ctx->get_saved_variables()[20];
    at::Tensor btypes_tensor = ctx->get_saved_variables()[21];
    at::Tensor type_map_tensor = ctx->get_saved_variables()[22];
    int nghost = ctx->get_saved_variables()[23].item<int>();
    double rmax = ctx->get_saved_variables()[24].item<double>();
    double rmin = ctx->get_saved_variables()[25].item<double>();
    double zbl_rmax = ctx->get_saved_variables()[26].item<double>();
    double zbl_rmin = ctx->get_saved_variables()[27].item<double>();
    at::Tensor zbl_cks_tensor = ctx->get_saved_variables()[28];
    at::Tensor zbl_dks_tensor = ctx->get_saved_variables()[29];

    // 1. 
    int batch_size = (int)bfirstneigh_tensor.size(0);
    int num_atoms = (int)bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = (int)bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int *type_map = (int*)type_map_tensor.data_ptr<int>();
    int num_neurons = (int)(w1_tensor.size(0) / ntypes);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    // 3. 
    at::Tensor bloss_der2coeffs_tensor;
    at::Tensor bloss_der2w0_tensor;
    at::Tensor bloss_der2w1_tensor;
    at::Tensor bloss_der2type_bias_tensor;

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(brcs_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({batch_size, num_coeffs}, float_options);
        bloss_der2w0_tensor = at::zeros({batch_size, num_neurons*alpha_scalar_moments}, float_options);
        bloss_der2w1_tensor = at::zeros({batch_size, num_neurons}, float_options);
        bloss_der2type_bias_tensor = at::zeros({batch_size, ntypes}, float_options);
        float *coeffs = (float*)coeffs_tensor.data_ptr<float>();
        float *w0 = (float*)w0_tensor.data_ptr<float>();
        float *w1 = (float*)w1_tensor.data_ptr<float>();
        float *type_bias = (float*)type_bias_tensor.data_ptr<float>();
        float *zbl_cks = (float*)zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = (float*)zbl_dks_tensor.data_ptr<float>();

        for (int bb=0; bb<batch_size; bb++)
        {
            float *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<float>();
            float *loss_der2w0 = bloss_der2w0_tensor[bb].data_ptr<float>();
            float *loss_der2w1 = bloss_der2w1_tensor[bb].data_ptr<float>();
            float *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<float>();
            float etot_dft = betot_dft_tensor[bb].item<float>();
            float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
            float *virial_dft = (float*)bvirial_dft_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<float>::find_loss_backward(
                loss_der2coeffs,
                loss_der2w0,
                loss_der2w1,
                loss_der2type_bias,
                e_weight,
                f_weight,
                v_weight,
                etot_dft,
                force_dft,
                virial_dft,
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
        bloss_der2coeffs_tensor = at::zeros({batch_size, num_coeffs}, float_options);
        bloss_der2w0_tensor = at::zeros({batch_size, num_neurons*alpha_scalar_moments}, float_options);
        bloss_der2w1_tensor = at::zeros({batch_size, num_neurons}, float_options);
        bloss_der2type_bias_tensor = at::zeros({batch_size, ntypes}, float_options);
        double *coeffs = (double*)coeffs_tensor.data_ptr<double>();
        double *w0 = (double*)w0_tensor.data_ptr<double>();
        double *w1 = (double*)w1_tensor.data_ptr<double>();
        double *type_bias = (double*)type_bias_tensor.data_ptr<double>();
        double *zbl_cks = (double*)zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = (double*)zbl_dks_tensor.data_ptr<double>();

        for (int bb=0; bb<batch_size; bb++)
        {
            double *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<double>();
            double *loss_der2w0 = bloss_der2w0_tensor[bb].data_ptr<double>();
            double *loss_der2w1 = bloss_der2w1_tensor[bb].data_ptr<double>();
            double *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<double>();
            double etot_dft = betot_dft_tensor[bb].item<double>();
            double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
            double *virial_dft = (double*)bvirial_dft_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();

            NNMtp<double>::find_loss_backward(
                loss_der2coeffs,
                loss_der2w0,
                loss_der2w1,
                loss_der2type_bias,
                e_weight,
                f_weight,
                v_weight,
                etot_dft,
                force_dft,
                virial_dft,
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

    return {
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        at::Tensor(),
        torch::matmul(bgrad_output_tensor, bloss_der2coeffs_tensor),
        torch::matmul(bgrad_output_tensor, bloss_der2w0_tensor),
        torch::matmul(bgrad_output_tensor, bloss_der2w1_tensor),
        torch::matmul(bgrad_output_tensor, bloss_der2type_bias_tensor),
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





torch::autograd::variable_list NNMtpToEFLossOp(
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
    return NNMtpToEFLossFunction::apply(
        e_weight,
        f_weight,
        betot_dft_tensor,
        bforce_dft_tensor,
        chebyshev_size,
        coeffs_tensor,
        w0_tensor,
        w1_tensor,
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
        type_map_tensor,
        nghost,
        rmax,
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
}


torch::autograd::variable_list NNMtpToLossOp(
        double e_weight,
        double f_weight,
        double v_weight,
        const at::Tensor& betot_dft_tensor,
        const at::Tensor& bforce_dft_tensor,
        const at::Tensor& bvirial_dft_tensor,
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
    return NNMtpToLossFunction::apply(
        e_weight,
        f_weight,
        v_weight,
        betot_dft_tensor,
        bforce_dft_tensor,
        bvirial_dft_tensor,
        chebyshev_size,
        coeffs_tensor,
        w0_tensor,
        w1_tensor,
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
        type_map_tensor,
        nghost,
        rmax,
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
}

};  // namespace : nnmtp
};  // namespace : ai2pot
