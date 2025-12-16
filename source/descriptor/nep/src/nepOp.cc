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

#include "../include/nepOp.h"
#include "../include/nep.h"
#include "../include/nep_cpu_launcher.h"
#include "../include/nepLoss_cpu_launcher.h"



namespace ai2pot {
namespace nep {


torch::autograd::variable_list NepToEFFunction::forward(
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
    const at::Tensor& q_scaler_tensor)
{
    // 1. 
    int batch_size = bfirstneigh_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int *type_map = type_map_tensor.data_ptr<int>();
    int num_neurons = (int)(w1_tensor.size(0) / ntypes);
    int num_descriptors = (int)(w0_tensor.size(0) / (ntypes*num_neurons));

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(brcs_tensor.scalar_type())
                                        .device(brcs_tensor.device());
    
    // 3.
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad+nghost, 3}, float_options);

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *w0 = w0_tensor.data_ptr<float>();
        float *w1 = w1_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();

        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_ef_cpu_launcher<float>(
                betot,
                bforce,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                q_scaler);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    } else {
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *w0 = w0_tensor.data_ptr<double>();
        double *w1 = w1_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();

        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_ef_cpu_launcher<double>(
                betot,
                bforce,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                q_scaler);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    }

    return {betot_tensor, bforce_tensor};
}


torch::autograd::variable_list NepToEFFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
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
        at::Tensor()};
}


torch::autograd::variable_list NepToEFLossFunction::forward(
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
    const at::Tensor& q_scaler_tensor)
{
    // 1. 
    int batch_size = bfirstneigh_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int *type_map = (int*)type_map_tensor.data_ptr<int>();
    int num_neurons = (int)(w1_tensor.size(0) / ntypes);
    int num_descriptors = (int)(w0_tensor.size(0) / (ntypes*num_neurons));

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(brcs_tensor.scalar_type())
                                        .device(brcs_tensor.device());

    // 3. 
    at::Tensor bloss_tensor = at::zeros({batch_size}, float_options);
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *bloss = bloss_tensor.data_ptr<float>();
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *w0 = w0_tensor.data_ptr<float>();
        float *w1 = w1_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();

        float *betot_dft = betot_dft_tensor.data_ptr<float>();
        float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();

        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_ef_cpu_launcher<float>(
                betot,
                bforce,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                q_scaler);
            
            find_ef_loss_cpu_launcher<float>(
                bloss,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                e_weight,
                f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft);

        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    } else {
        double *bloss = bloss_tensor.data_ptr<double>();
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *w0 = w0_tensor.data_ptr<double>();
        double *w1 = w1_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
    
        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();

        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        if (brcs_tensor.device() == c10::kCPU) {
            find_ef_cpu_launcher<double>(
                betot,
                bforce,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                q_scaler);
            
            find_ef_loss_cpu_launcher<double>(
                bloss,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                e_weight,
                f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ... 
            #endif
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
        torch::tensor(n_radial_basis, int_options),
        torch::tensor(n_angular_basis, int_options),
        torch::tensor(l_max, int_options),
        coeffs_tensor,
        w0_tensor,
        w1_tensor,
        type_bias_tensor,
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
        q_scaler_tensor});
    
    return {bloss_tensor};
}


torch::autograd::variable_list NepToEFLossFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    // 0. 
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    
    double e_weight = ctx->get_saved_variables()[0].item<double>();
    double f_weight = ctx->get_saved_variables()[1].item<double>();
    at::Tensor betot_dft_tensor = ctx->get_saved_variables()[2];
    at::Tensor bforce_dft_tensor = ctx->get_saved_variables()[3];
    int chebyshev_size = ctx->get_saved_variables()[4].item<int>();
    int n_radial_basis = ctx->get_saved_variables()[5].item<int>();
    int n_angular_basis = ctx->get_saved_variables()[6].item<int>();
    int l_max = ctx->get_saved_variables()[7].item<int>();
    at::Tensor coeffs_tensor = ctx->get_saved_variables()[8];
    at::Tensor w0_tensor = ctx->get_saved_variables()[9];
    at::Tensor w1_tensor = ctx->get_saved_variables()[10];
    at::Tensor type_bias_tensor = ctx->get_saved_variables()[11];
    at::Tensor binum_tensor = ctx->get_saved_variables()[12];
    at::Tensor bilist_tensor = ctx->get_saved_variables()[13];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[14];
    at::Tensor bfirstneigh_tensor = ctx->get_saved_variables()[15];
    at::Tensor brcs_tensor = ctx->get_saved_variables()[16];
    at::Tensor btypes_tensor = ctx->get_saved_variables()[17];
    at::Tensor type_map_tensor = ctx->get_saved_variables()[18];
    int nghost = ctx->get_saved_variables()[19].item<int>();
    double rmax = ctx->get_saved_variables()[20].item<double>();
    double rmin = ctx->get_saved_variables()[21].item<double>();
    at::Tensor q_scaler_tensor = ctx->get_saved_variables()[22];

    // 1.
    int batch_size = bfirstneigh_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int *type_map = type_map_tensor.data_ptr<int>();
    int num_neurons = (int)(w1_tensor.size(0) / ntypes);
    int num_descriptors = (int)(w0_tensor.size(0) / (ntypes*num_neurons));
    int num_coeffs = ntypes * ntypes * (n_radial_basis + n_angular_basis) * chebyshev_size;

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(brcs_tensor.scalar_type())
                                        .device(brcs_tensor.device());

    // 3. 
    at::Tensor bloss_der2coeffs_tensor = at::zeros({batch_size, num_coeffs}, float_options);
    at::Tensor bloss_der2w0_tensor = at::zeros({batch_size, ntypes*num_neurons*num_descriptors}, float_options);
    at::Tensor bloss_der2w1_tensor = at::zeros({batch_size, ntypes*num_neurons}, float_options);
    at::Tensor bloss_der2type_bias_tensor = at::zeros({batch_size, ntypes}, float_options);
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *bloss_der2coeffs = bloss_der2coeffs_tensor.data_ptr<float>();
        float *bloss_der2w0 = bloss_der2w0_tensor.data_ptr<float>();
        float *bloss_der2w1 = bloss_der2w1_tensor.data_ptr<float>();
        float *bloss_der2type_bias = bloss_der2type_bias_tensor.data_ptr<float>();
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *w0 = w0_tensor.data_ptr<float>();
        float *w1 = w1_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();

        float *betot_dft = betot_dft_tensor.data_ptr<float>();
        float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();

        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_ef_cpu_launcher<float>(
                betot,
                bforce,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                q_scaler);

            find_ef_loss_backward_cpu_launcher<float>(
                bloss_der2coeffs,
                bloss_der2w0,
                bloss_der2w1,
                bloss_der2type_bias,
                e_weight,
                f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                q_scaler);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    } else {
        double *bloss_der2coeffs = bloss_der2coeffs_tensor.data_ptr<double>();
        double *bloss_der2w0 = bloss_der2w0_tensor.data_ptr<double>();
        double *bloss_der2w1 = bloss_der2w1_tensor.data_ptr<double>();
        double *bloss_der2type_bias = bloss_der2type_bias_tensor.data_ptr<double>();
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *w0 = w0_tensor.data_ptr<double>();
        double *w1 = w1_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();

        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();

        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_ef_cpu_launcher<double>(
                betot,
                bforce,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                q_scaler);

            find_ef_loss_backward_cpu_launcher<double>(
                bloss_der2coeffs,
                bloss_der2w0,
                bloss_der2w1,
                bloss_der2type_bias,
                e_weight,
                f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                w1,
                type_bias,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin,
                q_scaler);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
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
        at::Tensor()};
}


torch::autograd::variable_list NepToDescriptorsFunction::forward(
    torch::autograd::AutogradContext *ctx,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin)
{
    // 1. 
    int batch_size = bfirstneigh_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int *type_map = type_map_tensor.data_ptr<int>();
    int num_descriptors = NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    // 2.
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(brcs_tensor.scalar_type())
                                        .device(brcs_tensor.device());

    // 3. 
    at::Tensor bdescriptors_tensor = at::zeros({batch_size, natoms_pad, num_descriptors}, float_options);

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *bdescriptors = bdescriptors_tensor.data_ptr<float>();
        float *coeffs = coeffs_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_descriptors_cpu_launcher<float>(
                bdescriptors,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                coeffs,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    } else {
        double *bdescriptors = bdescriptors_tensor.data_ptr<double>();
        double *coeffs = coeffs_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_descriptors_cpu_launcher<double>(
                bdescriptors,
                chebyshev_size,
                n_radial_basis,
                n_angular_basis,
                l_max,
                coeffs,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                ntypes,
                type_map,
                umax_num_neigh_atoms,
                nghost,
                rmax,
                rmin);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // ...
            #endif
        }
    }

    return {bdescriptors_tensor};
}


torch::autograd::variable_list NepToDescriptorsFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
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
        at::Tensor()};
}


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
    const at::Tensor& q_scaler_tensor)
{
    return NepToEFFunction::apply(
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        coeffs_tensor,
        w0_tensor,
        w1_tensor,
        type_bias_tensor,
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
        q_scaler_tensor);
}


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
    const at::Tensor& q_scaler_tensor)
{
    return NepToEFLossFunction::apply(
        e_weight,
        f_weight,
        betot_dft_tensor,
        bforce_dft_tensor,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        coeffs_tensor,
        w0_tensor,
        w1_tensor,
        type_bias_tensor,
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
        q_scaler_tensor);
}


torch::autograd::variable_list NepToDescriptorsOp(
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    const at::Tensor& brcs_tensor,
    const at::Tensor& btypes_tensor,
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin)
{
    return NepToDescriptorsFunction::apply(
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        coeffs_tensor,
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        type_map_tensor,
        nghost,
        rmax,
        rmin);
}


};  // namespace : nep
};  // namespace : ai2pot
