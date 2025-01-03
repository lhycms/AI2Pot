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
#include <cassert>

#include "../include/mtpBasis.h"
#include "../include/mtpBasisOp.h"


namespace ai2pot {
namespace mtpr {

torch::autograd::variable_list MtpBasisFunction::forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor &alpha_index_basic_tensor,
    const at::Tensor &alpha_index_times_tensor,
    const at::Tensor &alpha_moment_mapping_tensor,
    const at::Tensor &num_mus4moms_tensor,
    const at::Tensor &mus4moms_tensor,
    const at::Tensor &nmus_tensor,
    int ntypes,
    int chebyshev_size,
    const at::Tensor &coeffs_tensor,
    const at::Tensor &binum_tensor,
    const at::Tensor &bilist_tensor,
    const at::Tensor &bnumneigh_tensor,
    const at::Tensor &bfirstneigh_tensor,
    const at::Tensor &brcs_tensor,
    const at::Tensor &btypes_tensor,
    int umax_num_neighs,
    double rmax,
    double rmin)
{
    assert(bilist_tensor.scalar_type() == torch::kInt32);
    assert(bnumneigh_tensor.scalar_type() == torch::kInt32);
    assert(bfirstneigh_tensor.scalar_type() == torch::kInt32);
    assert(btypes_tensor.scalar_type() == torch::kInt32);
    assert(coeffs_tensor.scalar_type() == brcs_tensor.scalar_type());
    assert( (int)coeffs_tensor.flatten().size(0) == (ntypes * ntypes * nmus_tensor.item<int>() * chebyshev_size) );

    int batch_size = (int)bilist_tensor.size(0);
    int nlocal = (int)bilist_tensor.size(1);
    int num_coeffs = (int)coeffs_tensor.flatten().size(0);
    int alpha_moments_count = (int)mus4moms_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
    int max_num_mus4moms = (int)mus4moms_tensor.size(1);
    int *num_mus4moms = num_mus4moms_tensor.data_ptr<int>();
    int *mus4moms = mus4moms_tensor.data_ptr<int>();
    int nmus = nmus_tensor.item<int>();
    int *binum = binum_tensor.data_ptr<int>();

    c10::TensorOptions int_options = c10::TensorOptions()
        .dtype(torch::kInt32)
        .device(brcs_tensor.device());
    c10::TensorOptions float_options;
    bool calculate_der2xyz = false;
    bool calculate_der2coeffs = false;
    if (brcs_tensor.requires_grad())
        calculate_der2xyz = true;
    if (coeffs_tensor.requires_grad())
        calculate_der2coeffs = true;

    at::Tensor bmtp_basis_val_tensor;
    at::Tensor bmtp_basis_der_tensor;
    at::Tensor bmtp_basis_der2coeffs_tensor;
    at::Tensor mtp_size_tensor = at::zeros(2, int_options);
    mtp_size_tensor[0] = alpha_scalar_moments;
    mtp_size_tensor[1] = num_coeffs;

    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat32)
            .device(brcs_tensor.device());
        bmtp_basis_val_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments}, float_options);
        bmtp_basis_val_tensor.requires_grad_(true);
        if (calculate_der2xyz) {
            bmtp_basis_der_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments, umax_num_neighs, 3}, float_options);
            bmtp_basis_der_tensor.requires_grad_(calculate_der2xyz);
        }
        if (calculate_der2coeffs) {
            bmtp_basis_der2coeffs_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments, num_coeffs}, float_options);
            bmtp_basis_der2coeffs_tensor.requires_grad_(calculate_der2coeffs);
        }

        for (int bb=0; bb<batch_size; bb++) {
            float *mtp_basis_val = bmtp_basis_val_tensor[bb].data_ptr<float>();
            float (*mtp_basis_der)[3] = (calculate_der2xyz) ? (float (*)[3])bmtp_basis_der_tensor[bb].data_ptr<float>() : nullptr;
            float *mtp_basis_der2coeffs = (calculate_der2coeffs) ? bmtp_basis_der2coeffs_tensor[bb].data_ptr<float>() : nullptr;
            float *coeffs = coeffs_tensor.data_ptr<float>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = btypes_tensor[bb].data_ptr<int>();

            MtpBasis<float>::find_val_der(
                mtp_basis_val,
                mtp_basis_der,
                mtp_basis_der2coeffs,
                calculate_der2xyz,
                calculate_der2coeffs,
                chebyshev_size,
                coeffs,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moments,
                alpha_moment_mapping,
                max_num_mus4moms,
                num_mus4moms,
                mus4moms,
                nmus,
                binum[bb],
                ilist,
                numneigh,
                firstneigh,
                rcs,
                types,
                ntypes,
                umax_num_neighs,
                rmax,
                rmin);
        }
    } else {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(brcs_tensor.device());
        bmtp_basis_val_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments}, float_options);
        bmtp_basis_val_tensor.requires_grad_(true);
        if (calculate_der2xyz) {
            bmtp_basis_der_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments, umax_num_neighs, 3}, float_options);
            bmtp_basis_der_tensor.requires_grad_(calculate_der2xyz);
        }
        if (calculate_der2coeffs) {
            bmtp_basis_der2coeffs_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments, num_coeffs}, float_options);
            bmtp_basis_der2coeffs_tensor.requires_grad_(calculate_der2coeffs);
        }

        for (int bb=0; bb<batch_size; bb++) {
            double *mtp_basis_val = bmtp_basis_val_tensor[bb].data_ptr<double>();
            double (*mtp_basis_der)[3] = (calculate_der2xyz) ? (double (*)[3])bmtp_basis_der_tensor[bb].data_ptr<double>() : nullptr;
            double *mtp_basis_der2coeff = (calculate_der2coeffs) ? bmtp_basis_der2coeffs_tensor[bb].data_ptr<double>() : nullptr;
            double *coeffs = coeffs_tensor.data_ptr<double>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = btypes_tensor[bb].data_ptr<int>();

            MtpBasis<double>::find_val_der(
                mtp_basis_val,
                mtp_basis_der,
                mtp_basis_der2coeff,
                calculate_der2xyz,
                calculate_der2coeffs,
                chebyshev_size,
                coeffs,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moments,
                alpha_moment_mapping,
                max_num_mus4moms,
                num_mus4moms,
                mus4moms,
                nmus,
                binum[bb],
                ilist,
                numneigh,
                firstneigh,
                rcs,
                types,
                ntypes,
                umax_num_neighs,
                rmax,
                rmin);
        }
    }
    ctx->save_for_backward({binum_tensor,
                            brcs_tensor,
                            mtp_size_tensor,
                            bmtp_basis_der_tensor,
                            bmtp_basis_der2coeffs_tensor});
    return {bmtp_basis_val_tensor};
}


torch::autograd::variable_list MtpBasisFunction::backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor)
{
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    at::Tensor binum_tensor = ctx->get_saved_variables()[0];
    at::Tensor brcs_tensor = ctx->get_saved_variables()[1];
    at::Tensor mtp_size_tensor = ctx->get_saved_variables()[2];
    at::Tensor bmtp_basis_der_tensor = ctx->get_saved_variables()[3];
    at::Tensor bmtp_basis_der2coeffs_tensor = ctx->get_saved_variables()[4];

    int batch_size = (int)brcs_tensor.size(0);
    int nlocal = (int)brcs_tensor.size(1);
    int umax_num_neighs = (int)brcs_tensor.size(2);
    int alpha_scalar_moments = (int)mtp_size_tensor[0].item<int>();
    int num_coeffs = (int)mtp_size_tensor[1].item<int>();
    int *binum = binum_tensor.data_ptr<int>();

    c10::TensorOptions float_options;
    bool calculate_der2xyz = false;
    bool calculate_der2coeffs = false;
    if (bmtp_basis_der_tensor.numel())
        calculate_der2xyz = true;
    if (bmtp_basis_der2coeffs_tensor.numel())
        calculate_der2coeffs = true;

    at::Tensor bout_der_tensor;
    at::Tensor bout_der2coeffs_tensor;

    if (bgrad_output_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat32)
            .device(bgrad_output_tensor.device());
        if (calculate_der2xyz) {
            bout_der_tensor = at::zeros({batch_size, nlocal, umax_num_neighs, 3}, float_options);
            bout_der_tensor.requires_grad_(calculate_der2xyz);
        }
        if (calculate_der2coeffs) {
            bout_der2coeffs_tensor = at::zeros({num_coeffs}, float_options);
            bout_der2coeffs_tensor.requires_grad_(calculate_der2coeffs);
        }

        for (int bb=0; bb<batch_size; bb++) {
            float *out_der = (calculate_der2xyz) ? bout_der_tensor[bb].data_ptr<float>() : nullptr;
            float *out_der2coeffs = (calculate_der2coeffs) ? bout_der2coeffs_tensor.data_ptr<float>() : nullptr;
            float *mtp_basis_der = (calculate_der2xyz) ? bmtp_basis_der_tensor[bb].data_ptr<float>() : nullptr;
            float *mtp_basis_der2coeffs = (calculate_der2coeffs) ? bmtp_basis_der2coeffs_tensor[bb].data_ptr<float>() : nullptr;
            float *grad_output = bgrad_output_tensor[bb].data_ptr<float>();
            
            MtpBasis<float>::find_der_backward(out_der,
                                               out_der2coeffs,
                                               calculate_der2xyz,
                                               calculate_der2coeffs,
                                               grad_output,
                                               mtp_basis_der,
                                               mtp_basis_der2coeffs,
                                               alpha_scalar_moments,
                                               binum[bb],
                                               umax_num_neighs,
                                               num_coeffs);
        }
    } else {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(bgrad_output_tensor.device());
        if (calculate_der2xyz) {
            bout_der_tensor = at::zeros({batch_size, nlocal, umax_num_neighs, 3}, float_options);
            bout_der_tensor.requires_grad_(calculate_der2xyz);
        }
        if (calculate_der2coeffs) {
            bout_der2coeffs_tensor = at::zeros({num_coeffs}, float_options);
            bout_der2coeffs_tensor.requires_grad_(calculate_der2coeffs);
        }

        for (int bb=0; bb<batch_size; bb++) {
            double *out_der = (calculate_der2xyz) ? bout_der_tensor[bb].data_ptr<double>() : nullptr;
            double *out_der2coeffs = (calculate_der2coeffs) ? bout_der2coeffs_tensor.data_ptr<double>() : nullptr;
            double *mtp_basis_der = (calculate_der2xyz) ? bmtp_basis_der_tensor[bb].data_ptr<double>() : nullptr;
            double *mtp_basis_der2coeffs = (calculate_der2coeffs) ? bmtp_basis_der2coeffs_tensor[bb].data_ptr<double>() : nullptr;
            double *grad_output = bgrad_output_tensor[bb].data_ptr<double>();

            MtpBasis<double>::find_der_backward(out_der,
                                                out_der2coeffs,
                                                calculate_der2xyz,
                                                calculate_der2coeffs,
                                                grad_output,
                                                mtp_basis_der,
                                                mtp_basis_der2coeffs,
                                                alpha_scalar_moments,
                                                binum[bb],
                                                umax_num_neighs,
                                                num_coeffs);
        }
    }
    return {
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(),
        bout_der2coeffs_tensor,
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        bout_der_tensor,
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}


torch::autograd::variable_list MtpBasisOp(
    const at::Tensor &alpha_index_basic_tensor,
    const at::Tensor &alpha_index_times_tensor,
    const at::Tensor &alpha_moment_mapping_tensor,
    const at::Tensor &num_mus4moms_tensor,
    const at::Tensor &mus4moms_tensor,
    const at::Tensor &nmus_tensor,
    int ntypes,
    int chebyshev_size,
    const at::Tensor &coeffs_tensor,
    const at::Tensor &binum_tensor,
    const at::Tensor &bilist_tensor,
    const at::Tensor &bnumneigh_tensor,
    const at::Tensor &bfirstneigh_tensor,
    const at::Tensor &brcs_tensor,
    const at::Tensor &btypes_tensor,
    int umax_num_neighs,
    double rmax,
    double rmin)
{
    return MtpBasisFunction::apply(
        alpha_index_basic_tensor,
        alpha_index_times_tensor,
        alpha_moment_mapping_tensor,
        num_mus4moms_tensor,
        mus4moms_tensor,
        nmus_tensor,
        ntypes,
        chebyshev_size,
        coeffs_tensor,
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        umax_num_neighs,
        rmax,
        rmin);
}

};  // namespace : mtpr
};  // namespace : ai2pot
