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
#include <cassert>

#include "../include/mtpBasisGradOp.h"
#include "../include/mtpBasisGrad.h"


namespace ai2pot {
namespace mtpr {

torch::autograd::variable_list MtpBasisGradFunction::forward(
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
    int *mus4mom = mus4moms_tensor.data_ptr<int>();
    int nmus = nmus_tensor.item<int>();
    int *binum = binum_tensor.data_ptr<int>();

    c10::TensorOptions int_options = c10::TensorOptions()
        .dtype(torch::kInt32)
        .device(brcs_tensor.device());
    c10::TensorOptions float_options;
    bool calculate_mtp_basis = false;

    at::Tensor bmtp_basis_val_tensor;
    at::Tensor bmbg_val_tensor;
    at::Tensor bmbg_der2coeff_tensor;

    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(brcs_tensor.scalar_type())
                            .device(brcs_tensor.device());

        if (calculate_mtp_basis)
            bmtp_basis_val_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments}, float_options);
        bmbg_val_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments, umax_num_neighs, 3}, float_options);
        bmbg_der2coeff_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments, umax_num_neighs, 3, num_coeffs}, float_options);

        for (int bb=0; bb<batch_size; bb++) {
            float *mtp_basis_val = calculate_mtp_basis ? bmtp_basis_val_tensor[bb].data_ptr<float>() : nullptr;
            float (*mbg_val)[3] = (float (*)[3])bmbg_val_tensor[bb].data_ptr<float>();
            float *mbg_der2coeff = bmbg_der2coeff_tensor[bb].data_ptr<float>();
            float *coeffs = coeffs_tensor.data_ptr<float>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = btypes_tensor[bb].data_ptr<int>();

            MtpBasisGrad<float>::find_val_der(
                mtp_basis_val,
                mbg_val,
                mbg_der2coeff,
                calculate_mtp_basis,
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
                mus4mom,
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
                            .dtype(brcs_tensor.scalar_type())
                            .device(brcs_tensor.device());
        
        if (calculate_mtp_basis)
            bmtp_basis_val_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments}, float_options);
        bmbg_val_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments, umax_num_neighs, 3}, float_options);
        bmbg_der2coeff_tensor = at::zeros({batch_size, nlocal, alpha_scalar_moments, umax_num_neighs, 3, num_coeffs}, float_options);

        for (int bb=0; bb<batch_size; bb++) {
            double *mtp_basis_val = calculate_mtp_basis ? bmtp_basis_val_tensor[bb].data_ptr<double>() : nullptr;
            double (*mbg_val)[3] = (double (*)[3])bmbg_val_tensor[bb].data_ptr<double>();
            double *mbg_der2coeff = bmbg_der2coeff_tensor[bb].data_ptr<double>();
            double *coeffs = coeffs_tensor.data_ptr<double>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = btypes_tensor[bb].data_ptr<int>();

            MtpBasisGrad<double>::find_val_der(
                mtp_basis_val,
                mbg_val,
                mbg_der2coeff,
                calculate_mtp_basis,
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
                mus4mom,
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
    
    ctx->save_for_backward({
        binum_tensor,
        bnumneigh_tensor,
        bmbg_der2coeff_tensor});
    return {bmbg_val_tensor};
}


torch::autograd::variable_list MtpBasisGradFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    at::Tensor binum_tensor = ctx->get_saved_variables()[0];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[1];
    at::Tensor bmbg_der2coeff_tensor = ctx->get_saved_variables()[2];
    int *binum = binum_tensor.data_ptr<int>();
    
    int batch_size = (int)bgrad_output_tensor.size(0);
    int nlocal = (int)bgrad_output_tensor.size(1);
    int umax_num_neighs = (int)bgrad_output_tensor.size(2);
    int num_coeffs = (int)bgrad_output_tensor.size(4);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(bgrad_output_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bout_der2coeffs_tensor;

    if (bgrad_output_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(bgrad_output_tensor.scalar_type())
                            .device(bgrad_output_tensor.device());
        bout_der2coeffs_tensor = at::zeros({batch_size, num_coeffs}, float_options);
        
        for (int bb=0; bb<batch_size; bb++) {
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            float *out_der2coeffs = bout_der2coeffs_tensor[bb].data_ptr<float>();
            float *grad_output = bgrad_output_tensor[bb].data_ptr<float>();
            float *mbg_der2coeff = bmbg_der2coeff_tensor[bb].data_ptr<float>();

            MtpBasisGrad<float>::find_der_backward(
                out_der2coeffs,
                grad_output,
                mbg_der2coeff,
                binum[bb],
                numneigh,
                umax_num_neighs,
                num_coeffs);
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(bgrad_output_tensor.scalar_type())
                            .device(bgrad_output_tensor.device());

        for (int bb=0; bb<batch_size; bb++) {
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            double *out_der2coeff = bout_der2coeffs_tensor[bb].data_ptr<double>();
            double *grad_output = bgrad_output_tensor[bb].data_ptr<double>();
            double *mbg_der2coeff = bmbg_der2coeff_tensor[bb].data_ptr<double>();
            
            MtpBasisGrad<double>::find_der_backward(
                out_der2coeff,
                grad_output,
                mbg_der2coeff,
                binum[bb],
                numneigh,
                umax_num_neighs,
                num_coeffs);
        }
    }
}


torch::autograd::variable_list MtpBasisGradOp(
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
    return MtpBasisGradFunction::apply(
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
