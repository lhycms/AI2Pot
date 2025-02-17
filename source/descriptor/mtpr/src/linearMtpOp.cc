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
    int nbatches = betot_dft_tensor.size(0);
    int alpha_index_basic_count = alpha_index_basic.size(0);
    int alpha_index_times_count = alpha_index_times.size(0);
    int alpha_scalar_moment = alpha_moment_mapping.size(0);
    int umax_num_neighs = bfirstneigh_tensor.size(2);

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
            float *force_dft = bforce_dft_tensor[bb].data_ptr<float>();
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
            float *rcs = brcs_tensor[bb].data_ptr<float>();
            int *types = btypes_tensor[bb].data_ptr<int>();


            LinearMtp<float>::find_loss(
                loss,
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
                (float (*)[3])rcs,
                types,
                ntypes,
                umax_num_neighs,
                nghost,
                rmax,
                rmin);
        }
    } else {
        
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


};  // namespace : mtpr
};  // namespace : ai2pot
