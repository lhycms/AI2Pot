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



#if defined(USE_CUDA) or defined(__INTELLISENSE__)
#include "../gpu/include/linearMtp_torch_launcher.h"
#include "../gpu/include/linearMtpLoss_torch_launcher.h"
#include "../../correction/gpu/include/zbl_torch_launcher.h"

namespace ai2pot {
namespace mtpr {
// 1. linearMtp_torch_launcher.cu
// 1.1. find_efv_torch_launcher()
extern template void find_efv_torch_launcher<float>(
    float *d_etot_ptr,
    float (*d_force)[3],
    float *d_virial,
    int chebyshev_size,
    float *d_coeffs,
    float *d_linear_coeffs,
    float *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin);

extern template void find_efv_torch_launcher<double>(
    double *d_etot_ptr,
    double (*d_force)[3],
    double *d_virial,
    int chebyshev_size,
    double *d_coeffs,
    double *d_linear_coeffs,
    double *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin);


// 1.2. find_ef_torch_launcher()
extern template void ai2pot::mtpr::find_ef_torch_launcher<float>(
    float *d_etot_ptr,
    float (*d_force)[3],
    int chebyshev_size,
    float *d_coeffs,
    float *d_linear_coeffs,
    float *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin);

extern template void ai2pot::mtpr::find_ef_torch_launcher<double>(
    double *d_etot_ptr,
    double (*d_force)[3],
    int chebyshev_size,
    double *d_coeffs,
    double *d_linear_coeffs,
    double *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin);



// 2. linearMtpLoss_torch_launcher.cu
// 2.1. find_loss_torch_launcher()
extern template void ai2pot::mtpr::find_loss_torch_launcher<float>(
    float *d_loss_ptr,
    int inum,
    int *d_ilist,
    float e_weight,
    float f_weight,
    float v_weight,
    float etot_ml,
    float etot_dft,
    float (*d_force_ml)[3],
    float (*d_force_dft)[3],
    float *d_virial_ml,
    float *d_virial_dft);

extern template void ai2pot::mtpr::find_loss_torch_launcher<double>(
    double *d_loss_ptr,
    int inum,
    int *d_ilist,
    double e_weight,
    double f_weight,
    double v_weight,
    double etot_ml,
    double etot_dft,
    double (*d_force_ml)[3],
    double (*d_force_dft)[3],
    double *d_virial_ml,
    double *d_virial_dft);

// 2.2. find_ef_loss_torch_launcher()
extern template void ai2pot::mtpr::find_ef_loss_torch_launcher<float>(
    float *d_loss_ptr,
    int inum,
    int *d_ilist,
    float e_weight,
    float f_weight,
    float etot_ml,
    float etot_dft,
    float (*d_force_ml)[3],
    float (*d_force_dft)[3]);


extern template void ai2pot::mtpr::find_ef_loss_torch_launcher<double>(
    double *d_loss_ptr,
    int inum,
    int *d_ilist,
    double e_weight,
    double f_weight,
    double etot_ml,
    double etot_dft,
    double (*d_force_ml)[3],
    double (*d_force_dft)[3]);

// 2.3. find_loss_backward_torch_launcher()
extern template void ai2pot::mtpr::find_loss_backward_torch_launcher<float>(
    float *d_loss_der2coeffs,
    float *d_loss_der2linear_coeffs,
    float *d_loss_der2type_bias,
    float e_weight,
    float f_weight,
    float v_weight,
    float etot_ml,
    float etot_dft,
    float (*d_force_ml)[3],
    float (*d_force_dft)[3],
    float *d_virial_ml,
    float *d_virial_dft,
    int chebyshev_size,
    float *d_coeffs,
    float *d_linear_coeffs,
    float *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin);


extern template void ai2pot::mtpr::find_loss_backward_torch_launcher<double>(
    double *d_loss_der2coeffs,
    double *d_loss_der2linear_coeffs,
    double *d_loss_der2type_bias,
    double e_weight,
    double f_weight,
    double v_weight,
    double etot_ml,
    double etot_dft,
    double (*d_force_ml)[3],
    double (*d_force_dft)[3],
    double *d_virial_ml,
    double *d_virial_dft,
    int chebyshev_size,
    double *d_coeffs,
    double *d_linear_coeffs,
    double *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin);

// 2.4. find_ef_loss_backward_torch_launcher()
extern template void ai2pot::mtpr::find_ef_loss_backward_torch_launcher<float>(
    float *d_loss_der2coeffs,
    float *d_loss_der2linear_coeffs,
    float *d_loss_der2type_bias,
    float e_weight,
    float f_weight,
    float etot_ml,
    float etot_dft,
    float (*d_force_ml)[3],
    float (*d_force_dft)[3],
    int chebyshev_size,
    float *d_coeffs,
    float *d_linear_coeffs,
    float *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin);

extern template void ai2pot::mtpr::find_ef_loss_backward_torch_launcher<double>(
    double *d_loss_der2coeffs,
    double *d_loss_der2linear_coeffs,
    double *d_loss_der2type_bias,
    double e_weight,
    double f_weight,
    double etot_ml,
    double etot_dft,
    double (*d_force_ml)[3],
    double (*d_force_dft)[3],
    int chebyshev_size,
    double *d_coeffs,
    double *d_linear_coeffs,
    double *d_type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin);

};  // namespace : mtpr
};  // namespace : ai2pot



namespace ai2pot {
namespace correction {

extern template void ai2pot::correction::correct_zbl_efv_torch_launcher<float>(
    float *d_etot_ptr,
    float *d_force,
    float *d_virial,
    float rmax,
    float rmin,
    float *d_cks,
    float *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms);

extern template void ai2pot::correction::correct_zbl_efv_torch_launcher<double>(
    double *d_etot_ptr,
    double *d_force,
    double *d_virial,
    double rmax,
    double rmin,
    double *d_cks,
    double *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms);


extern template void ai2pot::correction::correct_zbl_ef_torch_launcher<float>(
    float *d_etot_ptr,
    float *d_force,
    float rmax,
    float rmin,
    float *d_cks,
    float *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    float (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms
);


extern template void ai2pot::correction::correct_zbl_ef_torch_launcher<double>(
    double *d_etot_ptr,
    double *d_force,
    double rmax,
    double rmin,
    double *d_cks,
    double *d_dks,
    int inum,
    int *d_ilist,
    int *d_numneigh,
    int *d_firstneigh,
    double (*d_rcs)[3],
    int *d_types,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms
);

};  // namespace : correction
};  // namespace : ai2pot

#endif






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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    int nbatches = (int)betot_dft_tensor.size(0);
    int num_atoms = bilist_tensor.size(1);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moment = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neighs = (int)bfirstneigh_tensor.size(2);
    int ntypes = (int)type_map_tensor.size(0);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bloss_tensor;
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
    at::Tensor tmp_etot_ml_tensor;
    at::Tensor tmp_force_ml_tensor;
    at::Tensor tmp_virial_ml_tensor;
#endif

    if (brcs_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(brcs_tensor.device());
        bloss_tensor = at::zeros({nbatches}, float_options);
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
        tmp_etot_ml_tensor = at::tensor(0, float_options);
        tmp_force_ml_tensor = at::zeros({num_atoms + nghost, 3}, float_options);
        tmp_virial_ml_tensor = at::zeros({9}, float_options);
#endif
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

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
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                tmp_etot_ml_tensor.zero_();
                tmp_force_ml_tensor.zero_();
                tmp_virial_ml_tensor.zero_();

                float *tmp_etot_ml_ptr = (float*)tmp_etot_ml_tensor.data_ptr<float>();
                float (*tmp_force_ml)[3] = (float (*)[3])tmp_force_ml_tensor.data_ptr<float>();
                float *tmp_virial_ml = (float*)tmp_virial_ml_tensor.data_ptr<float>();
                
                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_efv_torch_launcher(
                        tmp_etot_ml_ptr,
                        (float*)tmp_force_ml,
                        tmp_virial_ml,
                        (float)zbl_rmax,
                        (float)zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neighs);
                find_efv_torch_launcher(tmp_etot_ml_ptr,
                                        tmp_force_ml,
                                        tmp_virial_ml,
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
                                        type_map,
                                        umax_num_neighs,
                                        nghost,
                                        (float)rmax,
                                        (float)rmin);

                find_loss_torch_launcher(
                    loss,
                    inum,
                    ilist,
                    (float)e_weight,
                    (float)f_weight,
                    (float)v_weight,
                    tmp_etot_ml_tensor.item<float>(),
                    etot_dft,
                    tmp_force_ml,
                    force_dft,
                    tmp_virial_ml,
                    virial_dft);

#endif
            }
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(brcs_tensor.device());
        bloss_tensor = at::zeros({nbatches}, float_options);
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
        tmp_etot_ml_tensor = at::tensor(0, float_options);
        tmp_force_ml_tensor = at::zeros({num_atoms + nghost, 3}, float_options);
        tmp_virial_ml_tensor = at::zeros({9}, float_options);
#endif

        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

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
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                tmp_etot_ml_tensor.zero_();
                tmp_force_ml_tensor.zero_();
                tmp_virial_ml_tensor.zero_();

                double *tmp_etot_ml_ptr = (double*)tmp_etot_ml_tensor.data_ptr<double>();
                double (*tmp_force_ml)[3] = (double (*)[3])tmp_force_ml_tensor.data_ptr<double>();
                double *tmp_virial_ml = (double*)tmp_virial_ml_tensor.data_ptr<double>();

                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_efv_torch_launcher(
                        tmp_etot_ml_ptr,
                        (double*)tmp_force_ml,
                        tmp_virial_ml,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neighs);
                find_efv_torch_launcher(tmp_etot_ml_ptr,
                                        tmp_force_ml,
                                        tmp_virial_ml,
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
                                        type_map,
                                        umax_num_neighs,
                                        nghost,
                                        rmax,
                                        rmin);

                find_loss_torch_launcher(
                    loss,
                    inum,
                    ilist,
                    e_weight,
                    f_weight,
                    v_weight,
                    tmp_etot_ml_tensor.item<double>(),
                    etot_dft,
                    tmp_force_ml,
                    force_dft,
                    tmp_virial_ml,
                    virial_dft);
#endif
            }
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
                            type_map_tensor,
                            at::tensor(nghost, int_options),
                            at::tensor(rmax, float_options),
                            at::tensor(rmin, float_options),
                            at::tensor(zbl_rmax, float_options),
                            at::tensor(zbl_rmin, float_options),
                            zbl_cks_tensor,
                            zbl_dks_tensor
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
    at::Tensor type_map_tensor = ctx->get_saved_variables()[21];
    int ntypes = type_map_tensor.size(0);

    int nghost = ctx->get_saved_variables()[22].item<int>();
    double rmax = ctx->get_saved_variables()[23].item<double>();
    double rmin = ctx->get_saved_variables()[24].item<double>();
    double zbl_rmax = ctx->get_saved_variables()[25].item<double>();
    double zbl_rmin = ctx->get_saved_variables()[26].item<double>();
    at::Tensor zbl_cks_tensor = ctx->get_saved_variables()[27];
    at::Tensor zbl_dks_tensor = ctx->get_saved_variables()[28];
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    int num_linear_coeffs = (int)linear_coeffs_tensor.size(0);

    int nbatches = (int)bgrad_output_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moment = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neighs = (int)bfirstneigh_tensor.size(2);
    int num_atoms = (int)bilist_tensor.size(1);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bloss_der2coeffs_tensor;
    at::Tensor bloss_der2linear_coeffs_tensor;
    at::Tensor bloss_der2type_bias_tensor;
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
    at::Tensor tmp_etot_ml_tensor;
    at::Tensor tmp_force_ml_tensor;
    at::Tensor tmp_virial_ml_tensor;
#endif
    
    if (bgrad_output_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
        tmp_etot_ml_tensor = at::tensor(0, float_options);
        tmp_force_ml_tensor = at::zeros({num_atoms + nghost, 3}, float_options);
        tmp_virial_ml_tensor = at::zeros({9}, float_options);
#endif
        float* zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float* zbl_dks = zbl_dks_tensor.data_ptr<float>();

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
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                tmp_etot_ml_tensor.zero_();
                tmp_force_ml_tensor.zero_();
                tmp_virial_ml_tensor.zero_();

                float *tmp_etot_ml_ptr = (float*)tmp_etot_ml_tensor.data_ptr<float>();
                float (*tmp_force_ml)[3] = (float (*)[3])tmp_force_ml_tensor.data_ptr<float>();
                float *tmp_virial_ml = (float*)tmp_virial_ml_tensor.data_ptr<float>();

                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_efv_torch_launcher(
                        tmp_etot_ml_ptr,
                        (float*)tmp_force_ml,
                        tmp_virial_ml,
                        (float)zbl_rmax,
                        (float)zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neighs);
                find_efv_torch_launcher(
                    tmp_etot_ml_ptr,
                    tmp_force_ml,
                    tmp_virial_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    (float)rmax,
                    (float)rmin);

                find_loss_backward_torch_launcher(
                    loss_der2coeffs,
                    loss_der2linear_coeffs,
                    loss_der2type_bias,
                    (float)e_weight,
                    (float)f_weight,
                    (float)v_weight,
                    tmp_etot_ml_tensor.item<float>(),
                    etot_dft,
                    tmp_force_ml,
                    force_dft,
                    tmp_virial_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    (float)rmax,
                    (float)rmin);
#endif
            }
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);
#if defined(USE_CUDA) or defined (__INTELLISENSE__)
        tmp_etot_ml_tensor = at::tensor(0, float_options);
        tmp_force_ml_tensor = at::zeros({num_atoms + nghost, 3}, float_options);
        tmp_virial_ml_tensor = at::zeros({9}, float_options);
#endif

        double* zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double* zbl_dks = zbl_dks_tensor.data_ptr<double>();

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
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                tmp_etot_ml_tensor.zero_();
                tmp_force_ml_tensor.zero_();
                tmp_virial_ml_tensor.zero_();
                double *tmp_etot_ml_ptr = (double*)tmp_etot_ml_tensor.data_ptr<double>();
                double (*tmp_force_ml)[3] = (double (*)[3])tmp_force_ml_tensor.data_ptr<double>();
                double *tmp_virial_ml = (double*)tmp_virial_ml_tensor.data_ptr<double>();

                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_efv_torch_launcher(
                        tmp_etot_ml_ptr,
                        (double*)tmp_force_ml,
                        tmp_virial_ml,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neighs);
                find_efv_torch_launcher(
                    tmp_etot_ml_ptr,
                    tmp_force_ml,
                    tmp_virial_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin);

                find_loss_backward_torch_launcher(
                    loss_der2coeffs,
                    loss_der2linear_coeffs,
                    loss_der2type_bias,
                    e_weight,
                    f_weight,
                    v_weight,
                    tmp_etot_ml_tensor.item<double>(),
                    etot_dft,
                    tmp_force_ml,
                    force_dft,
                    tmp_virial_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin);
#endif
            }
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
            torch::matmul(bgrad_output_tensor, bloss_der2coeffs_tensor),
            torch::matmul(bgrad_output_tensor, bloss_der2linear_coeffs_tensor),
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    int nbatches = (int)betot_dft_tensor.size(0);
    int num_atoms = (int)bilist_tensor.size(1);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moment = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neighs = (int)bfirstneigh_tensor.size(2);
    int ntypes = (int)type_map_tensor.size(0);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bloss_tensor;
#if defined(USE_CUDA) or (__INTELLISENSE__)
    at::Tensor tmp_etot_ml_tensor;
    at::Tensor tmp_force_ml_tensor;
#endif

    if (brcs_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(brcs_tensor.device());
        bloss_tensor = at::zeros({nbatches}, float_options);
#if defined(USE_CUDA) or (__INTELLISENSE__)
        tmp_etot_ml_tensor = at::tensor(0, float_options);
        tmp_force_ml_tensor = at::zeros({num_atoms + nghost, 3}, float_options);
#endif
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

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
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                tmp_etot_ml_tensor.zero_();
                tmp_force_ml_tensor.zero_();

                float *tmp_etot_ml_ptr = (float*)tmp_etot_ml_tensor.data_ptr<float>();
                float (*tmp_force_ml)[3] = (float (*)[3])tmp_force_ml_tensor.data_ptr<float>();


                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_ef_torch_launcher(
                        tmp_etot_ml_ptr,
                        (float*)tmp_force_ml,
                        (float)zbl_rmax,
                        (float)zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neighs);
                find_ef_torch_launcher(
                    tmp_etot_ml_ptr,
                    tmp_force_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    (float)rmax,
                    (float)rmin);

                find_ef_loss_torch_launcher(
                    loss,
                    inum,
                    ilist,
                    (float)e_weight,
                    (float)f_weight,
                    tmp_etot_ml_tensor.item<float>(),
                    etot_dft,
                    tmp_force_ml,
                    force_dft);

#endif
            }
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(brcs_tensor.device());
        bloss_tensor = at::zeros({nbatches}, float_options);
#if defined(USE_CUDA) or (__INTELLISENSE__)
        tmp_etot_ml_tensor = at::tensor(0, float_options);
        tmp_force_ml_tensor = at::zeros({num_atoms + nghost, 3}, float_options);
#endif
        double* zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double* zbl_dks = zbl_dks_tensor.data_ptr<double>();

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
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
                LinearMtp<double>::find_ef_loss(
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                tmp_etot_ml_tensor.zero_();
                tmp_force_ml_tensor.zero_();

                double *tmp_etot_ml_ptr = tmp_etot_ml_tensor.data_ptr<double>();
                double (*tmp_force_ml)[3] = (double (*)[3])tmp_force_ml_tensor.data_ptr<double>();

                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_ef_torch_launcher(
                        tmp_etot_ml_ptr,
                        (double*)tmp_force_ml,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neighs);
                find_ef_torch_launcher(
                    tmp_etot_ml_ptr,
                    tmp_force_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin);

                find_ef_loss_torch_launcher(
                    loss,
                    inum,
                    ilist,
                    e_weight,
                    f_weight,
                    tmp_etot_ml_tensor.item<double>(),
                    etot_dft,
                    tmp_force_ml,
                    force_dft);
#endif
            }
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
                            type_map_tensor,
                            at::tensor(nghost, int_options),
                            at::tensor(rmax, float_options),
                            at::tensor(rmin, float_options),
                            at::tensor(zbl_rmax, float_options),
                            at::tensor(zbl_rmin, float_options),
                            zbl_cks_tensor,
                            zbl_dks_tensor
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
    at::Tensor type_map_tensor = ctx->get_saved_variables()[19];
    int ntypes = type_map_tensor.size(0);
    int nghost = ctx->get_saved_variables()[20].item<int>();
    double rmax = ctx->get_saved_variables()[21].item<double>();
    double rmin = ctx->get_saved_variables()[22].item<double>();
    double zbl_rmax = ctx->get_saved_variables()[23].item<double>();
    double zbl_rmin = ctx->get_saved_variables()[24].item<double>();
    at::Tensor zbl_cks_tensor = ctx->get_saved_variables()[25];
    at::Tensor zbl_dks_tensor = ctx->get_saved_variables()[26];
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    int num_linear_coeffs = (int)linear_coeffs_tensor.size(0);

    int nbatches = (int)bgrad_output_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moment = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neighs = (int)bfirstneigh_tensor.size(2);
    int num_atoms = (int)bilist_tensor.size(1);

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    at::Tensor bloss_der2coeffs_tensor;
    at::Tensor bloss_der2linear_coeffs_tensor;
    at::Tensor bloss_der2type_bias_tensor;
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
    at::Tensor tmp_etot_ml_tensor;
    at::Tensor tmp_force_ml_tensor;
#endif
    
    if (bgrad_output_tensor.dtype() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
        tmp_etot_ml_tensor = at::tensor(0, float_options);
        tmp_force_ml_tensor = at::zeros({num_atoms + nghost, 3}, float_options);
#endif

        float* zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float* zbl_dks = zbl_dks_tensor.data_ptr<float>();

        for (int bb=0; bb<nbatches; bb++) {
            float *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<float>();
            float *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<float>();
            float *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<float>();
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
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                tmp_etot_ml_tensor.zero_();
                tmp_force_ml_tensor.zero_();

                float *tmp_etot_ml_ptr = (float*)tmp_etot_ml_tensor.data_ptr<float>();
                float (*tmp_force_ml)[3] = (float (*)[3])tmp_force_ml_tensor.data_ptr<float>();

                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_ef_torch_launcher(
                        tmp_etot_ml_ptr,
                        (float*)tmp_force_ml,
                        (float)zbl_rmax,
                        (float)zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neighs);
                find_ef_torch_launcher(
                    tmp_etot_ml_ptr,
                    tmp_force_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    (float)rmax,
                    (float)rmin);
                
                find_ef_loss_backward_torch_launcher(
                    loss_der2coeffs,
                    loss_der2linear_coeffs,
                    loss_der2type_bias,
                    (float)e_weight,
                    (float)f_weight,
                    tmp_etot_ml_tensor.item<float>(),
                    etot_dft,
                    tmp_force_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    (float)rmax,
                    (float)rmin);
#endif
            }
        }
    } else {
        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(bgrad_output_tensor.device());
        bloss_der2coeffs_tensor = at::zeros({nbatches, num_coeffs}, float_options);
        bloss_der2linear_coeffs_tensor = at::zeros({nbatches, num_linear_coeffs}, float_options);
        bloss_der2type_bias_tensor = at::zeros({nbatches, ntypes}, float_options);
#if defined(USE_CUDA) or defined (__INTELLISENSE__)
        tmp_etot_ml_tensor = at::tensor(0, float_options);
        tmp_force_ml_tensor = at::zeros({num_atoms+nghost, 3}, float_options);
#endif

        double* zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double* zbl_dks = zbl_dks_tensor.data_ptr<double>();

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
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                tmp_etot_ml_tensor.zero_();
                tmp_force_ml_tensor.zero_();

                double *tmp_etot_ml_ptr = (double*)tmp_etot_ml_tensor.data_ptr<double>();
                double (*tmp_force_ml)[3] = (double (*)[3])tmp_force_ml_tensor.data_ptr<double>();

                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_ef_torch_launcher(
                        tmp_etot_ml_ptr,
                        (double*)tmp_force_ml,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neighs);
                find_ef_torch_launcher(
                    tmp_etot_ml_ptr,
                    tmp_force_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin);

                find_ef_loss_backward_torch_launcher(
                    loss_der2coeffs,
                    loss_der2linear_coeffs,
                    loss_der2type_bias,
                    e_weight,
                    f_weight,
                    tmp_etot_ml_tensor.item<double>(),
                    etot_dft,
                    tmp_force_ml,
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
                    type_map,
                    umax_num_neighs,
                    nghost,
                    rmax,
                    rmin);
#endif
            }
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
            torch::matmul(bgrad_output_tensor, bloss_der2coeffs_tensor),
            torch::matmul(bgrad_output_tensor, bloss_der2linear_coeffs_tensor),
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



torch::autograd::variable_list LinearMtpToEFVFunction::forward(
    torch::autograd::AutogradContext *ctx,
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor
)
{
    // 
    int nbatches = binum_tensor.size(0);
    int natoms = bnumneigh_tensor.size(1);  // not inum
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);

    //
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    // 
    at::Tensor betot_tensor;
    at::Tensor bforce_tensor;
    at::Tensor bvirial_tensor;

    // 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                        .dtype(brcs_tensor.scalar_type())
                        .device(brcs_tensor.device());
        betot_tensor = at::zeros({nbatches}, float_options);
        bforce_tensor = at::zeros({nbatches, natoms+nghost, 3}, float_options);
        bvirial_tensor = at::zeros({nbatches, 9}, float_options);

        float* coeffs = coeffs_tensor.data_ptr<float>();
        float* linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float* type_bias = type_bias_tensor.data_ptr<float>();
        int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
        float* zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float* zbl_dks = zbl_dks_tensor.data_ptr<float>();

        for (int bb=0; bb<nbatches; bb++) {
            float* etot_ptr = betot_tensor[bb].data_ptr<float>();
            float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
            float *virial = bvirial_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = btypes_tensor[bb].data_ptr<int>();
            int *type_map = type_map_tensor.data_ptr<int>();
            
            if (brcs_tensor.device() == c10::kCPU) {
                LinearMtp<float>::find_efv(
                    *etot_ptr,
                    force,
                    virial,
                    chebyshev_size,
                    coeffs,
                    linear_coeffs,
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
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_efv_torch_launcher(
                        etot_ptr,
                        (float*)force,
                        virial,
                        (float)zbl_rmax,
                        (float)zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms);

                find_efv_torch_launcher(
                    etot_ptr,
                    force,
                    virial,
                    chebyshev_size,
                    coeffs,
                    linear_coeffs,
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
                    (float)rmax,
                    (float)rmin);
#endif
            }
        }
    } else {
        float_options = c10::TensorOptions()
                        .dtype(brcs_tensor.scalar_type())
                        .device(brcs_tensor.device());
        betot_tensor = at::zeros({nbatches}, float_options);
        bforce_tensor = at::zeros({nbatches, natoms+nghost, 3}, float_options);
        bvirial_tensor = at::zeros({nbatches, 9}, float_options);

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();


        for (int bb=0; bb<nbatches; bb++) {
            double *etot_ptr = betot_tensor[bb].data_ptr<double>();
            double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
            double *virial = bvirial_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = btypes_tensor[bb].data_ptr<int>();
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
                LinearMtp<double>::find_efv(
                    *etot_ptr,
                    force,
                    virial,
                    chebyshev_size,
                    coeffs,
                    linear_coeffs,
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
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_efv_torch_launcher(
                        etot_ptr,
                        (double*)force,
                        virial,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms);

                find_efv_torch_launcher(
                    etot_ptr,
                    force,
                    virial,
                    chebyshev_size,
                    coeffs,
                    linear_coeffs,
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
                    rmin);
#endif
            }
        }
    }

    float_options = c10::TensorOptions()
                    .dtype(torch::kFloat64)
                    .device(brcs_tensor.device());
    ctx->save_for_backward({
        at::tensor(chebyshev_size, float_options),
        coeffs_tensor,
        linear_coeffs_tensor,
        type_bias_tensor,
        at::tensor(alpha_moments_count, float_options),
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
        type_map_tensor,
        at::tensor(nghost, int_options),
        at::tensor(rmax, float_options),
        at::tensor(rmin, float_options),
        at::tensor(zbl_rmax, float_options),
        at::tensor(zbl_rmin, float_options),
        zbl_cks_tensor,
        zbl_dks_tensor});

    if (brcs_tensor.requires_grad())
        betot_tensor.requires_grad_(true);

    return {betot_tensor, bforce_tensor, bvirial_tensor};
}


torch::autograd::variable_list LinearMtpToEFVFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    return {at::Tensor(),
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


torch::autograd::variable_list LinearMtpToEFFunction::forward(
    torch::autograd::AutogradContext *ctx,
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    // 
    int nbatches = binum_tensor.size(0);
    int natoms = bnumneigh_tensor.size(1);  // not inum
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);

    //
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    // 
    at::Tensor betot_tensor;
    at::Tensor bforce_tensor;
    at::Tensor bvirial_tensor;

    // 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                        .dtype(brcs_tensor.scalar_type())
                        .device(brcs_tensor.device());
        betot_tensor = at::zeros({nbatches}, float_options);
        bforce_tensor = at::zeros({nbatches, natoms+nghost, 3}, float_options);

        float* coeffs = coeffs_tensor.data_ptr<float>();
        float* linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float* type_bias = type_bias_tensor.data_ptr<float>();
        int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
        float* zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float* zbl_dks = zbl_dks_tensor.data_ptr<float>();

        for (int bb=0; bb<nbatches; bb++) {
            float* etot_ptr = betot_tensor[bb].data_ptr<float>();
            float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = btypes_tensor[bb].data_ptr<int>();
            int *type_map = type_map_tensor.data_ptr<int>();
            
            if (brcs_tensor.device() == c10::kCPU) {
                LinearMtp<float>::find_ef(
                    *etot_ptr,
                    force,
                    chebyshev_size,
                    coeffs,
                    linear_coeffs,
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
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_ef_torch_launcher(
                        etot_ptr,
                        (float*)force,
                        (float)zbl_rmax,
                        (float)zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms);

                find_ef_torch_launcher(
                    etot_ptr,
                    force,
                    chebyshev_size,
                    coeffs,
                    linear_coeffs,
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
                    (float)rmax,
                    (float)rmin);
#endif
            }
        }
    } else {
        float_options = c10::TensorOptions()
                        .dtype(brcs_tensor.scalar_type())
                        .device(brcs_tensor.device());
        betot_tensor = at::zeros({nbatches}, float_options);
        bforce_tensor = at::zeros({nbatches, natoms+nghost, 3}, float_options);

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();


        for (int bb=0; bb<nbatches; bb++) {
            double *etot_ptr = betot_tensor[bb].data_ptr<double>();
            double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = btypes_tensor[bb].data_ptr<int>();
            int *type_map = type_map_tensor.data_ptr<int>();

            if (brcs_tensor.device() == c10::kCPU) {
                LinearMtp<double>::find_ef(
                    *etot_ptr,
                    force,
                    chebyshev_size,
                    coeffs,
                    linear_coeffs,
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
            } else {
#if defined(USE_CUDA) or defined(__INTELLISENSE__)
                if (zbl_rmax > 0.0)
                    ai2pot::correction::correct_zbl_ef_torch_launcher(
                        etot_ptr,
                        (double*)force,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms);

                find_ef_torch_launcher(
                    etot_ptr,
                    force,
                    chebyshev_size,
                    coeffs,
                    linear_coeffs,
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
                    rmin);
#endif
            }
        }
    }

    float_options = c10::TensorOptions()
                    .dtype(torch::kFloat64)
                    .device(brcs_tensor.device());
    ctx->save_for_backward({
        at::tensor(chebyshev_size, float_options),
        coeffs_tensor,
        linear_coeffs_tensor,
        type_bias_tensor,
        at::tensor(alpha_moments_count, float_options),
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
        type_map_tensor,
        at::tensor(nghost, int_options),
        at::tensor(rmax, float_options),
        at::tensor(rmin, float_options),
        at::tensor(zbl_rmax, float_options),
        at::tensor(zbl_rmin, float_options),
        zbl_cks_tensor,
        zbl_dks_tensor});

    if (brcs_tensor.requires_grad())
        betot_tensor.requires_grad_(true);

    return {betot_tensor, bforce_tensor};
}


torch::autograd::variable_list LinearMtpToEFFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    return {at::Tensor(),
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


torch::autograd::variable_list LinearMtpToDescriptorsFunction::forward(
    torch::autograd::AutogradContext *ctx,
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin)
{
    // 1. 
    int nbatches = binum_tensor.size(0);
    int num_atoms = bfirstneigh_tensor.size(1);
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);

    // 2. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    // 3. 
    at::Tensor bdescriptors_tensor;

    // 4.
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(brcs_tensor.device());
        bdescriptors_tensor = at::zeros({nbatches, num_atoms, alpha_scalar_moments}, float_options);
        const int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        const int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        const int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
        float *coeffs = (float*)coeffs_tensor.data_ptr<float>();
        float *linear_coeffs = (float*)linear_coeffs_tensor.data_ptr<float>();
        float *type_bias = (float*)type_bias_tensor.data_ptr<float>();

        for (int bb=0; bb<nbatches; bb++) {
            float *descriptors = (float*)bdescriptors_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();
            int *type_map = (int*)type_map_tensor.data_ptr<int>();

            LinearMtp<float>::find_descriptors(
                descriptors,
                chebyshev_size,
                coeffs,
                linear_coeffs,
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
                (float)rmax,
                (float)rmin);
        }

    } else {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat64)
                        .device(brcs_tensor.device());
        bdescriptors_tensor = at::zeros({nbatches, num_atoms, alpha_scalar_moments}, float_options);
        const int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        const int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        const int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
        double *coeffs = (double*)coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = (double*)linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = (double*)type_bias_tensor.data_ptr<double>();

        for (int bb=0; bb<nbatches; bb++) {
            double *descriptors = (double*)bdescriptors_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();
            int *type_map = (int*)type_map_tensor.data_ptr<int>();

            LinearMtp<double>::find_descriptors(
                descriptors,
                chebyshev_size,
                coeffs,
                linear_coeffs,
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
                (double)rmax,
                (double)rmin);
        }
    }


    float_options = c10::TensorOptions()
                    .dtype(torch::kFloat64)
                    .device(brcs_tensor.device());
    ctx->save_for_backward({
        torch::tensor(chebyshev_size, int_options),
        coeffs_tensor,
        linear_coeffs_tensor,
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
        torch::tensor(rmin, float_options)});

   return {bdescriptors_tensor};
}



torch::autograd::variable_list LinearMtpToDescriptorsFunction::backward(
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
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
        type_map_tensor,
        nghost,
        rmax,
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
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
        type_map_tensor,
        nghost,
        rmax,
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
}


torch::autograd::variable_list LinearMtpToEFVOp(
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    return LinearMtpToEFVFunction::apply(
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
        type_map_tensor,
        nghost,
        rmax,
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
}


torch::autograd::variable_list LinearMtpToEFOp(
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    return LinearMtpToEFFunction::apply(
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
        type_map_tensor,
        nghost,
        rmax,
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
}


torch::autograd::variable_list LinearMtpToDescriptorsOp(
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
    const at::Tensor& type_map_tensor,
    int nghost,
    double rmax,
    double rmin)
{
    return LinearMtpToDescriptorsFunction::apply(
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
        type_map_tensor,
        nghost,
        rmax,
        rmin);
}


};  // namespace : mtpr
};  // namespace : ai2pot
