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
#include "../include/linearMtp_cpu_launcher.h"
#include "../include/linearMtpLoss_cpu_launcher.h"
#include "../include/linearMtpLoss.h"
#include "../include/linearMtpOp.h"
#include "../../correction/include/zbl.h"



#if defined(USE_CUDA) or defined(__INTELLISENSE__)
#include "../gpu/include/linearMtp_torch_launcher.h"
#include "../gpu/include/linearMtpLoss_torch_launcher.h"
#include "../../correction/gpu/include/zbl_torch_launcher.h"

namespace ai2pot {
namespace mtpr {
// 1. linearMtp_torch_launcher.cu
// 1.1. find_efv_torch_launcher()
extern template void ai2pot::mtpr::find_efv_torch_launcher<float>(
    float *d_betot_ptr,
    float (*d_bforce)[3],
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin,
    float *d_q_scaler);

extern template void ai2pot::mtpr::find_efv_torch_launcher<double>(
    double *d_betot_ptr,
    double (*d_bforce)[3],
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin,
    double *d_q_scaler);


// 1.2. find_ef_torch_launcher()
extern template void ai2pot::mtpr::find_ef_torch_launcher<float>(
    float *d_betot_ptr,
    float (*d_bforce)[3],
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin,
    float *d_q_scaler);

extern template void ai2pot::mtpr::find_ef_torch_launcher<double>(
    double *d_betot_ptr,
    double (*d_bforce)[3],
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin,
    double *d_q_scaler);


// 1.3. find_descriptors_torch_launcher()
extern template void ai2pot::mtpr::find_descriptors_torch_launcher<float>(
    float *d_bdescriptors,
    int chebyshev_size,
    float *d_coeffs,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin);


extern template void ai2pot::mtpr::find_descriptors_torch_launcher<double>(
    double *d_bdescriptors,
    int chebyshev_size,
    double *d_coeffs,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*d_alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*d_alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *d_alpha_moment_mapping,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin);



// 2. linearMtpLoss_torch_launcher.cu
// 2.1. find_loss_torch_launcher()
extern template void ai2pot::mtpr::find_loss_torch_launcher<float>(
    float *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    float e_weight,
    float f_weight,
    float v_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3],
    float *d_bvirial_ml,
    float *d_bvirial_dft);

extern template void ai2pot::mtpr::find_loss_torch_launcher<double>(
    double *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    double e_weight,
    double f_weight,
    double v_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3],
    double *d_bvirial_ml,
    double *d_bvirial_dft);

// 2.2. find_ef_loss_torch_launcher()
extern template void ai2pot::mtpr::find_ef_loss_torch_launcher<float>(
    float *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    float e_weight,
    float f_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3]);


extern template void ai2pot::mtpr::find_ef_loss_torch_launcher<double>(
    double *d_bloss_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    double e_weight,
    double f_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3]);


// 2.3. find_loss_backward_torch_launcher()
extern template void ai2pot::mtpr::find_loss_backward_torch_launcher<float>(
    float *d_bloss_der2coeffs,
    float *d_bloss_der2linear_coeffs,
    float *d_bloss_der2type_bias,
    float e_weight,
    float f_weight,
    float v_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3],
    float *d_bvirial_ml,
    float *d_bvirial_dft,
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin,
    float *d_q_scaler);


extern template void ai2pot::mtpr::find_loss_backward_torch_launcher<double>(
    double *d_bloss_der2coeffs,
    double *d_bloss_der2linear_coeffs,
    double *d_bloss_der2type_bias,
    double e_weight,
    double f_weight,
    double v_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3],
    double *d_bvirial_ml,
    double *d_bvirial_dft,
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin,
    double *d_q_scaler);


// 2.4. find_ef_loss_backward_torch_launcher()
extern template void ai2pot::mtpr::find_ef_loss_backward_torch_launcher<float>(
    float *d_bloss_der2coeffs,
    float *d_bloss_der2linear_coeffs,
    float *d_bloss_der2type_bias,
    float e_weight,
    float f_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3],
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin,
    float *d_q_scaler);

extern template void ai2pot::mtpr::find_ef_loss_backward_torch_launcher<double>(
    double *d_bloss_der2coeffs,
    double *d_bloss_der2linear_coeffs,
    double *d_bloss_der2type_bias,
    double e_weight,
    double f_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3],
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
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin,
    double *d_q_scaler);


// 2.5.1. find_num_real_atoms_in_batch
extern void find_num_real_atoms_in_batch_torch_launcher(
    int *d_num_real_atoms_in_batch_ptr,
    int batch_size,
    int *d_binum);

// 2.5.2. find_e_se
extern template void find_e_se_torch_launcher(
    float *d_e_se_ptr,
    int batch_size,
    int *d_binum,
    float *d_betot_ml,
    float *d_betot_dft);

extern template void find_e_se_torch_launcher(
    double *d_e_se_ptr,
    int batch_size,
    int *d_binum,
    double *d_betot_ml,
    double *d_betot_dft);

// 2.5.3. find_f_se
extern template void find_f_se_torch_launcher(
    float *d_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3]);

extern template void find_f_se_torch_launcher(
    double *d_f_se_ptr,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3]);

// 2.5.4. find_v_se
extern template void find_v_se_torch_launcher(
    float *d_v_se_ptr,
    int batch_size,
    float *d_bvirial_ml,
    float *d_bvirial_dft);

extern template void find_v_se_torch_launcher(
    double *d_v_se_ptr,
    int batch_size,
    double *d_bvirial_ml,
    double *d_bvirial_dft);


};  // namespace : mtpr
};  // namespace : ai2pot



namespace ai2pot {
namespace correction {

extern template void ai2pot::correction::correct_zbl_efv_torch_launcher<float>(
    float *d_betot_ptr,
    float *d_bforce,
    float *d_bvirial,
    float rmax,
    float rmin,
    float *d_cks,
    float *d_dks,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost);

extern template void ai2pot::correction::correct_zbl_efv_torch_launcher<double>(
    double *d_betot_ptr,
    double *d_bforce,
    double *d_bvirial,
    double rmax,
    double rmin,
    double *d_cks,
    double *d_dks,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost);


extern template void ai2pot::correction::correct_zbl_ef_torch_launcher<float>(
    float *d_betot_ptr,
    float *d_bforce,
    float rmax,
    float rmin,
    float *d_cks,
    float *d_dks,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    float (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost);


extern template void ai2pot::correction::correct_zbl_ef_torch_launcher<double>(
    double *d_betot_ptr,
    double *d_bforce,
    double rmax,
    double rmin,
    double *d_cks,
    double *d_dks,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int *d_bnumneigh,
    int *d_bfirstneigh,
    double (*d_brcs)[3],
    int *d_btypes,
    int ntypes,
    int *d_type_map,
    int umax_num_neigh_atoms,
    int nghost);

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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    int batch_size = (int)betot_dft_tensor.size(0);
    int natoms_pad = (int)bilist_tensor.size(1);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = (int)bfirstneigh_tensor.size(2);
    int ntypes = (int)type_map_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
    int *type_map = type_map_tensor.data_ptr<int>();

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                       .dtype(brcs_tensor.scalar_type())
                                       .device(brcs_tensor.device());

    at::Tensor bloss_tensor = at::zeros({batch_size}, float_options);
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);
    at::Tensor bvirial_tensor = at::zeros({batch_size, 9}, float_options);

    if (brcs_tensor.dtype() == torch::kFloat32) {
        float *bloss = bloss_tensor.data_ptr<float>();
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();
        float *bvirial = bvirial_tensor.data_ptr<float>();
        float *betot_dft = betot_dft_tensor.data_ptr<float>();
        float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();
        float *bvirial_dft = bvirial_dft_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float *loss_ptr = &(bloss_tensor.data_ptr<float>()[bb]);    // note
                float *etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);  // note
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                float *virial = bvirial_tensor[bb].data_ptr<float>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<float> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_efv(
                        (*etot_ptr),
                        (float*)force,
                        virial,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }
            find_efv_cpu_launcher<float>(
                betot,
                bforce,
                bvirial,
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
            
            find_loss_cpu_launcher<float>(
                bloss,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                e_weight,
                f_weight,
                v_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                bvirial,
                bvirial_dft);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_efv_torch_launcher(
                    betot,
                    (float*)bforce,
                    bvirial,
                    (float)rmax,
                    (float)rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_efv_torch_launcher(
                betot,
                bforce,
                bvirial,
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
                (float)rmax,
                (float)rmin,
                q_scaler);
            
            find_loss_torch_launcher(
                bloss,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                (float)e_weight,
                (float)f_weight,
                (float)v_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                bvirial,
                bvirial_dft);
            #endif
        }
    } else {
        double *bloss = bloss_tensor.data_ptr<double>();
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();
        double *bvirial = bvirial_tensor.data_ptr<double>();
        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();
        double *bvirial_dft = bvirial_dft_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();
        
        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *loss_ptr = &(bloss_tensor.data_ptr<double>()[bb]);
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                double *virial = bvirial_tensor[bb].data_ptr<double>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<double> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_efv(
                        (*etot_ptr),
                        (double*)force,
                        virial,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }
            find_efv_cpu_launcher<double>(
                betot,
                bforce,
                bvirial,
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
            
            find_loss_cpu_launcher<double>(
                bloss,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                e_weight,
                f_weight,
                v_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                bvirial,
                bvirial_dft);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_efv_torch_launcher(
                    betot,
                    (double*)bforce,
                    bvirial,
                    rmax,
                    rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_efv_torch_launcher(
                betot,
                bforce,
                bvirial,
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
            
            find_loss_torch_launcher(
                bloss,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                e_weight,
                f_weight,
                v_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                bvirial,
                bvirial_dft);
            #endif
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
                            q_scaler_tensor,
                            at::tensor(zbl_rmax, float_options),
                            at::tensor(zbl_rmin, float_options),
                            zbl_cks_tensor,
                            zbl_dks_tensor
                            });

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
    at::Tensor q_scaler_tensor = ctx->get_saved_variables()[25];
    double zbl_rmax = ctx->get_saved_variables()[26].item<double>();
    double zbl_rmin = ctx->get_saved_variables()[27].item<double>();
    at::Tensor zbl_cks_tensor = ctx->get_saved_variables()[28];
    at::Tensor zbl_dks_tensor = ctx->get_saved_variables()[29];
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    int num_linear_coeffs = (int)linear_coeffs_tensor.size(0);

    int batch_size = (int)bgrad_output_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = (int)bfirstneigh_tensor.size(2);
    int natoms_pad = (int)bilist_tensor.size(1);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
    int *type_map = type_map_tensor.data_ptr<int>();

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                       .dtype(brcs_tensor.scalar_type())
                                       .device(brcs_tensor.device());

    at::Tensor bloss_der2coeffs_tensor = at::zeros({batch_size, num_coeffs}, float_options);
    at::Tensor bloss_der2linear_coeffs_tensor = at::zeros({batch_size, num_linear_coeffs}, float_options);
    at::Tensor bloss_der2type_bias_tensor = at::zeros({batch_size, ntypes}, float_options);
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);
    at::Tensor bvirial_tensor = at::zeros({batch_size, 9}, float_options);
    
    if (bgrad_output_tensor.dtype() == torch::kFloat32) {
        float *bloss_der2coeffs = bloss_der2coeffs_tensor.data_ptr<float>();
        float *bloss_der2linear_coeffs = bloss_der2linear_coeffs_tensor.data_ptr<float>();
        float *bloss_der2type_bias = bloss_der2type_bias_tensor.data_ptr<float>();
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();
        float *bvirial = bvirial_tensor.data_ptr<float>();
        float *betot_dft = betot_dft_tensor.data_ptr<float>();
        float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();
        float *bvirial_dft = bvirial_dft_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float* zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float* zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<float>();
                float *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<float>();
                float *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<float>();
                float *etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                float *virial = bvirial_tensor[bb].data_ptr<float>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<float> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_efv(
                        (*etot_ptr),
                        (float*)force,
                        virial,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }
            find_efv_cpu_launcher<float>(
                betot,
                bforce,
                bvirial,
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
            
            find_loss_backward_cpu_launcher<float>(
                bloss_der2coeffs,
                bloss_der2linear_coeffs,
                bloss_der2type_bias,
                e_weight,
                f_weight,
                v_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                bvirial,
                bvirial_dft,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_efv_torch_launcher(
                    betot,
                    (float*)bforce,
                    bvirial,
                    (float)zbl_rmax,
                    (float)zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_efv_torch_launcher(
                betot,
                bforce,
                bvirial,
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
                (float)rmax,
                (float)rmin,
                q_scaler);
            
            find_loss_backward_torch_launcher(
                bloss_der2coeffs,
                bloss_der2linear_coeffs,
                bloss_der2type_bias,
                (float)e_weight,
                (float)f_weight,
                (float)v_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                bvirial,
                bvirial_dft,
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
                (float)rmax,
                (float)rmin,
                q_scaler);
            #endif
        }
    } else {
        double *bloss_der2coeffs = bloss_der2coeffs_tensor.data_ptr<double>();
        double *bloss_der2linear_coeffs = bloss_der2linear_coeffs_tensor.data_ptr<double>();
        double *bloss_der2type_bias = bloss_der2type_bias_tensor.data_ptr<double>();
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();
        double *bvirial = bvirial_tensor.data_ptr<double>();
        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();
        double *bvirial_dft = bvirial_dft_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double* zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double* zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<double>();
                double *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<double>();
                double *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<double>();
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double etot_dft = betot_dft_tensor[bb].item<double>();
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
                double *virial = bvirial_tensor[bb].data_ptr<double>();
                double *virial_dft = bvirial_dft_tensor[bb].data_ptr<double>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<double> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_efv(
                        (*etot_ptr),
                        (double*)force,
                        virial,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }
            find_efv_cpu_launcher<double>(
                betot,
                bforce,
                bvirial,
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
            
            find_loss_backward_cpu_launcher<double>(
                bloss_der2coeffs,
                bloss_der2linear_coeffs,
                bloss_der2type_bias,
                e_weight,
                f_weight,
                v_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                bvirial,
                bvirial_dft,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_efv_torch_launcher(
                    betot,
                    (double*)bforce,
                    bvirial,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_efv_torch_launcher(
                betot,
                bforce,
                bvirial,
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
            
            find_loss_backward_torch_launcher(
                bloss_der2coeffs,
                bloss_der2linear_coeffs,
                bloss_der2type_bias,
                e_weight,
                f_weight,
                v_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
                bvirial,
                bvirial_dft,
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
            #endif
        }
    }

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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    int batch_size = (int)betot_dft_tensor.size(0);
    int natoms_pad = (int)bilist_tensor.size(1);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = (int)bfirstneigh_tensor.size(2);
    int ntypes = (int)type_map_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
    int *type_map = type_map_tensor.data_ptr<int>();

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                       .dtype(brcs_tensor.scalar_type())
                                       .device(brcs_tensor.device());

    at::Tensor bloss_tensor = at::zeros({batch_size}, float_options);
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);

    if (brcs_tensor.dtype() == torch::kFloat32) {
        float *bloss = bloss_tensor.data_ptr<float>();
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();
        float *betot_dft = betot_dft_tensor.data_ptr<float>();
        float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float *loss_ptr = &(bloss_tensor.data_ptr<float>()[bb]);
                float *etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);
                float etot_dft = betot_dft_tensor[bb].item<float>();
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<float> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_ef(
                        (*etot_ptr),
                        (float*)force,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }
            find_ef_cpu_launcher<float>(
                betot,
                bforce,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_ef_torch_launcher(
                    betot,
                    (float*)bforce,
                    (float)zbl_rmax,
                    (float)zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_ef_torch_launcher(
                betot,
                bforce,
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
                (float)rmax,
                (float)rmin,
                q_scaler);
            
            find_ef_loss_torch_launcher(
                bloss,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                (float)e_weight,
                (float)f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft);
            #endif
        }
    } else {
        double *bloss = bloss_tensor.data_ptr<double>();
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();
        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double* zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double* zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *loss_ptr = &(bloss_tensor.data_ptr<double>()[bb]);
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double etot_dft = betot_dft_tensor[bb].item<double>();
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<double> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_ef(
                        (*etot_ptr),
                        (double*)force,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }
            find_ef_cpu_launcher<double>(
                betot,
                bforce,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_ef_torch_launcher(
                    betot,
                    (double*)bforce,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_ef_torch_launcher(
                betot,
                bforce,
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

            find_ef_loss_torch_launcher(
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
            #endif
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
                            q_scaler_tensor,
                            at::tensor(zbl_rmax, float_options),
                            at::tensor(zbl_rmin, float_options),
                            zbl_cks_tensor,
                            zbl_dks_tensor
                            });

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
    at::Tensor q_scaler_tensor = ctx->get_saved_variables()[23];
    double zbl_rmax = ctx->get_saved_variables()[24].item<double>();
    double zbl_rmin = ctx->get_saved_variables()[25].item<double>();
    at::Tensor zbl_cks_tensor = ctx->get_saved_variables()[26];
    at::Tensor zbl_dks_tensor = ctx->get_saved_variables()[27];
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    int num_linear_coeffs = (int)linear_coeffs_tensor.size(0);

    int batch_size = (int)bgrad_output_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = (int)bfirstneigh_tensor.size(2);
    int natoms_pad = (int)bilist_tensor.size(1);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
    int *type_map = type_map_tensor.data_ptr<int>();

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                       .dtype(brcs_tensor.scalar_type())
                                       .device(brcs_tensor.device());

    at::Tensor bloss_der2coeffs_tensor = at::zeros({batch_size, num_coeffs}, float_options);
    at::Tensor bloss_der2linear_coeffs_tensor = at::zeros({batch_size, num_linear_coeffs}, float_options);
    at::Tensor bloss_der2type_bias_tensor = at::zeros({batch_size, ntypes}, float_options);
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);
    
    if (bgrad_output_tensor.dtype() == torch::kFloat32) {
        float *bloss_der2coeffs = bloss_der2coeffs_tensor.data_ptr<float>();
        float *bloss_der2linear_coeffs = bloss_der2linear_coeffs_tensor.data_ptr<float>();
        float *bloss_der2type_bias = bloss_der2type_bias_tensor.data_ptr<float>();
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();
        float *betot_dft = betot_dft_tensor.data_ptr<float>();
        float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float* zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float* zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<float>();
                float *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<float>();
                float *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<float>();
                float *etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);
                float etot_dft = betot_dft_tensor[bb].item<float>();
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                float (*force_dft)[3] = (float (*)[3])bforce_dft_tensor[bb].data_ptr<float>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<float> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_ef(
                        (*etot_ptr),
                        (float*)force,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }

            find_ef_cpu_launcher<float>(
                betot,
                bforce,
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
                bloss_der2linear_coeffs,
                bloss_der2type_bias,
                e_weight,
                f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_ef_torch_launcher(
                    betot,
                    (float*)bforce,
                    (float)zbl_rmax,
                    (float)zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_ef_torch_launcher(
                betot,
                bforce,
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
                (float)rmax,
                (float)rmin,
                q_scaler);
            
            find_ef_loss_backward_torch_launcher(
                bloss_der2coeffs,
                bloss_der2linear_coeffs,
                bloss_der2type_bias,
                (float)e_weight,
                (float)f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
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
                (float)rmax,
                (float)rmin,
                q_scaler);
            #endif
        }
    } else {
        double *bloss_der2coeffs = bloss_der2coeffs_tensor.data_ptr<double>();
        double *bloss_der2linear_coeffs = bloss_der2linear_coeffs_tensor.data_ptr<double>();
        double *bloss_der2type_bias = bloss_der2type_bias_tensor.data_ptr<double>();
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();
        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double* zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double* zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *loss_der2coeffs = bloss_der2coeffs_tensor[bb].data_ptr<double>();
                double *loss_der2linear_coeffs = bloss_der2linear_coeffs_tensor[bb].data_ptr<double>();
                double *loss_der2type_bias = bloss_der2type_bias_tensor[bb].data_ptr<double>();
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double etot_dft = betot_dft_tensor[bb].item<double>();
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                double (*force_dft)[3] = (double (*)[3])bforce_dft_tensor[bb].data_ptr<double>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<double> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_ef(
                        (*etot_ptr),
                        (double*)force,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }
            find_ef_cpu_launcher<double>(
                betot,
                bforce,
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
                bloss_der2linear_coeffs,
                bloss_der2type_bias,
                e_weight,
                f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_ef_torch_launcher(
                    betot,
                    (double*)bforce,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_ef_torch_launcher(
                betot,
                bforce,
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
            
            find_ef_loss_backward_torch_launcher(
                bloss_der2coeffs,
                bloss_der2linear_coeffs,
                bloss_der2type_bias,
                e_weight,
                f_weight,
                betot,
                betot_dft,
                bforce,
                bforce_dft,
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
            #endif
        }
    }

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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    // 
    int batch_size = binum_tensor.size(0);
    int natoms_pad = bnumneigh_tensor.size(1);  // not inum
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
    int *type_map = type_map_tensor.data_ptr<int>();

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    //
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                       .dtype(brcs_tensor.scalar_type())
                                       .device(brcs_tensor.device());

    // 
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad+nghost, 3}, float_options);
    at::Tensor bvirial_tensor = at::zeros({batch_size, 9}, float_options);

    // 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();
        float *bvirial = bvirial_tensor.data_ptr<float>();

        float* coeffs = coeffs_tensor.data_ptr<float>();
        float* linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float* type_bias = type_bias_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float* q_scaler = q_scaler_tensor.data_ptr<float>();
        float* zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float* zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float *etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                float *virial = bvirial_tensor[bb].data_ptr<float>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = btypes_tensor[bb].data_ptr<int>();
                
                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<float> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_efv(
                        (*etot_ptr),
                        (float*)force,
                        virial,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }

            find_efv_cpu_launcher<float>(
                betot,
                bforce,
                bvirial,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_efv_torch_launcher(
                    betot,
                    (float*)bforce,
                    bvirial,
                    (float)zbl_rmax,
                    (float)zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_efv_torch_launcher(
                betot,
                bforce,
                bvirial,
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
                (float)rmax,
                (float)rmin,
                q_scaler);
            #endif
        }
    } else {
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();
        double *bvirial = bvirial_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                double *virial = bvirial_tensor[bb].data_ptr<double>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = btypes_tensor[bb].data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<double> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_efv(
                        (*etot_ptr),
                        (double*)force,
                        virial,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }

            find_efv_cpu_launcher<double>(
                betot,
                bforce,
                bvirial,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_efv_torch_launcher(
                    betot,
                    (double*)bforce,
                    bvirial,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_efv_torch_launcher(
                betot,
                bforce,
                bvirial,
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
            #endif
        }
    }

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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    // 
    int batch_size = binum_tensor.size(0);
    int natoms_pad = bnumneigh_tensor.size(1);  // not inum
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
    int *type_map = type_map_tensor.data_ptr<int>();

    int *binum = binum_tensor.data_ptr<int>();
    int *bilist = bilist_tensor.data_ptr<int>();
    int *bnumneigh = bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = bfirstneigh_tensor.data_ptr<int>();
    int *btypes = btypes_tensor.data_ptr<int>();

    //
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                       .dtype(brcs_tensor.scalar_type())
                                       .device(brcs_tensor.device());

    // 
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad+nghost, 3}, float_options);

    // 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();

        float* coeffs = coeffs_tensor.data_ptr<float>();
        float* linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float* type_bias = type_bias_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float* q_scaler = q_scaler_tensor.data_ptr<float>();
        float* zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float* zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float* etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = btypes_tensor[bb].data_ptr<int>();
                
                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<float> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_ef(
                        (*etot_ptr),
                        (float*)force,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }

            find_ef_cpu_launcher<float>(
                betot,
                bforce,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_ef_torch_launcher(
                    betot,
                    (float*)bforce,
                    (float)zbl_rmax,
                    (float)zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_ef_torch_launcher(
                betot,
                bforce,
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
                (float)rmax,
                (float)rmin,
                q_scaler);
            #endif
        }
    } else {
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                int inum = binum_tensor[bb].item<int>();
                int *ilist = bilist_tensor[bb].data_ptr<int>();
                int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = btypes_tensor[bb].data_ptr<int>();
                int *type_map = type_map_tensor.data_ptr<int>();

                if (zbl_rmax > 0.0) {
                    ai2pot::correction::GroupZBL<double> gzbl(
                        ntypes,
                        type_map,
                        type_map,
                        zbl_rmax,
                        zbl_rmin,
                        zbl_cks,
                        zbl_dks);

                    gzbl.correct_ef(
                        (*etot_ptr),
                        (double*)force,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        rcs,
                        types,
                        ntypes,
                        type_map,
                        umax_num_neigh_atoms,
                        nghost);
                }
            }
            
            find_ef_cpu_launcher<double>(
                betot,
                bforce,
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
            if (zbl_rmax > 0.0)
                ai2pot::correction::correct_zbl_ef_torch_launcher(
                    betot,
                    (double*)bforce,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks,
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
                    nghost);

            find_ef_torch_launcher(
                betot,
                bforce,
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
            #endif
        }
    }


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
            at::Tensor(),
            at::Tensor()};
}


torch::autograd::variable_list LinearMtpToEsitesFunction::forward(
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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    // 1. 
    int batch_size = binum_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int ntypes = type_map_tensor.size(0);
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    // 2.
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    // 3.
    at::Tensor be_sites_tensor;

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(brcs_tensor.device());
        be_sites_tensor = at::zeros({batch_size, natoms_pad}, float_options);
        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();
        int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        for (int bb=0; bb<batch_size; bb++) {
            float *e_sites = (float*)be_sites_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();
            int *type_map = (int*)type_map_tensor.data_ptr<int>();

            if (zbl_rmax > 0.0) {
                ai2pot::correction::GroupZBL<float> gzbl(
                    ntypes,
                    type_map,
                    type_map,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);

                gzbl.correct_e_sites(
                    e_sites,
                    inum,
                    ilist,
                    numneigh,
                    firstneigh,
                    rcs,
                    types,
                    ntypes,
                    type_map,
                    umax_num_neigh_atoms,
                    nghost);
            }

            LinearMtp<float>::find_e_sites(
                e_sites,
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
                q_scaler);
        }
    } else {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat64)
                        .device(brcs_tensor.device());
        be_sites_tensor = at::zeros({batch_size, natoms_pad}, float_options);
        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        for (int bb=0; bb<batch_size; bb++) {
            double *e_sites = (double*)be_sites_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
            int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = (int*)btypes_tensor[bb].data_ptr<int>();
            int *type_map = (int*)type_map_tensor.data_ptr<int>();

            if (zbl_rmax > 0.0) {
                ai2pot::correction::GroupZBL<double> gzbl(
                    ntypes,
                    type_map,
                    type_map,
                    zbl_rmax,
                    zbl_rmin,
                    zbl_cks,
                    zbl_dks);

                gzbl.correct_e_sites(
                    e_sites,
                    inum,
                    ilist,
                    numneigh,
                    firstneigh,
                    rcs,
                    types,
                    ntypes,
                    type_map,
                    umax_num_neigh_atoms,
                    nghost);
            }

            LinearMtp<double>::find_e_sites(
                e_sites,
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
                q_scaler);
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
        torch::tensor(rmin, float_options),
        q_scaler_tensor,
        torch::tensor(zbl_rmax, float_options),
        torch::tensor(zbl_rmin, float_options),
        zbl_cks_tensor,
        zbl_dks_tensor});

    return {be_sites_tensor};
}


torch::autograd::variable_list LinearMtpToEsitesFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    // 1. 
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();

    int chebyshev_size = ctx->get_saved_variables()[0].item<int>();
    at::Tensor coeffs_tensor = ctx->get_saved_variables()[1];
    at::Tensor linear_coeffs_tensor = ctx->get_saved_variables()[2];
    at::Tensor type_bias_tensor = ctx->get_saved_variables()[3];
    int alpha_moments_count = ctx->get_saved_variables()[4].item<int>();
    at::Tensor alpha_index_basic_tensor = ctx->get_saved_variables()[5];
    at::Tensor alpha_index_times_tensor = ctx->get_saved_variables()[6];
    at::Tensor alpha_moment_mapping_tensor = ctx->get_saved_variables()[7];
    int nmus = ctx->get_saved_variables()[8].item<int>();
    at::Tensor binum_tensor = ctx->get_saved_variables()[9];
    at::Tensor bilist_tensor = ctx->get_saved_variables()[10];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[11];
    at::Tensor bfirstneigh_tensor = ctx->get_saved_variables()[12];
    at::Tensor brcs_tensor = ctx->get_saved_variables()[13];
    at::Tensor btypes_tensor = ctx->get_saved_variables()[14];
    at::Tensor type_map_tensor = ctx->get_saved_variables()[15];
    int nghost = ctx->get_saved_variables()[16].item<int>();
    double rmax = ctx->get_saved_variables()[17].item<double>();
    double rmin = ctx->get_saved_variables()[18].item<double>();
    at::Tensor q_scaler_tensor = ctx->get_saved_variables()[19];
    double zbl_rmax = ctx->get_saved_variables()[20].item<double>();
    double zbl_rmin = ctx->get_saved_variables()[21].item<double>();
    at::Tensor zbl_cks_tensor = ctx->get_saved_variables()[22];
    at::Tensor zbl_dks_tensor = ctx->get_saved_variables()[23];
    
    // 2. 
    int batch_size = binum_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int ntypes = type_map_tensor.size(0);
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    
    // 3. 
    c10::TensorOptions int_options = c10::TensorOptions()
                                     .dtype(torch::kInt32)
                                     .device(brcs_tensor.device());
    c10::TensorOptions float_options;

    // 4. 
    at::Tensor be_sites_der2coeffs_tensor;
    at::Tensor be_sites_der2linear_coeffs_tensor;
    at::Tensor be_sites_der2type_bias_tensor;

    // 5.
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
    int *type_map = (int*)type_map_tensor.data_ptr<int>();

    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(brcs_tensor.device());
        be_sites_der2coeffs_tensor = at::zeros({batch_size, natoms_pad, num_coeffs}, float_options);
        be_sites_der2linear_coeffs_tensor = at::zeros({batch_size, natoms_pad, alpha_scalar_moments}, float_options);
        be_sites_der2type_bias_tensor = at::zeros({batch_size, natoms_pad, ntypes}, float_options);
        float *coeffs = (float*)coeffs_tensor.data_ptr<float>();
        float *linear_coeffs = (float*)linear_coeffs_tensor.data_ptr<float>();
        float *type_bias = (float*)type_bias_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();

        for (int bb=0; bb<batch_size; bb++) {
            float *e_sites_der2coeffs = (float*)be_sites_der2coeffs_tensor[bb].data_ptr<float>();
            float *e_sites_der2linear_coeffs = (float*)be_sites_der2linear_coeffs_tensor[bb].data_ptr<float>();
            float *e_sites_der2type_bias = (float*)be_sites_der2type_bias_tensor[bb].data_ptr<float>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
            int *types = btypes_tensor[bb].data_ptr<int>();

            LinearMtp<float>::find_e_sites_backward(
                e_sites_der2coeffs,
                e_sites_der2linear_coeffs,
                e_sites_der2type_bias,
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
                q_scaler);
        }
    } else {
        float_options = c10::TensorOptions()
                        .dtype(torch::kFloat64)
                        .device(brcs_tensor.device());
        be_sites_der2coeffs_tensor = at::zeros({batch_size, natoms_pad, num_coeffs}, float_options);
        be_sites_der2linear_coeffs_tensor = at::zeros({batch_size, natoms_pad, alpha_scalar_moments}, float_options);
        be_sites_der2type_bias_tensor = at::zeros({batch_size, natoms_pad, ntypes}, float_options);
        double *coeffs = (double*)coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = (double*)linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = (double*)type_bias_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        
        for (int bb=0; bb<batch_size; bb++) {
            double *e_sites_der2coeffs = (double*)be_sites_der2coeffs_tensor[bb].data_ptr<double>();
            double *e_sites_der2linear_coeffs = (double*)be_sites_der2linear_coeffs_tensor[bb].data_ptr<double>();
            double *e_sites_der2type_bias = (double*)be_sites_der2type_bias_tensor[bb].data_ptr<double>();
            int inum = binum_tensor[bb].item<int>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
            int *types = btypes_tensor[bb].data_ptr<int>();

            LinearMtp<double>::find_e_sites_backward(
                e_sites_der2coeffs,
                e_sites_der2linear_coeffs,
                e_sites_der2type_bias,
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
                q_scaler);
        }
    }
    return {
        at::Tensor(),
        (bgrad_output_tensor.unsqueeze(-1) * be_sites_der2coeffs_tensor).sum(torch::IntArrayRef({0, 1})),
        (bgrad_output_tensor.unsqueeze(-1) * be_sites_der2linear_coeffs_tensor).sum(torch::IntArrayRef({0, 1})),
        (bgrad_output_tensor.unsqueeze(-1) * be_sites_der2type_bias_tensor).sum(torch::IntArrayRef({0, 1})),
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
    int batch_size = binum_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    int ntypes = type_map_tensor.size(0);
    const int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    const int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    const int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();
    int *type_map = type_map_tensor.data_ptr<int>();

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
    at::Tensor bdescriptors_tensor = at::zeros({batch_size, natoms_pad, alpha_scalar_moments}, float_options);

    // 4.
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *bdescriptors = bdescriptors_tensor.data_ptr<float>();
        float *coeffs = (float*)coeffs_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_descriptors_cpu_launcher<float>(
                bdescriptors,
                chebyshev_size,
                coeffs,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moments,
                alpha_moment_mapping,
                nmus,
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
            find_descriptors_torch_launcher<float>(
                bdescriptors,
                chebyshev_size,
                coeffs,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moments,
                alpha_moment_mapping,
                nmus,
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
            #endif
        }
    } else {
        double *bdescriptors = bdescriptors_tensor.data_ptr<double>();
        double *coeffs = (double*)coeffs_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            find_descriptors_cpu_launcher<double>(
                bdescriptors,
                chebyshev_size,
                coeffs,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moments,
                alpha_moment_mapping,
                nmus,
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
            find_descriptors_torch_launcher<double>(
                bdescriptors,
                chebyshev_size,
                coeffs,
                alpha_moments_count,
                alpha_index_basic_count,
                alpha_index_basic,
                alpha_index_times_count,
                alpha_index_times,
                alpha_scalar_moments,
                alpha_moment_mapping,
                nmus,
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
            #endif
        }
    }

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
    const at::Tensor& q_scaler_tensor,
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
        q_scaler_tensor,
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
    const at::Tensor& q_scaler_tensor,
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
        q_scaler_tensor,
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
    const at::Tensor& q_scaler_tensor,
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
        q_scaler_tensor,
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
    const at::Tensor& q_scaler_tensor,
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
        q_scaler_tensor,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
}


torch::autograd::variable_list LinearMtpToEsitesOp(
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
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    return LinearMtpToEsitesFunction::apply(
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
        q_scaler_tensor,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
}

 
torch::autograd::variable_list LinearMtpToDescriptorsOp(
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
    double rmin)
{
    return LinearMtpToDescriptorsFunction::apply(
        chebyshev_size,
        coeffs_tensor,
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
