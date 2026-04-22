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
#include "../../correction/include/zbl.h"


#if defined(USE_CUDA) or defined(__INTELLISENSE__)
#include "../gpu/include/nep_torch_launcher.h"
#include "../gpu/include/nepLoss_torch_launcher.h"
#include "../../correction/gpu/include/zbl_torch_launcher.h"


namespace ai2pot {
namespace nep {

// 1. nep_torch_launcher.cu
// 1.1. find_ef_torch_launcher()
extern template void find_ef_torch_launcher<float>(
    float *d_betot_ptr,
    float (*d_bforce)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    float *d_coeffs,
    float *d_w0,
    float *d_b0,
    float *d_w1,
    float *d_type_bias,
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
    float rmax_radial,
    float rmax_angular,
    float *d_q_scaler);

extern template void find_ef_torch_launcher<double>(
    double *d_betot_ptr,
    double (*d_bforce)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    double *d_coeffs,
    double *d_w0,
    double *d_b0,
    double *d_w1,
    double *d_type_bias,
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
    double rmax_radial,
    double rmax_angular,
    double *d_q_scaler);


// 1.2. find_efv_torch_launcher()
extern template void ai2pot::nep::find_efv_torch_launcher<float>(
    float *d_betot_ptr,
    float (*d_bforce)[3],
    float *d_bvirial,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    float *d_coeffs,
    float *d_w0,
    float *d_b0,
    float *d_w1,
    float *d_type_bias,
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
    float rmax_raidal,
    float rmax_angular,
    float *d_q_scaler);

extern template void ai2pot::nep::find_efv_torch_launcher<double>(
    double *d_betot_ptr,
    double (*d_bforce)[3],
    double *d_bvirial,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    double *d_coeffs,
    double *d_w0,
    double *d_b0,
    double *d_w1,
    double *d_type_bias,
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
    double rmax_radial,
    double rmax_angular,
    double *d_q_scaler);


// 1.3. find_descriptors_torch_launcher()
extern template void find_descriptor_torch_launcher<float>(
    float *d_bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    float *d_coeffs,
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
    float rmax_radial,
    float rmax_angular);

extern template void find_descriptor_torch_launcher<double>(
    double *d_bdescriptors,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    double *d_coeffs,
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
    double rmax_radial,
    double rmax_angular);



// 2. nepLoss_torch_launcher.cu
// 2.1. find_ef_loss_torch_launcher()
extern template void find_ef_loss_torch_launcher<float>(
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

extern template void find_ef_loss_torch_launcher<double>(
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


// 2.2. find_loss_torch_launcher()
extern template void find_loss_torch_launcher<float>(
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

extern template void find_loss_torch_launcher<double>(
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


// 2.3. find_ef_loss_backward_torch_launcher()
extern template void find_ef_loss_backward_torch_launcher<float>(
    float *d_bloss_der2coeffs,
    float *d_bloss_der2w0,
    float *d_bloss_der2b0,
    float *d_bloss_der2w1,
    float *d_bloss_der2type_bias,
    float e_weight,
    float f_weight,
    float *d_betot_ml,
    float *d_betot_dft,
    float (*d_bforce_ml)[3],
    float (*d_bforce_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    float *d_coeffs,
    float *d_w0,
    float *d_b0,
    float *d_w1,
    float *d_type_bias,
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
    float rmax_radial,
    float rmax_angular,
    float *d_q_scaler);

extern template void find_ef_loss_backward_torch_launcher<double>(
    double *d_bloss_der2coeffs,
    double *d_bloss_der2w0,
    double *d_bloss_der2b0,
    double *d_bloss_der2w1,
    double *d_bloss_der2type_bias,
    double e_weight,
    double f_weight,
    double *d_betot_ml,
    double *d_betot_dft,
    double (*d_bforce_ml)[3],
    double (*d_bforce_dft)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    int num_neurons,
    double *d_coeffs,
    double *d_w0,
    double *d_b0,
    double *d_w1,
    double *d_type_bias,
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
    double rmax_radial,
    double rmax_angular,
    double *d_q_scaler);


// 2.4. find_loss_backward_torch_launcher()


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


};  // namespace : nep
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
namespace nep {


torch::autograd::variable_list NepToEFFunction::forward(
    torch::autograd::AutogradContext *ctx,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& b0_tensor,
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
    double rmax_radial,
    double rmax_angular,
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
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
        float *b0 = b0_tensor.data_ptr<float>();
        float *w1 = w1_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();

        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float *etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                int inum = binum[bb];
                int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
                int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = (int*)btypes_tensor[bb].data_ptr<int>();
                
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            if (zbl_rmax > 0)
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                (float)rmax_radial,
                (float)rmax_angular,
                q_scaler);
            #endif
        }
    } else {
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *w0 = w0_tensor.data_ptr<double>();
        double *b0 = b0_tensor.data_ptr<double>();
        double *w1 = w1_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();

        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                int inum = binum[bb];
                int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
                int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = (int*)btypes_tensor[bb].data_ptr<int>();
                
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            if (zbl_rmax > 0)
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);
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
        at::Tensor(),
        at::Tensor()};
}


/*
torch::autograd::variable_list NepToEFVFunction::foward(
    torch::autograd::AutogradContext *ctx,
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    const at::Tensor& coeffs_tensor,
    const at::Tensor& w0_tensor,
    const at::Tensor& b0_tensor,
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
    double rmax_radial,
    double rmax_angular,
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
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
        float *bvirial = bvirial_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *w0 = w0_tensor.data_ptr<float>();
        float *b0 = b0_tensor.data_ptr<float>();
        float *w1 = w1_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();

        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            // TODO
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // TODO
            #endif
        }
    } else {
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();
        double *bvirial = bvirial_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *w0 = w0_tensor.data_ptr<double>();
        double *b0 = b0_tensor.data_ptr<double>();
        double *w1 = w1_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();

        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            // TODO
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            // TODO
            #endif
        }
    }
}
*/


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
    const at::Tensor& b0_tensor,
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
    double rmax_radial,
    double rmax_angular,
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
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
    at::Tensor num_real_atoms_in_batch_tensor = at::zeros({1}, int_options);
    at::Tensor e_rmse_tensor = at::zeros({1}, float_options);
    at::Tensor f_rmse_tensor = at::zeros({1}, float_options);
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *bloss = bloss_tensor.data_ptr<float>();
        float *e_rmse_ptr = e_rmse_tensor.data_ptr<float>();
        float *f_rmse_ptr = f_rmse_tensor.data_ptr<float>();
        int *num_real_atoms_in_batch_ptr = num_real_atoms_in_batch_tensor.data_ptr<int>();
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *w0 = w0_tensor.data_ptr<float>();
        float *b0 = b0_tensor.data_ptr<float>();
        float *w1 = w1_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();

        float *betot_dft = betot_dft_tensor.data_ptr<float>();
        float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();

        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float *etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                int inum = binum[bb];
                int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
                int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = (int*)btypes_tensor[bb].data_ptr<int>();
                
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
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
            
            find_ef_rmse_cpu_launcher<float>(
                *e_rmse_ptr,
                *f_rmse_ptr,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                betot,
                betot_dft,
                bforce,
                bforce_dft);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            if (zbl_rmax > 0)
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                (float)rmax_radial,
                (float)rmax_angular,
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
            
            find_num_real_atoms_in_batch_torch_launcher(
                num_real_atoms_in_batch_ptr,
                batch_size,
                binum);
            find_e_se_torch_launcher(
                e_rmse_ptr,
                batch_size,
                binum,
                betot,
                betot_dft);
            find_f_se_torch_launcher(
                f_rmse_ptr,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bforce,
                bforce_dft);
            e_rmse_tensor = torch::sqrt(e_rmse_tensor / batch_size);
            f_rmse_tensor = torch::sqrt(f_rmse_tensor / (3*num_real_atoms_in_batch_tensor));
            #endif
        }
    } else {
        double *bloss = bloss_tensor.data_ptr<double>();
        double *e_rmse_ptr = e_rmse_tensor.data_ptr<double>();
        double *f_rmse_ptr = f_rmse_tensor.data_ptr<double>();
        int *num_real_atoms_in_batch_ptr = num_real_atoms_in_batch_tensor.data_ptr<int>();
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *w0 = w0_tensor.data_ptr<double>();
        double *b0 = b0_tensor.data_ptr<double>();
        double *w1 = w1_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
    
        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();

        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                int inum = binum[bb];
                int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
                int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = (int*)btypes_tensor[bb].data_ptr<int>();
                
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
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
            
            find_ef_rmse_cpu_launcher<double>(
                *e_rmse_ptr,
                *f_rmse_ptr,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                betot,
                betot_dft,
                bforce,
                bforce_dft);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            if (zbl_rmax > 0)
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
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
            
            find_num_real_atoms_in_batch_torch_launcher(
                num_real_atoms_in_batch_ptr,
                batch_size,
                binum);
            find_e_se_torch_launcher(
                e_rmse_ptr,
                batch_size,
                binum,
                betot,
                betot_dft);
            find_f_se_torch_launcher(
                f_rmse_ptr,
                batch_size,
                natoms_pad,
                binum,
                bilist,
                bforce,
                bforce_dft);
            e_rmse_tensor = torch::sqrt(e_rmse_tensor / batch_size);
            f_rmse_tensor = torch::sqrt(f_rmse_tensor / (3*num_real_atoms_in_batch_tensor));
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
        b0_tensor,
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
        torch::tensor(rmax_radial, float_options),
        torch::tensor(rmax_angular, float_options),
        q_scaler_tensor,
        torch::tensor(zbl_rmax, float_options),
        torch::tensor(zbl_rmin, float_options),
        zbl_cks_tensor,
        zbl_dks_tensor});
    
    return {bloss_tensor,
            e_rmse_tensor,
            f_rmse_tensor};
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
    at::Tensor b0_tensor = ctx->get_saved_variables()[10];
    at::Tensor w1_tensor = ctx->get_saved_variables()[11];
    at::Tensor type_bias_tensor = ctx->get_saved_variables()[12];
    at::Tensor binum_tensor = ctx->get_saved_variables()[13];
    at::Tensor bilist_tensor = ctx->get_saved_variables()[14];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[15];
    at::Tensor bfirstneigh_tensor = ctx->get_saved_variables()[16];
    at::Tensor brcs_tensor = ctx->get_saved_variables()[17];
    at::Tensor btypes_tensor = ctx->get_saved_variables()[18];
    at::Tensor type_map_tensor = ctx->get_saved_variables()[19];
    int nghost = ctx->get_saved_variables()[20].item<int>();
    double rmax_radial = ctx->get_saved_variables()[21].item<double>();
    double rmax_angular = ctx->get_saved_variables()[22].item<double>();
    at::Tensor q_scaler_tensor = ctx->get_saved_variables()[23];
    double zbl_rmax = ctx->get_saved_variables()[24].item<double>();
    double zbl_rmin = ctx->get_saved_variables()[25].item<double>();
    at::Tensor zbl_cks_tensor = ctx->get_saved_variables()[26];
    at::Tensor zbl_dks_tensor = ctx->get_saved_variables()[27];

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
    at::Tensor bloss_der2b0_tensor = at::zeros({batch_size, ntypes*num_neurons}, float_options);
    at::Tensor bloss_der2w1_tensor = at::zeros({batch_size, ntypes*num_neurons}, float_options);
    at::Tensor bloss_der2type_bias_tensor = at::zeros({batch_size, ntypes}, float_options);
    at::Tensor betot_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);

    // 4. 
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *bloss_der2coeffs = bloss_der2coeffs_tensor.data_ptr<float>();
        float *bloss_der2w0 = bloss_der2w0_tensor.data_ptr<float>();
        float *bloss_der2b0 = bloss_der2b0_tensor.data_ptr<float>();
        float *bloss_der2w1 = bloss_der2w1_tensor.data_ptr<float>();
        float *bloss_der2type_bias = bloss_der2type_bias_tensor.data_ptr<float>();
        float *betot = betot_tensor.data_ptr<float>();
        float (*bforce)[3] = (float (*)[3])bforce_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *w0 = w0_tensor.data_ptr<float>();
        float *b0 = b0_tensor.data_ptr<float>();
        float *w1 = w1_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();

        float *betot_dft = betot_dft_tensor.data_ptr<float>();
        float (*bforce_dft)[3] = (float (*)[3])bforce_dft_tensor.data_ptr<float>();

        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                float *etot_ptr = &(betot_tensor.data_ptr<float>()[bb]);
                float (*force)[3] = (float (*)[3])bforce_tensor[bb].data_ptr<float>();
                int inum = binum[bb];
                int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
                int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
                float (*rcs)[3] = (float (*)[3])brcs_tensor[bb].data_ptr<float>();
                int *types = (int*)btypes_tensor[bb].data_ptr<int>();
                
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);

            find_ef_loss_backward_cpu_launcher<float>(
                bloss_der2coeffs,
                bloss_der2w0,
                bloss_der2b0,
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
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            if (zbl_rmax > 0)
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                (float)rmax_radial,
                (float)rmax_angular,
                q_scaler);
            
            find_ef_loss_backward_torch_launcher(
                bloss_der2coeffs,
                bloss_der2w0,
                bloss_der2b0,
                bloss_der2w1,
                bloss_der2type_bias,
                (float)e_weight,
                (float)f_weight,
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
                b0,
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
                (float)rmax_radial,
                (float)rmax_angular,
                q_scaler);
            #endif
        }
    } else {
        double *bloss_der2coeffs = bloss_der2coeffs_tensor.data_ptr<double>();
        double *bloss_der2w0 = bloss_der2w0_tensor.data_ptr<double>();
        double *bloss_der2b0 = bloss_der2b0_tensor.data_ptr<double>();
        double *bloss_der2w1 = bloss_der2w1_tensor.data_ptr<double>();
        double *bloss_der2type_bias = bloss_der2type_bias_tensor.data_ptr<double>();
        double *betot = betot_tensor.data_ptr<double>();
        double (*bforce)[3] = (double (*)[3])bforce_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *w0 = w0_tensor.data_ptr<double>();
        double *b0 = b0_tensor.data_ptr<double>();
        double *w1 = w1_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();

        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();

        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        if (brcs_tensor.device() == c10::kCPU) {
            for (int bb=0; bb<batch_size; bb++) {
                double *etot_ptr = &(betot_tensor.data_ptr<double>()[bb]);
                double (*force)[3] = (double (*)[3])bforce_tensor[bb].data_ptr<double>();
                int inum = binum[bb];
                int *ilist = (int*)bilist_tensor[bb].data_ptr<int>();
                int *numneigh = (int*)bnumneigh_tensor[bb].data_ptr<int>();
                int *firstneigh = (int*)bfirstneigh_tensor[bb].data_ptr<int>();
                double (*rcs)[3] = (double (*)[3])brcs_tensor[bb].data_ptr<double>();
                int *types = (int*)btypes_tensor[bb].data_ptr<int>();
                
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);

            find_ef_loss_backward_cpu_launcher<double>(
                bloss_der2coeffs,
                bloss_der2w0,
                bloss_der2b0,
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
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            if (zbl_rmax > 0)
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
                n_radial_basis,
                n_angular_basis,
                l_max,
                num_neurons,
                coeffs,
                w0,
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);
            
            find_ef_loss_backward_torch_launcher(
                bloss_der2coeffs,
                bloss_der2w0,
                bloss_der2b0,
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
                b0,
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
                rmax_radial,
                rmax_angular,
                q_scaler);
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
        torch::matmul(bgrad_output_tensor, bloss_der2b0_tensor),
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
    double rmax_radial,
    double rmax_angular)
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
                rmax_radial,
                rmax_angular);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            find_descriptor_torch_launcher(
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
                (float)rmax_radial,
                (float)rmax_angular);
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
                rmax_radial,
                rmax_angular);
        } else {
            #if defined(USE_CUDA) or defined(__INTELLISENSE__)
            find_descriptor_torch_launcher(
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
                rmax_radial,
                rmax_angular);
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
    const at::Tensor& b0_tensor,
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
    double rmax_radial,
    double rmax_angular,
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
{
    return NepToEFFunction::apply(
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        coeffs_tensor,
        w0_tensor,
        b0_tensor,
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
        rmax_radial,
        rmax_angular,
        q_scaler_tensor,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
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
    const at::Tensor& b0_tensor,
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
    double rmax_radial,
    double rmax_angular,
    const at::Tensor& q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor& zbl_cks_tensor,
    const at::Tensor& zbl_dks_tensor)
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
        b0_tensor,
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
        rmax_radial,
        rmax_angular,
        q_scaler_tensor,
        zbl_rmax,
        zbl_rmin,
        zbl_cks_tensor,
        zbl_dks_tensor);
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
    double rmax_radial,
    double rmax_angular)
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
        rmax_radial,
        rmax_angular);
}


};  // namespace : nep
};  // namespace : ai2pot
