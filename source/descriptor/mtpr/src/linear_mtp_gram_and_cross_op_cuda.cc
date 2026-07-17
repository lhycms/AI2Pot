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

#include "../include/linear_mtp_gram_and_cross_op_cuda.h"
#include "../gpu/include/linear_mtp_gram_and_cross_torch_launcher.h"
#include "../../correction/gpu/include/zbl_torch_launcher.h"


namespace ai2pot {
namespace mtpr {

// 1. find_efv_components()
extern template void find_efv_components_torch_launcher<float>(
    float *d_benergy_components,
    float *d_bforce_components,
    float *d_bvirial_components,
    int chebyshev_size,
    float scaling,
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
    int umax_num_neigh_atoms,
    int nghost,
    float rmax,
    float rmin,
    float *d_q_scaler);

extern template void find_efv_components_torch_launcher<double>(
    double *d_benergy_components,
    double *d_bforce_components,
    double *d_bvirial_components,
    int chebyshev_size,
    double scaling,
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
    int umax_num_neigh_atoms,
    int nghost,
    double rmax,
    double rmin,
    double *d_q_scaler);

// 2. find_lin_matrix_lin_vector()
extern template void find_lin_matrix_lin_vector_torch_launcher<float>(
    float *d_lin_matrix,
    float *d_lin_vector,
    float e_weight,
    float f_weight,
    float v_weight,
    float *d_betot_residual,
    float (*d_bforce_residual)[3],
    float *d_bvirial_residual,
    float *d_benergy_components,
    float *d_bforce_components,
    float *d_bvirial_components,
    int alpha_scalar_moments,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int ntypes);

extern template void find_lin_matrix_lin_vector_torch_launcher<double>(
    double *d_lin_matrix,
    double *d_lin_vector,
    double e_weight,
    double f_weight,
    double v_weight,
    double *d_betot_residual,
    double (*d_bforce_residual)[3],
    double *d_bvirial_residual,
    double *d_benergy_components,
    double *d_bforce_components,
    double *d_bvirial_components,
    int alpha_scalar_moments,
    int nmus,
    int batch_size,
    int natoms_pad,
    int *d_binum,
    int *d_bilist,
    int ntypes);


// 3. mirror_lin_matrix()
extern template void mirror_lin_matrix_torch_launcher<float>(
    float *d_lin_matrix,
    int num_parameters);

extern template void mirror_lin_matrix_torch_launcher<double>(
    double *d_lin_matrix,
    int num_parameters);

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
};  // namespace : mtpr
};  // namespace : correction




namespace ai2pot {
namespace mtpr {

torch::autograd::variable_list LinMatrixLinVectorFunctionCUDA::forward(
    torch::autograd::AutogradContext *ctx,
    double e_weight,
    double f_weight,
    double v_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    const at::Tensor& bvirial_dft_tensor,
    int chebyshev_size,
    double scaling,
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
    const at::Tensor &q_scaler_tensor,
    double zbl_rmax,
    double zbl_rmin,
    const at::Tensor &zbl_cks_tensor,
    const at::Tensor &zbl_dks_tensor)
{
    // 1.
    int batch_size = bfirstneigh_tensor.size(0);
    int natoms_pad = bfirstneigh_tensor.size(1);
    int umax_num_neigh_atoms = bfirstneigh_tensor.size(2);
    
    int alpha_index_basic_count = alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = alpha_moment_mapping_tensor.size(0);
    int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
    int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = (int*)alpha_moment_mapping_tensor.data_ptr<int>();

    int ntypes = type_map_tensor.size(0);
    int *type_map = (int*)type_map_tensor.data_ptr<int>();
    int *binum = (int*)binum_tensor.data_ptr<int>();
    int *bilist = (int*)bilist_tensor.data_ptr<int>();
    int *bnumneigh = (int*)bnumneigh_tensor.data_ptr<int>();
    int *bfirstneigh = (int*)bfirstneigh_tensor.data_ptr<int>();
    int *btypes = (int*)btypes_tensor.data_ptr<int>();

    int num_parameters = alpha_scalar_moments + ntypes;

    // 2.
    c10::TensorOptions int_options = c10::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(brcs_tensor.device());
    c10::TensorOptions float_options = c10::TensorOptions()
                                        .dtype(brcs_tensor.scalar_type())
                                        .device(brcs_tensor.device());

    // 3.
    at::Tensor lin_matrix_tensor = at::zeros({num_parameters, num_parameters}, float_options);
    at::Tensor lin_vector_tensor = at::zeros({num_parameters}, float_options);
    at::Tensor benergy_components_tensor = at::zeros({batch_size, num_parameters}, float_options);
    at::Tensor bforce_components_tensor = at::zeros({batch_size, natoms_pad, 3, num_parameters}, float_options);
    at::Tensor bvirial_components_tensor = at::zeros({batch_size, 9, num_parameters}, float_options);
    at::Tensor betot_zbl_tensor = at::zeros({batch_size}, float_options);
    at::Tensor bforce_zbl_tensor = at::zeros({batch_size, natoms_pad, 3}, float_options);
    at::Tensor bvirial_zbl_tensor = at::zeros({batch_size, 9}, float_options);

    // 4.
    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float *lin_matrix = lin_matrix_tensor.data_ptr<float>();
        float *lin_vector = lin_vector_tensor.data_ptr<float>();
        float *benergy_components = benergy_components_tensor.data_ptr<float>();
        float *bforce_components = bforce_components_tensor.data_ptr<float>();
        float *bvirial_components = bvirial_components_tensor.data_ptr<float>();
        
        float *betot_zbl = betot_zbl_tensor.data_ptr<float>();
        float (*bforce_zbl)[3] = (float (*)[3])bforce_zbl_tensor.data_ptr<float>();
        float *bvirial_zbl = bvirial_zbl_tensor.data_ptr<float>();
        float *q_scaler = q_scaler_tensor.data_ptr<float>();
        float *zbl_cks = zbl_cks_tensor.data_ptr<float>();
        float *zbl_dks = zbl_dks_tensor.data_ptr<float>();

        float *coeffs = coeffs_tensor.data_ptr<float>();
        float *linear_coeffs = linear_coeffs_tensor.data_ptr<float>();
        float *type_bias = type_bias_tensor.data_ptr<float>();
        float (*brcs)[3] = (float (*)[3])brcs_tensor.data_ptr<float>();

        if (zbl_rmax > 0.0)
            ai2pot::correction::correct_zbl_efv_torch_launcher(
                betot_zbl,
                (float*)bforce_zbl,
                bvirial_zbl,
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
        
        at::Tensor betot_residual_tensor = betot_dft_tensor - betot_zbl_tensor;
        at::Tensor bforce_residual_tensor = bforce_dft_tensor - bforce_zbl_tensor;
        at::Tensor bvirial_residual_tensor = bvirial_dft_tensor - bvirial_zbl_tensor;
        float *betot_residual = betot_residual_tensor.data_ptr<float>();
        float (*bforce_residual)[3] = (float (*)[3])bforce_residual_tensor.data_ptr<float>();
        float *bvirial_residual = bvirial_residual_tensor.data_ptr<float>();
        
        find_efv_components_torch_launcher<float>(
            benergy_components,
            bforce_components,
            bvirial_components,
            chebyshev_size,
            scaling,
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
            umax_num_neigh_atoms,
            nghost,
            (float)rmax,
            (float)rmin,
            q_scaler);
        
        
        find_lin_matrix_lin_vector_torch_launcher<float>(
            lin_matrix,
            lin_vector,
            e_weight,
            f_weight,
            v_weight,
            betot_residual,
            bforce_residual,
            bvirial_residual,
            benergy_components,
            bforce_components,
            bvirial_components,
            alpha_scalar_moments,
            nmus,
            batch_size,
            natoms_pad,
            binum,
            bilist,
            ntypes);

        mirror_lin_matrix_torch_launcher<float>(
            lin_matrix,
            num_parameters);
    } else {
        double *lin_matrix = lin_matrix_tensor.data_ptr<double>();
        double *lin_vector = lin_vector_tensor.data_ptr<double>();
        double *benergy_components = benergy_components_tensor.data_ptr<double>();
        double *bforce_components = bforce_components_tensor.data_ptr<double>();
        double *bvirial_components = bvirial_components_tensor.data_ptr<double>();
        
        double *betot_dft = betot_dft_tensor.data_ptr<double>();
        double (*bforce_dft)[3] = (double (*)[3])bforce_dft_tensor.data_ptr<double>();
        double *bvirial_dft = bvirial_dft_tensor.data_ptr<double>();
        double *betot_zbl = betot_zbl_tensor.data_ptr<double>();
        double (*bforce_zbl)[3] = (double (*)[3])bforce_zbl_tensor.data_ptr<double>();
        double *bvirial_zbl = bvirial_zbl_tensor.data_ptr<double>();
        double *q_scaler = q_scaler_tensor.data_ptr<double>();
        double *zbl_cks = zbl_cks_tensor.data_ptr<double>();
        double *zbl_dks = zbl_dks_tensor.data_ptr<double>();

        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();
        double (*brcs)[3] = (double (*)[3])brcs_tensor.data_ptr<double>();

        if (zbl_rmax > 0.0)
            ai2pot::correction::correct_zbl_efv_torch_launcher(
                betot_zbl,
                (double*)bforce_zbl,
                bvirial_zbl,
                (double)zbl_rmax,
                (double)zbl_rmin,
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
        
        at::Tensor betot_residual_tensor = betot_dft_tensor - betot_zbl_tensor;
        at::Tensor bforce_residual_tensor = bforce_dft_tensor - bforce_zbl_tensor;
        at::Tensor bvirial_residual_tensor = bvirial_dft_tensor - bvirial_zbl_tensor;
        double *betot_residual = betot_residual_tensor.data_ptr<double>();
        double (*bforce_residual)[3] = (double (*)[3])bforce_residual_tensor.data_ptr<double>();
        double *bvirial_residual = bvirial_residual_tensor.data_ptr<double>();
        
        find_efv_components_torch_launcher<double>(
            benergy_components,
            bforce_components,
            bvirial_components,
            chebyshev_size,
            scaling,
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
            umax_num_neigh_atoms,
            nghost,
            (double)rmax,
            (double)rmin,
            q_scaler);
        
        find_lin_matrix_lin_vector_torch_launcher<double>(
            lin_matrix,
            lin_vector,
            e_weight,
            f_weight,
            v_weight,
            betot_residual,
            bforce_residual,
            bvirial_residual,
            benergy_components,
            bforce_components,
            bvirial_components,
            alpha_scalar_moments,
            nmus,
            batch_size,
            natoms_pad,
            binum,
            bilist,
            ntypes);

        mirror_lin_matrix_torch_launcher<double>(
            lin_matrix,
            num_parameters);
    }

    // 5.
    return {lin_matrix_tensor,
            lin_vector_tensor};
}


torch::autograd::variable_list LinMatrixLinVectorFunctionCUDA::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list bgrad_outputs_tensor)
{
    at::Tensor bgrad_output_tensor = bgrad_outputs_tensor[0];
    if ( !bgrad_output_tensor.is_contiguous() )
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    
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


torch::autograd::variable_list LinMatrixLinVectorOpCUDA(
    double e_weight,
    double f_weight,
    double v_weight,
    const at::Tensor& betot_dft_tensor,
    const at::Tensor& bforce_dft_tensor,
    const at::Tensor& bvirial_dft_tensor,
    int chebyshev_size,
    double scaling,
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
    return LinMatrixLinVectorFunctionCUDA::apply(
        e_weight,
        f_weight,
        v_weight,
        betot_dft_tensor,
        bforce_dft_tensor,
        bvirial_dft_tensor,
        chebyshev_size,
        scaling,
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


};  // namespace : mtpr
};  // namespace : ai2pot