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
    const at::Tensor &mus4moms_tensor,
    const at::Tensor &coeffs_tensor,
    const at::Tensor &bilist_tensor,
    const at::Tensor &bnumneigh_tensor,
    const at::Tensor &bfirstnumneigh_tensor,
    const at::Tensor &brcs_tensor,
    const at::Tensor &btypes_tensor,
    int umax_num_neighs,
    double rmax,
    double rmin)
{
    assert(bilist_tensor.scalar_type() == torch::kInt32);
    assert(bnumneigh_tensor.scalar_type() == torch::kInt32);
    assert(bfirstnumneigh_tensor.scalar_type() == torch::kInt32);
    assert(btypes_tensor.scalar_type() == torch::kInt32);
    assert(coeffs_tensor.scalar_type() == brcs_tensor.scalar_type());

    int nbatches = (int)bilist_tensor.size(0);
    int nlocal = (int)bilist_tensor.size(1);
    int alpha_moments_count = (int)mus4moms_tensor.size(0);
    int alpha_index_basic_count = (int)alpha_index_basic_tensor.size(0);
    int alpha_index_times_count = (int)alpha_index_times_tensor.size(0);
    int alpha_scalar_moments = (int)alpha_moment_mapping_tensor.size(0);
    int max_mus4moms = (int)mus4moms_tensor.size(1);
    int num_coeffs = (int)coeffs_tensor.flatten().size(0);
    c10::TensorOptions int_options = c10::TensorOptions()
        .dtype(torch::kInt32)
        .device(brcs_tensor.device());
    c10::TensorOptions float_options;
    at::Tensor bmtp_basis_tensor;
    at::Tensor bmtp_basis_rij_jacobian_tensor;
    at::Tensor bmtp_basis_coeffs_jacobian_tensor;


    if (brcs_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat32)
            .device(brcs_tensor.device());
        bmtp_basis_tensor = at::zeros({nbatches, alpha_scalar_moments}, float_options);
        bmtp_basis_rij_jacobian_tensor = at::zeros({nbatches, alpha_scalar_moments, nlocal, 3}, float_options);
        bmtp_basis_coeffs_jacobian_tensor = at::zeros({nbatches, alpha_scalar_moments, num_coeffs}, float_options);

        
    } else {

    }

    ctx->save_for_backward({bmtp_basis_rij_jacobian_tensor, bmtp_basis_coeffs_jacobian_tensor});
    return {bmtp_basis_tensor};
}

};  // namespace : mtpr
};  // namespace : ai2pot