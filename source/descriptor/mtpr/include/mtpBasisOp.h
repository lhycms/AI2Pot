#ifndef AI2POT_MTPR_MTP_BASIS_OP_H
#define AI2POT_MTPR_MTP_BASIS_OP_H
#include <torch/torch.h>

namespace ai2pot {
namespace mtpr {

class MtpBasisFunction : public torch::autograd::Function<MtpBasisFunction>
{
public:
    static torch::autograd::variable_list forward(
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
        double rmin);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list bgrad_outputs_tensor);
};  // class : MtpBasisFunction


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
    double rmin);

};  // namespace : mtpr
};  // namespace : ai2pot


#endif