#ifndef AI2POT_VIRIAL_SR_OP_H
#define AI2POT_VIRIAL_SR_OP_H
#include <torch/torch.h>

namespace ai2pot {
namespace fvt {

class VirialSrFunction : public torch::autograd::Function<VirialSrFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &bnumneigh_tensor,
        const at::Tensor &brcs_tensor,
        int umax_num_neighs,
        const at::Tensor &bdei_drij_tensor);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs);
};  // class : VirialSrFunction


torch::autograd::variable_list VirialSrOp(
    const at::Tensor &bnumneigh_tensor,
    const at::Tensor &brcs_tensor,
    int umax_num_neighs,
    const at::Tensor &bdei_drij_tensor);

};  // namespace : fvt
};  // namespace : ai2pot

#endif