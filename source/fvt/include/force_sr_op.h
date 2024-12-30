#ifndef AI2POT_FORCE_SR_OP_H
#define AI2POT_FORCE_SR_OP_H
#include <torch/torch.h>


namespace ai2pot {
namespace fvt {

class ForceSrFunction : public torch::autograd::Function<ForceSrFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor& binum_tensor,
        const at::Tensor& bilist_tensor,
        const at::Tensor& bnumneigh_tensor,
        const at::Tensor& bfirstneigh_tensor,
        int nghost,
        int umax_num_neighs,
        const at::Tensor& bdei_drij_tensor);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs);
};  // class : ForceSrFunction


torch::autograd::variable_list ForceSrOp(
    const at::Tensor& binum_tensor,
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    int nghost,
    int umax_num_neighs,
    const at::Tensor& bdei_drij_tensor);

};  // namespace : fvt
};  // namespace : ai2pot

#endif