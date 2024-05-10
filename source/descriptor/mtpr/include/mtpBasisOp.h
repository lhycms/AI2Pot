#include <torch/torch.h>


namespace ai2pot {
namespace mtpr {

class MtpBasisFunction : public torch::autograd::Function<MtpBasisFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &bilist_tensor,
        const at::Tensor &bnumneigh_tensor,
        const at::Tenosr &bfirstneigh_tensor,
        const at::Tensor &brcs_tensor,
        const at::Tenosr &btypes_tensor,
        int umax_num_neighs,
        double rmax,
        double rmin);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs);
};  // class : MtpBasisFunction


torch::autograd::variable_list MtpBasisOp(
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