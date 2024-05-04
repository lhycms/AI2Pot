#ifndef MATERSDK_MTPM_RADIAL_OP
#define MATERSDK_MTPM_RADIAL_OP
#include <torch/torch.h>
#include "./mtpMRadial.h"

namespace matersdk {
namespace mtp {

class MtpQFunction : public torch::autograd::Function<MtpQFunction>
{
public:
    // Returns: shape = [nneigh, size]
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        int64_t size,
        const at::Tensor& rcuts_tensor,        // rs_tensor = [rcut, rcut_smooth]
        const at::Tensor& rcs_tensor);         // .shape = [nneigh, 3];
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs);
};  // class MtpQFunction


torch::autograd::variable_list MtpQOp(
    int64_t size,
    const at::Tensor& rcuts_tensor,
    const at::Tensor& distances_tensor);

};  // namespace : mtp
};  // namespace : matersdk

#endif
