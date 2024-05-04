#ifndef MATERSDK_MTPM_ANGULAR_OP_H
#define MATERSDK_MTPM_ANGULAR_OP_H
#include <torch/torch.h>

namespace matersdk {
namespace mtp {

class OuterNu0Function : public torch::autograd::Function<OuterNu0Function>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        at::Tensor ircs_tensor);    // .shape = [nneighs, 3]
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs);

};  // class : OuterNuFunction

torch::autograd::variable_list OuterNu0Op(const at::Tensor& ircs_tensor);

at::Tensor MtpMAngularOp(
    const at::Tensor& ircs_tensor,  // .shape = [nneighs, 3]
    int nu);

}; // namespace : mtp
}; // namespace : matersdk

#endif
