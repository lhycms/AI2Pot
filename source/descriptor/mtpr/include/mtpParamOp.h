#ifndef AI2POT_MTP_PARAM_OP_H
#define AI2POT_MTP_PARAM_OP_H
#include <torch/torch.h>

namespace ai2pot {
namespace mtpr {

class MtpParamFunction : public torch::autograd::Function<MtpParamFunction>
{
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int mtp_level);
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs);
};  // class : MtpParamFunction

torch::autograd::variable_list MtpParamOp(int mtp_level);

};  // namespace : mtpr
};  // namespace : ai2pot

#endif