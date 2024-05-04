#ifndef MATERSDK_MTPM_MODULE_H
#define MATERSDK_MTPM_MODULE_H
#include <torch/torch.h>

namespace matersdk {
namespace mtp {
class MtpMModuleImpl : public torch::nn::Module
{
public:
    MtpMModuleImpl(
        int64_t nmus,
        int64_t ntypes,
        int64_t size,
        at::Tensor& rcuts_tensor);
    
    at::Tensor forward(
        int64_t mu,
        int64_t nu,
        int64_t iidx,
        const at::Tensor& ifirstneigh_tensor,
        const at::Tensor& types,
        const at::Tensor& ircs_tensor);

private:
    int64_t nmus;
    int64_t ntypes;
    int64_t size;
    at::Tensor rcuts_tensor;
    torch::nn::ParameterList cheby_coeff_list;
};  // class : MtpMModuleImpl

TORCH_MODULE(MtpMModule);

};  // namespace : matersdk
};  // namespace : mtp

#endif