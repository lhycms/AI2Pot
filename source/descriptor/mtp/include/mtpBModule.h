#ifndef MATERSDK_MTPB_MODULE_H
#define MATERSDK_MTPB_MODULE_H
#include <torch/torch.h>
#include "./mtpLevel.h"
#include "./mtpMModule.h"

namespace matersdk {
namespace mtp {

class MtpBModuleImpl : public torch::nn::Module
{
public:
    MtpBModuleImpl(
        int64_t max_level,
        //int64_t nmus,
        int64_t ntypes,
        int64_t size,
        at::Tensor& rcuts_tensor);

    /**
     * @brief 
     * 
     * @param ilist_tensor .sizes() = [nbatches, ncenters]
     * @param firstneigh_tensor .sizes() = [nbatches, ncenters, umax_nneighs]
     * @param rcs_tensor .sizes() = [nbatches, ncenters, umax_nneighs, 3]
     * @param types_tensor .sizes() = [nbatches, ncenters + nghost]
     * @return at::Tensor .sizes() = [nbatches, ncenters, nbasis]
     */
    at::Tensor forward(
        at::Tensor& ilist_tensor,
        at::Tensor& firstneigh_tensor,
        at::Tensor& rcs_tensor,
        at::Tensor& types_tensor);
    
private:
    int64_t _max_level;
    int64_t _nmus;
    int64_t _ntypes;
    int64_t _size;
    at::Tensor _rcuts_tensor;
    MtpMCoeffPairCombs _coeff_pair_combs;
    torch::nn::ModuleList _mtp_m_list;   // .size() = [1]
};  // class : MtpBModuleImpl
TORCH_MODULE(MtpBModule);


};  // namespace : mtp
};  // namespace : matersdk

#endif
