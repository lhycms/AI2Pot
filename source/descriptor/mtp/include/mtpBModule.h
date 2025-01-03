/*
    Copyright 2025 Hanyu Liu
    This file is part of AI2Pot.
    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef AI2POT_MTPB_MODULE_H
#define AI2POT_MTPB_MODULE_H
#include <torch/torch.h>
#include "./mtpLevel.h"
#include "./mtpMModule.h"

namespace ai2pot {
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
};  // namespace : ai2pot

#endif
