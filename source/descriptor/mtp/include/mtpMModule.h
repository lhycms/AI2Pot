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

#ifndef AI2POT_MTPM_MODULE_H
#define AI2POT_MTPM_MODULE_H
#include <torch/torch.h>

namespace ai2pot {
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

};  // namespace : mtp
};  // namespace : ai2pot

#endif