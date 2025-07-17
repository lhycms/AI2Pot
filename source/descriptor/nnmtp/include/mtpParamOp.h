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

#ifndef AI2POT_MTP_PARAM_OP_H
#define AI2POT_MTP_PARAM_OP_H
#include <torch/torch.h>

namespace ai2pot {
namespace nnmtp {

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

};  // namespace : nnmtp
};  // namespace : ai2pot

#endif