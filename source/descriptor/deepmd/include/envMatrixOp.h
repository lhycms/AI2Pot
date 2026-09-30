/*
    Copyright 2025 Hanyu Liu

    SPDX-License-Identifier: LGPL-3.0-only

    This file is part of AI2Pot.

    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, version 3 of the License.

    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with AI2Pot. If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef AI2POT_ENV_MATRIX_OP_H
#define AI2POT_ENV_MATRIX_OP_H

#include <torch/torch.h>
#include "./envMatrix.h"


namespace ai2pot {
namespace deepPotSE {

class EnvMatrixFunction : public torch::autograd::Function<EnvMatrixFunction> {
public: 
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& ilist_tensor,
        const at::Tensor& numneigh_tensor,
        const at::Tensor& firstneigh_tensor,
        const at::Tensor& relative_coords_tensor,
        const at::Tensor& types_tensor,
        const at::Tensor& umax_num_neigh_atoms_lst_tensor,
        double rcut,
        double rcut_smooth);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs);
};  // class : EnvMatrixFunction


torch::autograd::variable_list EnvMatrixOp(
    const at::Tensor& ilist_tensor,
    const at::Tensor& numneigh_tensor,
    const at::Tensor& firstneigh_tensor,
    const at::Tensor& relative_coords_tensor,
    const at::Tensor& types_tensor,
    const at::Tensor& umax_num_neigh_atoms_lst_tensor,
    double rcut,
    double rcut_smooth);

};  // namespace : deepPotSE
};  // namespace : ai2pot

#endif