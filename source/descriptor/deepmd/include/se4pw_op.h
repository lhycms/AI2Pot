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

#ifndef AI2POT_SE4PW_OP_H
#define AI2POT_SE4PW_OP_H

#include "./se4pw.h"
#include <torch/torch.h>
#include <cstdlib>

namespace ai2pot {
namespace deepPotSE {
class Se4pwOp {
public:
    static torch::autograd::variable_list forward(
        int batch_size,
        int inum,
        at::Tensor& ilist,
        at::Tensor& numneigh,
        at::Tensor& firstneigh,
        at::Tensor& x,
        at::Tensor& types,  // x, types 一一对应
        int ntypes,
        at::Tensor& num_neigh_atoms_lst,
        double rcut,             // Python 中默认是双精度浮点数
        double rcut_smooth);     // Python 中默认是双精度浮点数


    static at::Tensor get_prim_indices_from_ai2pot(
        int batch_size,
        int inum,
        at::Tensor& ilist,
        at::Tensor& numneigh,
        at::Tensor& firstneigh,
        at::Tensor& types,
        int ntypes,
        at::Tensor& num_neigh_atoms_lst);

};  // class : Se4pwOp
};  // namespace : deepPotSE
};  // namespace : ai2pot


#endif