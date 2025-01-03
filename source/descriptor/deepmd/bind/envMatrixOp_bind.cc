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

#include <torch/torch.h>
#include <torch/extension.h>

#include "../include/envMatrixOp.h"


TORCH_LIBRARY(deepmd, m) {
    m.def(
        "EnvMatrixOp",
        [](const at::Tensor &ilist_tensor, 
           const at::Tensor &numneigh_tensor,
           const at::Tensor &firstneigh_tensor,
           const at::Tensor &relative_coords_tensor,
           const at::Tensor &types_tensor,
           const at::Tensor &umax_num_neigh_atoms_lst_tensor,
           double rcut,
           double rcut_smooth)
        {
            return ai2pot::deepPotSE::EnvMatrixOp(ilist_tensor,
                                                    numneigh_tensor,
                                                    firstneigh_tensor,
                                                    relative_coords_tensor,
                                                    types_tensor,
                                                    umax_num_neigh_atoms_lst_tensor,
                                                    rcut,
                                                    rcut_smooth);
        }
    );
}
