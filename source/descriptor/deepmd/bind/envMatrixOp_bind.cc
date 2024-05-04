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
            return matersdk::deepPotSE::EnvMatrixOp(ilist_tensor,
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
