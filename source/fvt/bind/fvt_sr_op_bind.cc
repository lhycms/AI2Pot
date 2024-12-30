#include <torch/torch.h>
#include <torch/extension.h>
#include <climits>
#include <cassert>

#include "../include/force_sr_op.h"
#include "../include/virial_sr_op.h"

TORCH_LIBRARY(fvt, m) {
    m.def(
        "ForceSrOp",
        [](const at::Tensor &binum_tensor,
           const at::Tensor &bilist_tensor,
           const at::Tensor &bnumneigh_tensor,
           const at::Tensor &bfirstneigh_tensor,
           int64_t nghost,
           int64_t umax_num_neighs,
           const at::Tensor &bdei_drij_tensor)
        {   
            assert(nghost <= INT_MAX);
            assert(umax_num_neighs <= INT_MAX);
            return ai2pot::fvt::ForceSrOp(binum_tensor,
                                          bilist_tensor,
                                          bnumneigh_tensor,
                                          bfirstneigh_tensor,
                                          (int)nghost,
                                          (int)umax_num_neighs,
                                          bdei_drij_tensor);
        }
    );

    m.def(
        "VirialSrOp",
        [](const at::Tensor &bnumneigh_tensor,
           const at::Tensor &brcs_tensor,
           int64_t umax_num_neighs,
           const at::Tensor &bdei_drij_tensor)
        {
            assert(umax_num_neighs <= INT_MAX);
            return ai2pot::fvt::VirialSrOp(bnumneigh_tensor,
                                            brcs_tensor,
                                            (int)umax_num_neighs,
                                            bdei_drij_tensor);
        }
    );
}
