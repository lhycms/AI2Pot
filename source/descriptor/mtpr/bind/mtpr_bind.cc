#include <torch/torch.h>
#include <torch/extension.h>
#include <climits>
#include <cassert>

#include "../include/mtpParamOp.h"
#include "../include/mtpBasisOp.h"


TORCH_LIBRARY(mtpr, m) {
    m.def(
        "mtpParamOp",
        [](int64_t mtp_level)
        {
            assert(mtp_level <= INT_MAX);
            return ai2pot::mtpr::MtpParamOp((int)mtp_level);
        }
    );

    m.def(
        "mtpBasisOp",
        [](const at::Tensor &alpha_index_basic_tensor,
           const at::Tensor &alpha_index_times_tensor,
           const at::Tensor &alpha_moment_mapping_tensor,
           const at::Tensor &num_mus4moms_tensor,
           const at::Tensor &mus4moms_tensor,
           const at::Tensor &nmus_tensor,
           int64_t ntypes,
           int64_t chebyshev_size,
           const at::Tensor &coeffs_tensor,
           const at::Tensor &binum_tensor,
           const at::Tensor &bilist_tensor,
           const at::Tensor &bnumneigh_tensor,
           const at::Tensor &bfirstneigh_tensor,
           const at::Tensor &brcs_tensor,
           const at::Tensor &btypes_tensor,
           int64_t umax_num_neighs,
           double rmax,
           double rmin)
        {
            assert(ntypes <= INT_MAX);
            assert(chebyshev_size <= INT_MAX);
            assert(umax_num_neighs <= INT_MAX);

            return ai2pot::mtpr::MtpBasisOp(
                alpha_index_basic_tensor,
                alpha_index_times_tensor,
                alpha_moment_mapping_tensor,
                num_mus4moms_tensor,
                mus4moms_tensor,
                nmus_tensor,
                (int)ntypes,
                (int)chebyshev_size,
                coeffs_tensor,
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                (int)umax_num_neighs,
                rmax,
                rmin);
        }
    );
}