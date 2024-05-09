#include <torch/torch.h>
#include <torch/extension.h>
#include <climits>
#include <cassert>

#include "../include/mtpParamOp.h"


TORCH_LIBRARY(mtpr, m) {
    m.def(
        "mtpParamOp",
        [](int64_t mtp_level)
        {
            assert(mtp_level <= INT_MAX);
            return ai2pot::mtpr::MtpParamOp((int)mtp_level);
        }
    );

    
}