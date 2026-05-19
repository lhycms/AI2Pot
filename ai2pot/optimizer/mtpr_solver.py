from typing import (Optional, List, Union)

import torch
from torch.utils.data import (Dataset, DataLoader)
import lightning as L

from ai2pot.fromcc import coeffsSchmidtOrthOp
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp


class LinearMtpSolver(object):
    BATCH_SIZE_HERE: int = 500

    def __init__(self):
        pass


    def orthogonalize(self,
                     linear_mtp: LitLinearMtp):
        with torch.no_grad():
            orthogonal_coeffs_tensor: torch.Tensor = coeffsSchmidtOrthOp(linear_mtp.ntypes,
                                                                         linear_mtp.nmus,
                                                                         linear_mtp.chebyshev_size,
                                                                         linear_mtp.coeffs_tensor)[0]
            linear_mtp.coeffs_tensor.copy_(orthogonal_coeffs_tensor)
        


