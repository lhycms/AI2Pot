# Copyright (C) 2025 Hanyu Liu
#
# This file is part of AI2Pot.
#
# AI2Pot is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI2Pot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.


from typing import List
import math
import torch
import torch.nn as nn


def build_cart_basis(direction_tensor: torch.Tensor):
    sqrt_three: float = math.sqrt(3.0)
    x: torch.Tensor = direction_tensor[:, 0]
    y: torch.Tensor = direction_tensor[:, 1]
    z: torch.Tensor = direction_tensor[:, 2]
    r2 = x*x + y*y + z*z

    l0_basis_tensor: torch.Tensor = torch.ones_like(x).unsqueeze(-1)
    l1_basis_tensor: torch.Tensor = torch.stack([x, y, z], dim=-1)
    l2_basis_tensor = torch.stack([
        sqrt_three * x * y,
        sqrt_three * y * z,
        0.5 * (3.0 * z * z - r2),
        sqrt_three * x * z,
        0.5 * sqrt_three * (x * x - y * y),
    ], dim=-1)

    return torch.cat(
        [l0_basis_tensor, l1_basis_tensor, l2_basis_tensor],
        dim=-1
    )
