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

#include "../include/mtpBasis.cuh"


// Explicitly instantiate template for float
template
void find_mtp_basis_val_der_cuda_launcher<float>();

// Explicitly instantiate template for double
template
void find_mtp_basis_val_der_cuda_launcher<double>();
