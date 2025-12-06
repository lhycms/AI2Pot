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

#include "../include/schmidt_orth_torch_launcher.cuh"


namespace ai2pot {
namespace mtpr {

template void find_coeffs_orthogonal_basis_set_torch_launcher<float>(
    float *d_orth_coeffs,
    int ntypes,
    int nmus,
    int chebyshev_size,
    float *d_coeffs);

template void find_coeffs_orthogonal_basis_set_torch_launcher<double>(
    double *d_orth_coeffs,
    int ntypes,
    int nmus,
    int chebyshev_size,
    double *d_coeffs);

};  // namespace : mtpr
};  // namespace : ai2pot