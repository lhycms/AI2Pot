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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "./mtp_utilities.cuh"


namespace ai2pot {
namespace mtpr {


template <typename CoordType>
__device__
void accumulate_efv_components_atom(
    CoordType *energy_components,
    CoordType *force_components,
    CoordType *virial_components,
    int chebyshev_size,
    CoordType *coeffs,
    CoordType *linear_coeffs,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_times_count)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int silist,
    int snumneigh,
    int *sfirstneigh,
    CoordType (*srcs)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    
}


};  // namespace : mtpr
};  // namespace : ai2pot