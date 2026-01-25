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

#ifndef AI2POT_NEP_NEP_BASIS_CUH
#define AI2POT_NEP_NEP_BASIS_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "./nep_utilities.cuh"


namespace ai2pot {
namespace nep {

template <typename CoordType>
static __host__ __device__ 
void find_switch_func(CoordType &val,
                      CoordType &der2r,
                      CoordType rmax,
                      CoordType distance_ij);


template <typename CoordType>
static __host__ __device__ 
void find_rb_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType distance);


template <typename CoordType>
static __host__ __device__ 
void find_rq_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType distance_ij);


template <typename CoordType>
static __host__ __device__
void find_switch_func(CoordType& val,
                      CoordType& der2r,
                      CoordType rmax,
                      CoordType distance_ij)
{
    CoordType uu = M_PI * distance_ij / rmax;

    if (distance_ij <= rmax) {
        val = 0.5 * (1 + std::cos(uu));
        der2r = -0.5 * std::sin(uu) * M_PI / rmax;
    } else {
        val = 0;
        der2r = 0;
    }
}


template <typename CoordType>
static __host__ __device__
void find_rb_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType distance_ij)
{
    CoordType T_vals[MAX_CHEBYSHEV_SIZE] = {0.};
    CoordType T_ders2uu[MAX_CHEBYSHEV_SIZE] = {0.};
    CoordType ders2uu[MAX_CHEBYSHEV_SIZE] = {0.};
    CoordType uu = 2*std::pow(distance_ij/rmax-1, 2) - 1;
    CoordType uu_coeff = 4.0/rmax * (distance_ij/rmax - 1);

    for (int ii=0; ii<chebyshev_size; ii++) {
        if (ii == 0) {
            T_vals[ii] = 1.0;
            T_ders2uu[ii] = 0.0;

            vals[ii] = (T_vals[ii] + 1) * 0.5;
            ders2uu[ii] = T_ders2uu[ii] * 0.5;
            ders2r[ii] = uu_coeff * ders2uu[ii];
        } else if (ii == 1)  {
            T_vals[ii] = uu;
            T_ders2uu[ii] = 1.0;

            vals[ii] = (T_vals[ii] + 1) / 2;
            ders2uu[ii] = T_ders2uu[ii] * 0.5;
            ders2r[ii] = uu_coeff * ders2uu[ii];
        } else {
            T_vals[ii]= 2*uu*T_vals[ii-1] - T_vals[ii-2];
            T_ders2uu[ii] = 2*T_vals[ii-1] 
                            + 2*uu*T_ders2uu[ii-1] 
                            - T_ders2uu[ii-2];

            vals[ii] = (T_vals[ii] + 1) / 2;
            ders2uu[ii] = T_ders2uu[ii] * 0.5;
            ders2r[ii] = uu_coeff * ders2uu[ii];
        }
    }
}


template <typename CoordType>
static __host__ __device__ 
void find_rq_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType distance_ij)
{
    CoordType rb_chebyshev_vals[MAX_CHEBYSHEV_SIZE] = {0.0};
    CoordType rb_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE] = {0.0};
    CoordType switch_func_val = 0.0;
    CoordType switch_func_der2r = 0.0;
    find_rb_chebyshev(rb_chebyshev_vals,
                      rb_chebyshev_ders2r,
                      chebyshev_size,
                      rmax,
                      distance_ij);
    find_switch_func(switch_func_val,
                     switch_func_der2r,
                     rmax,
                     distance_ij);
    for (int ii=0; ii<chebyshev_size; ii++) {
        vals[ii] = switch_func_val * rb_chebyshev_vals[ii];
        ders2r[ii] = (switch_func_der2r * rb_chebyshev_vals[ii]
                      + switch_func_val * rb_chebyshev_ders2r[ii]);
    }
}

};  // namespace : nep
};  // namespace : ai2pot


#endif
