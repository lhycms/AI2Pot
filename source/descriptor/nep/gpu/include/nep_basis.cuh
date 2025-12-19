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

#ifndef AI2POT_MTP_BASIS_CUH
#define AI2POT_MTP_BASIS_CUH
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
                      CoordType rmin,
                      CoordType distance_ij);


template <typename CoordType>
static __host__ __device__ 
void find_rb_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType rmin,
                       CoordType distance);


template <typename CoordType>
static __host__ __device__ 
void find_rq_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType rmin,
                       CoordType distance_ij);


template <typename CoordType>
static __host__ __device__
void find_switch_func(CoordType& val,
                      CoordType& der2r,
                      CoordType rmax,
                      CoordType rmin,
                      CoordType distance_ij)
{
    CoordType uu = (distance_ij - rmin) / (rmax - rmin);

    if (distance_ij < rmin) {
        val = 0;
        der2r = 0;
    } else if ((distance_ij >= rmin) && (distance_ij <= rmax)) {
        val = std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu -10) + 1;
        der2r = 1 / (rmax - rmin) * (-30*std::pow(uu, 4) + 60*std::pow(uu, 3) - 30*std::pow(uu, 2));;
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
                       CoordType rmin,
                       CoordType distance_ij)
{
    CoordType ders2uu[MAX_CHEBYSHEV_SIZE] = {0.};
    CoordType uu = (2*distance_ij - (rmax + rmin)) / (rmax - rmin);
    CoordType uu_coeff = 2 / (rmax - rmin);

    for (int ii=0; ii<chebyshev_size; ii++) {
        if (ii == 0) {
            vals[ii] = 1;
            ders2uu[ii] = 0;
            ders2r[ii] = ders2uu[ii] * uu_coeff;
        } else if (ii == 1)  {
            vals[ii] = uu;
            ders2uu[ii] = 1;
            ders2r[ii] = ders2uu[ii] * uu_coeff;
        } else {
            vals[ii] = 2*uu*vals[ii-1] - vals[ii-2];
            ders2uu[ii] = 2*vals[ii-1] + 2*uu*ders2uu[ii-1] - ders2uu[ii-2];
            ders2r[ii] = ders2uu[ii] * uu_coeff;
        }
    }
}


template <typename CoordType>
static __host__ __device__ 
void find_rq_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType rmin,
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
                      rmin,
                      distance_ij);
    find_switch_func(switch_func_val,
                     switch_func_der2r,
                     rmax,
                     rmin,
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
