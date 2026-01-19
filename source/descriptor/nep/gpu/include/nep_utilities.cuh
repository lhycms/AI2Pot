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

#ifndef AI2POT_NEP_NEP_UTILITIES_CUH
#define AI2POT_NEP_NEP_UTILITIES_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cassert>

// 1. Hyperparameters.
#define MAX_CHEBYSHEV_SIZE 20
#define L_MAX 4
#define L_MAX_PLUS_ONE 5
#define MAX_NUM_RADIAL_BASIS 20
#define MAX_NUM_ANGULAR_BASIS 12
#define MAX_NUM_SINLM 288   // (3 + 5 + 7 + 9) * 12
#define MAX_NUM_DESCRIPTORS 68  // 20 + 12*4


// 3 + 5 + 7 + 9
const int FLATTEN_LM_DIM = 24;
__constant__ double C3B[FLATTEN_LM_DIM] = {
    0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
    0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
    0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
    1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
    0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
    1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606};



// 2. About CUDA
#ifdef __INTELLISENSE__
#define KERNEL_ARG2(grid_size, block_size)
#define KERNEL_ARG3(grid_size, block_size, sh_mem)
#define KERNEL_ARG4(grid_size, block_size, sh_mem, stream)
#else
#define KERNEL_ARG2(grid_size, block_size) <<<grid_size, block_size>>>
#define KERNEL_ARG3(grid_size, block_size, sh_mem) <<<grid_size, block_size, sh_mem>>>
#define KERNEL_ARG4(grid_size, block_size, sh_mem, stream) <<<grid_size, block_size, sh_mem, stream>>>
#endif


#define CHECK_CUDA_API(call)                                                    \
do {                                                                            \
    cudaError_t error_code = call;                                              \
    if (error_code != cudaSuccess)                                              \
    {                                                                           \
        printf("CUDA Error:\n");                                                \
        printf("\t1. File : %s\n", __FILE__);                                   \
        printf("\t2. Function : %s\n", __FUNCTION__);                           \
        printf("\t3. Line : %d\n", __LINE__);                                   \
        printf("\t4. Error code : %d\n", error_code);                           \
        printf("\t5. Error text : %s\n", cudaGetErrorString(error_code));       \
    }                                                                           \
} while(0);


#define CHECK_CUDA_KERNEL                               \
do {                                                    \
    CHECK_CUDA_API( cudaDeviceSynchronize() );          \
    CHECK_CUDA_API( cudaGetLastError() );               \
} while(0);



// 3. Nep Utilities
namespace ai2pot {
namespace nep {

class NepIndex {
public:
    static __host__ __device__ __forceinline__
    int get_num_descriptors(
        int n_radial_basis,
        int n_angular_basis,
        int l_max) 
    {
        assert((l_max >= 0) && (l_max<=4));

        return n_radial_basis + n_angular_basis * l_max;
    }


    static __host__ __device__ __forceinline__
    int get_num_Sinlm(
        int n_angular_basis,
        int l_max)
    {
        assert((l_max >= 0) && (l_max<=4));

        return n_angular_basis * (l_max*l_max + 2*l_max);
    }


    static __host__ __device__ __forceinline__
    int get_Clm_index(
        int l,
        int mp)
    {
        return (l*l - 1) + mp;
    }


    static __host__ __device__ __forceinline__
    int get_Sinlm_index(
        int l_max,
        int mu,
        int l,
        int mp)
    {
        return mu*(l_max*l_max + 2*l_max) + (l*l - 1) + mp;
    }

    static __host__ __device__ __forceinline__
    int get_qinl_index(
        int l_max,
        int mu,
        int l)
    {
        return mu*l_max + (l-1);
    }
};  // class : NepIndex



template <typename CoordType>
class Blm {
public:
    static __host__ __device__ __forceinline__
    void find_blm_val(
        CoordType *blm_ptr,
        int l,
        int mp,
        CoordType *rc,
        CoordType distance_ij)
    {
        assert((l>=1) && (l<=L_MAX));
        assert((mp>=0) && (mp<(2*l+1)));

        switch(l) {
            case 1:
                switch(mp) {
                    case 0:
                        (*blm_ptr) = rc[2];
                        break;
                    case 1:
                        (*blm_ptr) = rc[0];
                        break;
                    case 2:
                        (*blm_ptr) = rc[1];
                        break;
                }
                break;
            case 2:
                switch(mp) {
                    case 0:
                        //(*blm_ptr) = 3*std::pow(rc[2], 2) - std::pow(distance_ij, 2);
                        (*blm_ptr) = 3*rc[2]*rc[2] - distance_ij*distance_ij;
                        break;
                    case 1:
                        (*blm_ptr) = rc[0] * rc[2];
                        break;
                    case 2:
                        (*blm_ptr) = rc[1] * rc[2];
                        break;
                    case 3:
                        //(*blm_ptr) = std::pow(rc[0], 2) - std::pow(rc[1], 2);
                        (*blm_ptr) = rc[0]*rc[0] - rc[1]*rc[1];
                        break;
                    case 4:
                        (*blm_ptr) = 2*rc[0]*rc[1];
                        break;
                }
                break;
            case 3:
                switch(mp) {
                    case 0:
                        //(*blm_ptr) = (5*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2)) * rc[2];
                        (*blm_ptr) = (5*rc[2]*rc[2] - 3*distance_ij*distance_ij) * rc[2];
                        break;
                    case 1:
                        //(*blm_ptr) = (5*std::pow(rc[2], 2) - std::pow(distance_ij, 2)) * rc[0];
                        (*blm_ptr) = (5*rc[2]*rc[2] - distance_ij*distance_ij) * rc[0];
                        break;
                    case 2:
                        //(*blm_ptr) = (5*std::pow(rc[2], 2) - std::pow(distance_ij, 2)) * rc[1];
                        (*blm_ptr) = (5*rc[2]*rc[2] - distance_ij*distance_ij) * rc[1];
                        break;
                    case 3:
                        //(*blm_ptr) = (std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[2];
                        (*blm_ptr) = (rc[0]*rc[0] - rc[1]*rc[1]) * rc[2];
                        break;
                    case 4:
                        (*blm_ptr) = 2*rc[0]*rc[1]*rc[2];
                        break;
                    case 5:
                        //(*blm_ptr) = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0];
                        (*blm_ptr) = (rc[0]*rc[0] - 3*rc[1]*rc[1]) * rc[0];
                        break;
                    case 6:
                        //(*blm_ptr) = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[1];
                        (*blm_ptr) = (3*rc[0]*rc[0] - rc[1]*rc[1]) * rc[1];
                        break;
                }
                break;
            case 4:
                switch(mp) {
                    case 0:
                        //(*blm_ptr) = (35*std::pow(rc[2], 2) - 30*std::pow(distance_ij, 2))*std::pow(rc[2], 2)
                        //             + 3*std::pow(distance_ij, 4);
                        (*blm_ptr) = (35*rc[2]*rc[2] - 30*distance_ij*distance_ij)*rc[2]*rc[2]
                                     + 3*distance_ij*distance_ij*distance_ij*distance_ij;
                        break;
                    case 1:
                        //(*blm_ptr) = (7*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2))*rc[0]*rc[2];
                        (*blm_ptr) = (7*rc[2]*rc[2] - 3*distance_ij*distance_ij) * rc[0]*rc[2];
                        break;
                    case 2:
                        //(*blm_ptr) = (7*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2))*rc[1]*rc[2];
                        (*blm_ptr) = (7*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2))*rc[1]*rc[2];
                        break;
                    case 3:
                        //(*blm_ptr) = (7*std::pow(rc[2], 2) - std::pow(distance_ij, 2)) 
                        //             * (std::pow(rc[0], 2) - std::pow(rc[1], 2));
                        (*blm_ptr) = (7*rc[2]*rc[2] - distance_ij*distance_ij) 
                                     * (rc[0]*rc[0] - rc[1]*rc[1]);
                        break;
                    case 4:
                        //(*blm_ptr) = (7*std::pow(rc[2], 2) - std::pow(distance_ij, 2))
                        //             * 2 * rc[0] * rc[1];
                        (*blm_ptr) = (7*rc[2]*rc[2] - distance_ij*distance_ij)
                                     * 2 * rc[0] * rc[1];
                        break;
                    case 5:
                        //(*blm_ptr) = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0] * rc[2];
                        (*blm_ptr) = (rc[0]*rc[0] - 3*rc[1]*rc[1]) * rc[0] * rc[2];
                        break;
                    case 6:
                        //(*blm_ptr) = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[1] * rc[2];
                        (*blm_ptr) = (3*rc[0]*rc[0] - rc[1]*rc[1]) * rc[1] * rc[2];
                        break;
                    case 7:
                        //(*blm_ptr) = std::pow(std::pow(rc[0], 2) - std::pow(rc[1], 2), 2)
                        //             - 4*std::pow(rc[0], 2)*std::pow(rc[1], 2);
                        (*blm_ptr) = (rc[0]*rc[0] - rc[1]*rc[1]) * (rc[0]*rc[0] - rc[1]*rc[1])
                                     - 4*rc[0]*rc[0]*rc[1]*rc[1];
                        break;
                    case 8:
                        //(*blm_ptr) = 4*(std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[0] * rc[1];
                        (*blm_ptr) = 4*(rc[0]*rc[0] - rc[1]*rc[1]) * rc[0] * rc[1];
                        break;
                }
                break;
        }
    }


    static __host__ __device__ __forceinline__
    void find_blm_der2xyz(
        CoordType *blm_der2xyz,
        int l,
        int mp,
        CoordType *rc,
        CoordType distance_ij)
    {
        assert((l>=1) && (l<=L_MAX));
        assert((mp>=0) && (mp<(2*l+1)));

        switch(l) {
            case 1:
                switch(mp) {
                    case 0:
                        blm_der2xyz[0] = 0.0;
                        blm_der2xyz[1] = 0.0;
                        blm_der2xyz[2] = 1.0;
                        break;
                    case 1:
                        blm_der2xyz[0] = 1.0;
                        blm_der2xyz[1] = 0.0;
                        blm_der2xyz[2] = 0.0;
                        break;
                    case 2:
                        blm_der2xyz[0] = 0.0;
                        blm_der2xyz[1] = 1.0;
                        blm_der2xyz[2] = 0.0;
                        break;
                }
                break;
            case 2:
                switch(mp) {
                    case 0:
                        blm_der2xyz[0] = -2*rc[0];
                        blm_der2xyz[1] = -2*rc[1];
                        blm_der2xyz[2] = 4*rc[2];
                        break;
                    case 1:
                        blm_der2xyz[0] = rc[2];
                        blm_der2xyz[1] = 0.0;
                        blm_der2xyz[2] = rc[0];
                        break;
                    case 2:
                        blm_der2xyz[0] = 0.0;
                        blm_der2xyz[1] = rc[2];
                        blm_der2xyz[2] = rc[1];
                        break;
                    case 3:
                        blm_der2xyz[0] = 2*rc[0];
                        blm_der2xyz[1] = -2*rc[1];
                        blm_der2xyz[2] = 0.0;
                        break;
                    case 4:
                        blm_der2xyz[0] = 2*rc[1];
                        blm_der2xyz[1] = 2*rc[0];
                        blm_der2xyz[2] = 0.0;
                        break;
                }
                break;
            case 3:
                switch(mp) {
                    case 0:
                        blm_der2xyz[0] = -6*rc[0]*rc[2];
                        blm_der2xyz[1] = -6*rc[1]*rc[2];
                        //blm_der2xyz[2] = 9*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2);
                        blm_der2xyz[2] = 9*rc[2]*rc[2] - 3*distance_ij*distance_ij;
                        break;
                    case 1:
                        //blm_der2xyz[0] = -2*std::pow(rc[0], 2) + 5*std::pow(rc[2], 2)
                        //                 - std::pow(distance_ij, 2);
                        blm_der2xyz[0] = -2*rc[0]*rc[0] + 5*rc[2]*rc[2]
                                         - distance_ij*distance_ij;
                        blm_der2xyz[1] = -2*rc[0]*rc[1];
                        blm_der2xyz[2] = 8*rc[0]*rc[2];
                        break;
                    case 2:
                        blm_der2xyz[0] = -2*rc[0]*rc[1];
                        //blm_der2xyz[1] = -2*std::pow(rc[1], 2) + 5*std::pow(rc[2], 2) 
                        //                 - std::pow(distance_ij, 2);
                        blm_der2xyz[1] = -2*rc[1]*rc[1] + 5*rc[2]*rc[2] 
                                         - distance_ij*distance_ij;
                        blm_der2xyz[2] = 8*rc[1]*rc[2];
                        break;
                    case 3:
                        blm_der2xyz[0] = 2*rc[0]*rc[2];
                        blm_der2xyz[1] = -2*rc[1]*rc[2];
                        //blm_der2xyz[2] = std::pow(rc[0], 2) - std::pow(rc[1], 2);
                        blm_der2xyz[2] = rc[0]*rc[0] - rc[1]*rc[1];
                        break;
                    case 4:
                        blm_der2xyz[0] = 2*rc[1]*rc[2];
                        blm_der2xyz[1] = 2*rc[0]*rc[2];
                        blm_der2xyz[2] = 2*rc[0]*rc[1];
                        break;
                    case 5:
                        //blm_der2xyz[0] = 3*std::pow(rc[0], 2) - 3*std::pow(rc[1], 2);
                        blm_der2xyz[0] = 3*rc[0]*rc[0] - 3*rc[1]*rc[1];
                        blm_der2xyz[1] = -6*rc[0]*rc[1];
                        blm_der2xyz[2] = 0.0;
                        break;
                    case 6:
                        blm_der2xyz[0] = 6*rc[0]*rc[1];
                        //blm_der2xyz[1] = 3*std::pow(rc[0], 2) - 3*std::pow(rc[1], 2);
                        blm_der2xyz[1] = 3*rc[0]*rc[0] - 3*rc[1]*rc[1];
                        blm_der2xyz[2] = 0.0;
                        break;
                }
                break;
            case 4:
                switch(mp) {
                    case 0:
                        //blm_der2xyz[0] = -60*rc[0]*std::pow(rc[2], 2) 
                        //                 + 12*std::pow(distance_ij, 2)*rc[0];
                        blm_der2xyz[0] = -60*rc[0]*rc[2]*rc[2]
                                         + 12*distance_ij*distance_ij*rc[0];
                        //blm_der2xyz[1] = -60*rc[1]*std::pow(rc[2], 2) 
                        //                 + 12*std::pow(distance_ij, 2)*rc[1];
                        blm_der2xyz[1] = -60*rc[1]*rc[2]*rc[2]
                                         + 12*distance_ij*distance_ij*rc[1];
                        //blm_der2xyz[2] = 80*std::pow(rc[2], 3) 
                        //                 - 48*std::pow(distance_ij, 2)*rc[2];
                        blm_der2xyz[2] = 80*rc[2]*rc[2]
                                         - 48*distance_ij*distance_ij*rc[2];
                        break;
                    case 1:
                        //blm_der2xyz[0] = -6*std::pow(rc[0], 2)*rc[2] 
                        //                 + 7*std::pow(rc[2], 3) 
                        //                 - 3*std::pow(distance_ij, 2)*rc[2];
                        blm_der2xyz[0] = -6*rc[0]*rc[0]*rc[2]
                                         + 7*rc[2]*rc[2]*rc[2]
                                         - 3*distance_ij*distance_ij*rc[2];
                        blm_der2xyz[1] = -6*rc[0]*rc[1]*rc[2];;
                        //blm_der2xyz[2] = 15*rc[0]*std::pow(rc[2], 2) 
                        //                 - 3*std::pow(distance_ij, 2)*rc[0];
                        blm_der2xyz[2] = 15*rc[0]*rc[2]*rc[2]
                                         - 3*distance_ij*distance_ij*rc[0];
                        break;
                    case 2:
                        blm_der2xyz[0] = -6*rc[0]*rc[1]*rc[2];
                        //blm_der2xyz[1] = -6*std::pow(rc[1], 2)*rc[2] 
                        //                 + 7*std::pow(rc[2], 3) 
                        //                 - 3*std::pow(distance_ij, 2)*rc[2];
                        blm_der2xyz[1] = -6*rc[1]*rc[1]*rc[2] 
                                         + 7*rc[2]*rc[2]*rc[2]
                                         - 3*distance_ij*distance_ij*rc[2];
                        //blm_der2xyz[2] = 15*rc[1]*std::pow(rc[2], 2) 
                        //                 - 3*std::pow(distance_ij, 2)*rc[1];
                        blm_der2xyz[2] = 15*rc[1]*rc[2]*rc[2]
                                         - 3*distance_ij*distance_ij*rc[1];
                        break;
                    case 3:
                        //blm_der2xyz[0] = -2*std::pow(rc[0], 3) 
                        //                 + 2*rc[0]*std::pow(rc[1], 2) 
                        //                 + 14*rc[0]*std::pow(rc[2], 2) 
                        //                 - 2*std::pow(distance_ij, 2)*rc[0];
                        blm_der2xyz[0] = -2*rc[0]*rc[0]*rc[0]
                                         + 2*rc[0]*rc[1]*rc[1]
                                         + 14*rc[0]*rc[2]*rc[2]
                                         - 2*distance_ij*distance_ij*rc[0];
                        //blm_der2xyz[1] = -2*std::pow(rc[0], 2)*rc[1] 
                        //                 + 2*std::pow(rc[1], 3) 
                        //                 - 14*rc[1]*std::pow(rc[2], 2) 
                        //                 + 2*std::pow(distance_ij, 2)*rc[1];
                        blm_der2xyz[1] = -2*rc[0]*rc[0]*rc[1] 
                                         + 2*rc[1]*rc[1]*rc[1]
                                         - 14*rc[1]*rc[2]*rc[2]
                                         + 2*distance_ij*distance_ij*rc[1];
                        //blm_der2xyz[2] = 12*rc[2] * (std::pow(rc[0], 2) - std::pow(rc[1], 2));
                        blm_der2xyz[2] = 12*rc[2] * (rc[0]*rc[0] - rc[1]*rc[1]);
                        break;
                    case 4:
                        //blm_der2xyz[0] = -4*std::pow(rc[0], 2)*rc[1] 
                        //                 + 14*rc[1]*std::pow(rc[2], 2) 
                        //                 - 2*std::pow(distance_ij, 2)*rc[1];
                        blm_der2xyz[0] = -4*rc[0]*rc[0]*rc[1] 
                                         + 14*rc[1]*rc[2]*rc[2]
                                         - 2*distance_ij*distance_ij*rc[1];
                        //blm_der2xyz[1] = -4*rc[0]*std::pow(rc[1], 2) 
                        //                 + 14*rc[0]*std::pow(rc[2], 2) 
                        //                 - 2*std::pow(distance_ij, 2)*rc[0];
                        blm_der2xyz[1] = -4*rc[0]*rc[1]*rc[1]
                                         + 14*rc[0]*rc[2]*rc[2]
                                         - 2*distance_ij*distance_ij*rc[0];
                        blm_der2xyz[2] = 24*rc[0]*rc[1]*rc[2];
                        break;
                    case 5:
                        //blm_der2xyz[0] = 3*std::pow(rc[0], 2)*rc[2] 
                        //                 - 3*std::pow(rc[1], 2)*rc[2];
                        blm_der2xyz[0] = 3*rc[0]*rc[0]*rc[2] 
                                         - 3*rc[1]*rc[1]*rc[2];
                        blm_der2xyz[1] = -6*rc[0]*rc[1]*rc[2];
                        //blm_der2xyz[2] = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0];
                        blm_der2xyz[2] = (rc[0]*rc[0] - 3*rc[1]*rc[1]) * rc[0];
                        break;
                    case 6:
                        blm_der2xyz[0] = 6*rc[0]*rc[1]*rc[2];
                        //blm_der2xyz[1] = 3*std::pow(rc[0], 2)*rc[2] 
                        //                 - 3*std::pow(rc[1], 2)*rc[2];
                        blm_der2xyz[1] = 3*rc[0]*rc[0]*rc[2] 
                                         - 3*rc[1]*rc[1]*rc[2];
                        //blm_der2xyz[2] = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[1];
                        blm_der2xyz[2] = (3*rc[0]*rc[0] - rc[1]*rc[1]) * rc[1];
                        break;
                    case 7:
                        //blm_der2xyz[0] = 4*std::pow(rc[0], 3) - 12*rc[0]*std::pow(rc[1], 2);
                        blm_der2xyz[0] = 4*rc[0]*rc[0]*rc[0] - 12*rc[0]*rc[1]*rc[1];
                        //blm_der2xyz[1] = 4*std::pow(rc[1], 3) - 12*std::pow(rc[0], 2)*rc[1];
                        blm_der2xyz[1] = 4*rc[1]*rc[1]*rc[1] - 12*rc[0]*rc[0]*rc[1];
                        blm_der2xyz[2] = 0.0;
                        break;
                    case 8:
                        //blm_der2xyz[0] = 12*std::pow(rc[0], 2)*rc[1] 
                        //                 - 4*std::pow(rc[1], 3);
                        blm_der2xyz[0] = 12*rc[0]*rc[0]*rc[1] 
                                         - 4*rc[1]*rc[1]*rc[1];
                        //blm_der2xyz[1] = 4*std::pow(rc[0], 3) 
                        //                 - 12*rc[0]*std::pow(rc[1], 2);
                        blm_der2xyz[1] = 4*rc[0]*rc[0]*rc[0]
                                         - 12*rc[0]*rc[1]*rc[1];
                        blm_der2xyz[2] = 0.0;
                        break;
                }
                break;
        }
    }
};  // class : Blm


template <typename CoordType>
class TanhActivationFunc
{
public:
    static __host__ __device__ __forceinline__
    void find_val(CoordType &val, CoordType hidden_val);

    static __host__ __device__ __forceinline__
    void find_der(CoordType &der, CoordType hidden_val);

    static __host__ __device__ __forceinline__
    void find_der2der(CoordType &der2der, CoordType hidden_val);
};  // class : TanhActivationFunc

template <typename CoordType>
__host__ __device__ __forceinline__
void TanhActivationFunc<CoordType>::find_val(CoordType &val, CoordType hidden_val)
{
    val = (std::exp(hidden_val) - std::exp(-hidden_val))
          / (std::exp(hidden_val) + std::exp(-hidden_val));
}

template <typename CoordType>
__host__ __device__ __forceinline__
void TanhActivationFunc<CoordType>::find_der(CoordType &der, CoordType hidden_val)
{
    CoordType val = (std::exp(hidden_val) - std::exp(-hidden_val))
                    / (std::exp(hidden_val) + std::exp(-hidden_val));
    der = 1 - std::pow(val, 2);
}

template <typename CoordType>
__host__ __device__ __forceinline__
void TanhActivationFunc<CoordType>::find_der2der(CoordType &der2der, CoordType hidden_val)
{
    CoordType val = (std::exp(hidden_val) - std::exp(-hidden_val))
                    / (std::exp(hidden_val) + std::exp(-hidden_val));
    der2der = -2*val + 2*std::pow(val, 3);
}


};  // namespace : nep
};  // namespace : ai2pot

#endif
