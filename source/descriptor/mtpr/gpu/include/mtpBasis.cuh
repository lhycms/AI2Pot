#ifndef AI2POT_MTP_BASIS_CUH
#define AI2POT_MTP_BASIS_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include "./mtp_utilities.cuh"


template <typename CoordType>
__host__ __device__ 
void find_switch_func(CoordType &val,
                      CoordType &der2r,
                      CoordType rmax,
                      CoordType rmin,
                      CoordType distance_ij);


template <typename CoordType>
__host__ __device__ 
void find_rb_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType rmin,
                       CoordType distance);


template <typename CoordType>
__host__ __device__ 
void find_rq_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType rmin,
                       CoordType distance_ij);


template <typename CoordType>
__global__ 
void find_mtp_basis_val_der_cuda_kernel(
    CoordType *mtp_basis_val,
    CoordType (*mtp_basis_der)[3],
    CoordType *mtp_basis_der2coeffs,
    int chebyshev_size,
    CoordType *coeffs,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    const int max_num_mus4mom,
    const int *num_mus4moms,
    const int *mus4moms_ptr,
    int nmus,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*relative_coords)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    double rmax,
    double rmin);



template <typename CoordType>
__host__ __device__ 
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
__host__ __device__ 
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
__host__ __device__ 
void find_rq_chebyshev(CoordType *vals,
                       CoordType *ders2r,
                       int chebyshev_size,
                       CoordType rmax,
                       CoordType rmin,
                       CoordType distance_ij)
{
    CoordType rb_chebyshev_vals[MAX_CHEBYSHEV_SIZE];
    CoordType rb_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE];
    CoordType switch_func_val;
    CoordType switch_func_der2r;
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


template <typename CoordType>
__global__ void find_mtp_basis_val_der_cuda_kernel(
    CoordType *mtp_basis_val,
    CoordType (*mtp_basis_der)[3],
    CoordType *mtp_basis_der2coeffs,
    int chebyshev_size,
    CoordType *coeffs,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    const int max_num_mus4mom,
    const int *num_mus4moms,
    const int *mus4moms_ptr,
    int nmus,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*relative_coords)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    double rmax,
    double rmin)
{
    CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.};
    CoordType mom_ders[MAX_ALPHA_MOMENTS_COUNT][3] = {0.};
    CoordType mom_ders2coeffs[MAX_ALPHA_MOMENTS_COUNT][ntypes][ntypes][nmus][MAX_CHEBYSHEV_SIZE] = {0.};
    
}

//template <typename CoordType>
//void find_mtp_basis_val_der_cuda_launcher()
//{
//}


#endif