#ifndef AI2POT_MTP_BASIS_CUH
#define AI2POT_MTP_BASIS_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "./mtp_utilities.cuh"


namespace ai2pot {
namespace mtpr {

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
void find_mtp_basis_val_der_cuda_launcher(
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
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nx < inum)
    {
        const int ii = nx;
        // Step 1. 
        CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.};
        CoordType mom_ders[MAX_ALPHA_MOMENTS_COUNT][MAX_NNEI][3] = {0.};
        CoordType mom_ders2coeffs[MAX_ALPHA_MOMENTS_COUNT][MAX_NUM_TYPES][MAX_NUM_TYPES][MAX_NUM_MUS][MAX_CHEBYSHEV_SIZE] = {0.};
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE];
        CoordType rq_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE];

        int max_alpha_index_basic = 0;
        for (int ii=0; ii<alpha_index_basic_count; ii++) {
            int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][3];
            if (now_alpha_index_basic > max_alpha_index_basic)
                max_alpha_index_basic = now_alpha_index_basic;
        }
        max_alpha_index_basic++;

        CoordType auto_dist_powers_[MAX_ALPHA_INDEX_BASIC];
        CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3];
        CoordType NeighVect[3];
        CoordType distance_ij;
        int type_central;
        int type_outer;
        int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

        // Step 2.
        type_central = types[ilist[ii]];
        for (int jj=0; jj<numneigh[ii]; ii++) {
            type_outer = types[firstneigh[ii*umax_num_neigh_atoms + jj]];
            for (int a=0; a<3; a++)
                NeighVect[a] = relative_coords[ii*umax_num_neigh_atoms + jj][a];
            distance_ij = std::sqrt( std::pow(NeighVect[0], 2) +
                                        std::pow(NeighVect[1], 2) +
                                        std::pow(NeighVect[2], 2) );
            
            auto_dist_powers_[0] = 1;
            for (int a=0; a<3; a++)
                auto_coords_powers_[0][a] = 1;
            for (int k=0; k<max_alpha_index_basic; k++) {
                auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
                for (int a=0; a<3; a++)
                    auto_coords_powers_[k][a] = auto_coords_powers_[k-1][a] * NeighVect[a];
            }

            find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                            rq_chebyshev_ders2r,
                                            chebyshev_size,
                                            rmax,
                                            rmin,
                                            distance_ij);

            for (int i=0; i<alpha_index_basic_count; i++)
            {
                int mu = alpha_index_basic[i][0];
                int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
                CoordType powk = 1 / auto_dist_powers_[k];
                CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]];
                CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]];
                CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]];
                CoordType mult0 = pow0 * pow1 * pow2;

                for (int xi=0; xi<chebyshev_size; xi++) {
                    int idx = type_central*ntypes*nmus*chebyshev_size
                                + type_outer*nmus*chebyshev_size
                                + mu*chebyshev_size
                                + xi;
                    mom_vals[i] += coeffs[idx] * rq_chebyshev_vals[xi] * powk * mult0;
                    mom_ders2coeffs[i][type_central][type_outer][mu][xi] += rq_chebyshev_vals[xi] * powk * mult0;
                    mom_ders[i][jj][0] += NeighVect[0] / distance_ij * coeffs[idx] * mult0 *
                                            (rq_chebyshev_ders2r[xi] * powk
                                            - rq_chebyshev_vals[xi] * k * powk / distance_ij);
                    mom_ders[i][jj][1] += NeighVect[1] / distance_ij * coeffs[idx] * mult0 *
                                            (rq_chebyshev_ders2r[xi] * powk
                                            - rq_chebyshev_vals[xi] * k * powk / distance_ij);
                    mom_ders[i][jj][2] += NeighVect[2] / distance_ij * coeffs[idx] * mult0 *
                                            (rq_chebyshev_ders2r[xi] * powk
                                            - rq_chebyshev_vals[xi] * k * powk / distance_ij);
                    if (alpha_index_basic[i][1] != 0) {
                        mom_ders[i][jj][0] += coeffs[idx] * rq_chebyshev_vals[xi] * powk * alpha_index_basic[i][1]
                            * auto_coords_powers_[alpha_index_basic[i][1]-1][0]
                            * pow1
                            * pow2;
                    }
                    if (alpha_index_basic[i][2] != 0) {
                        mom_ders[i][jj][1] += coeffs[idx] * rq_chebyshev_vals[xi] * powk * alpha_index_basic[i][2]
                            * pow0
                            * auto_coords_powers_[alpha_index_basic[i][2]-1][1]
                            * pow2;
                    }
                    if (alpha_index_basic[i][3] != 0) {
                        mom_ders[i][jj][2] += coeffs[idx] * rq_chebyshev_vals[xi] * powk * alpha_index_basic[i][3]
                            * pow0
                            * pow1
                            * auto_coords_powers_[alpha_index_basic[i][3]-1][2];
                    }
                }
            }
        }

        for (int i=0; i<alpha_index_times_count; i++) 
        {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];
            mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
            
            for (int tmp_type_centeral=0; tmp_type_centeral<ntypes; tmp_type_centeral++) {
                for (int tmp_type_outer=0; tmp_type_outer<ntypes; tmp_type_outer++) {
                    for (int q=0; q<num_mus4moms[alpha_index_times[i][0]]; q++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            mom_ders2coeffs[alpha_index_times[i][3]][tmp_type_centeral][tmp_type_outer][mus4moms_ptr[alpha_index_times[i][0]*max_num_mus4mom + q]][xi] += val2
                                * mom_ders2coeffs[alpha_index_times[i][0]][tmp_type_centeral][tmp_type_outer][mus4moms_ptr[alpha_index_times[i][0]*max_num_mus4mom + q]][xi]
                                * val1;
                        }
                    }
                    for (int q=0; q<num_mus4moms[alpha_index_times[i][1]]; q++) {
                        for (int xi=0; xi<chebyshev_size; xi++) {
                            mom_ders2coeffs[alpha_index_times[i][3]][tmp_type_centeral][tmp_type_outer][mus4moms_ptr[alpha_index_times[i][1]*max_num_mus4mom + q]][xi] += val2
                                * val0
                                * mom_ders2coeffs[alpha_index_times[i][1]][tmp_type_centeral][tmp_type_outer][mus4moms_ptr[alpha_index_times[i][1]*max_num_mus4mom + q]][xi];
                        }
                    }
                }
            }

            for (int jj=0; jj<numneigh[ii]; jj++) 
            {
                mom_ders[alpha_index_times[i][3]][jj][0] += val2 * 
                    ( mom_ders[alpha_index_times[i][0]][jj][0] * val1
                      + val0 * mom_ders[alpha_index_times[i][1]][jj][0] );
                mom_ders[alpha_index_times[i][3]][jj][1] += val2 *
                    ( mom_ders[alpha_index_times[i][0]][jj][1] * val1
                      + val0 * mom_ders[alpha_index_times[i][1]][jj][1]);
                mom_ders[alpha_index_times[i][3]][jj][2] += val2 *
                    ( mom_ders[alpha_index_times[i][0]][jj][2] * val1
                      + val0 * mom_ders[alpha_index_times[i][1]][jj][2]);
            }
        }

        for (int i=0; i<alpha_scalar_moments; i++)
        {
            mtp_basis_val[ii*alpha_moments_count + i] = mom_vals[alpha_moment_mapping[i]];
            for (int jj=0; jj<numneigh; jj++)
                for (int a=0; a<3; a++)
                    mtp_basis_der[ii*alpha_moments_count*umax_num_neigh_atoms
                                  + alpha_moment_mapping[i]*umax_num_neigh_atoms
                                  + jj][a] = mom_ders[alpha_moment_mapping[i]][jj][a];
            for (int idx=0; idx<num_coeffs; idx++)
                mtp_basis_der2coeffs[ii*alpha_scalar_moments*num_coeffs + i*num_coeffs + idx] = mom_ders2coeffs[alpha_moment_mapping[i]*num_coeffs + idx];
        }
    }
}


template <typename CoordType>
void find_mtp_basis_val_der_cuda_launcher(CoordType *mtp_basis_val,
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
    const int block_size_x = 256;
    const int grid_size_x = (inum - 1) / block_size_x + 1;
    find_mtp_basis_val_der_cuda_kernel<CoordType> KERNEL_ARG2(grid_size_x, block_size_x) (
        mtp_basis_val,
        mtp_basis_der,
        mtp_basis_der2coeffs,
        chebyshev_size,
        coeffs,
        alpha_moments_count,
        alpha_index_basic_count,
        alpha_index_basic,
        alpha_index_times_count,
        alpha_index_times,
        alpha_scalar_moments,
        alpha_moment_mapping,
        max_num_mus4mom,
        num_mus4moms,
        mus4moms_ptr,
        nmus,
        inum,
        ilist,
        numneigh,
        firstneigh,
        relative_coords,
        types,
        ntypes,
        umax_num_neigh_atoms,
        rmax,
        rmin);
    CHECK( cudaDeviceSynchronize() );
}


};  // namespace : mtpr
};  // namespace : ai2pot



#endif