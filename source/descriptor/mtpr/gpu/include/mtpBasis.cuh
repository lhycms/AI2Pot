#ifndef AI2POT_MTP_BASIS_CUH
#define AI2POT_MTP_BASIS_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "./mtp_utilities.cuh"


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
    for (int ii=0; ii<inum; ii++) {
        memset(mom_vals, 0, MAX_ALPHA_MOMENTS_COUNT * sizeof(CoordType));
        memset(mom_ders, 0, MAX_ALPHA_MOMENTS_COUNT * MAX_NNEI * 3 * sizeof(CoordType));
        memset(mom_ders2coeffs, 0, MAX_ALPHA_MOMENTS_COUNT * MAX_NUM_TYPES * MAX_NUM_TYPES * MAX_NUM_MUS * MAX_CHEBYSHEV_SIZE * sizeof(CoordType));
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

            for (int i=0; i<alpha_index_times_count; i++) {
                CoordType val0 = mom_vals[alpha_index_times[i][0]];
                CoordType val1 = mom_vals[alpha_index_times[i][1]];
                CoordType val2 = alpha_index_times[i][2];

                mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
                
            }
        }
    }
}

//template <typename CoordType>
//void find_mtp_basis_val_der_cuda_launcher()
//{
//}


#endif