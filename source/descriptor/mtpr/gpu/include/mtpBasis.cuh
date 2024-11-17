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
    CoordType rmax,
    CoordType rmin,
    CoordType *moms_vals,
    CoordType (*moms_ders)[3],
    CoordType *moms_ders2coeffs);


template <typename CoordType>
__host__
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
    CoordType rmax,
    CoordType rmin,
    CoordType *moms_vals,
    CoordType *moms_ders,
    CoordType *moms_ders2coeffs);


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
    CoordType rmax,
    CoordType rmin,
    CoordType *moms_vals,
    CoordType (*moms_ders)[3],
    CoordType *moms_ders2coeffs)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (nx < inum)
    {
        const int ii = nx;
        // Step 1. 
        CoordType *mom_vals = &moms_vals[ii*alpha_moments_count];
        CoordType (*mom_ders)[3] = (CoordType (*)[3])&moms_ders[ii*alpha_moments_count*umax_num_neigh_atoms];
        CoordType *mom_ders2coeffs = &moms_ders2coeffs[ii*alpha_moments_count*ntypes*ntypes*nmus*chebyshev_size];
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);
        memset(mom_ders2coeffs, 0, sizeof(CoordType) * alpha_moments_count * ntypes * ntypes * nmus * chebyshev_size);
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE];
        CoordType rq_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE];

        int type_central;
        int type_outer;
        CoordType NeighVect[3];
        CoordType distance_ij;
        CoordType auto_dist_powers_[MAX_ALPHA_INDEX_BASIC];
        CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3];
        int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

        // Step 2.
        type_central = types[ilist[ii]];
        for (int jj=0; jj<numneigh[ii]; jj++) {
            type_outer = types[firstneigh[ii*umax_num_neigh_atoms + jj]];
            for (int a=0; a<3; a++)
                NeighVect[a] = relative_coords[ii*umax_num_neigh_atoms + jj][a];
            distance_ij = std::sqrt( std::pow(NeighVect[0], 2)
                                     + std::pow(NeighVect[1], 2)
                                     + std::pow(NeighVect[2], 2) );
            
            auto_dist_powers_[0] = 1;
            for (int a=0; a<3; a++)
                auto_coords_powers_[0][a] = 1;
            for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
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
                CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
                CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
                CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
                CoordType mult0 = pow0 * pow1 * pow2;
                
                for (int xi=0; xi<chebyshev_size; xi++) {
                    int idx = type_central*ntypes*nmus*chebyshev_size
                              + type_outer*nmus*chebyshev_size
                              + mu*chebyshev_size
                              + xi;
                    mom_vals[i] += coeffs[idx] * rq_chebyshev_vals[xi] * powk * mult0;
                    mom_ders2coeffs[i*num_coeffs + idx] += rq_chebyshev_vals[xi] * powk * mult0;
                    mom_ders[i*umax_num_neigh_atoms + jj][0] += NeighVect[0] / distance_ij * coeffs[idx] * mult0 *
                                                                (rq_chebyshev_ders2r[xi] * powk
                                                                 - rq_chebyshev_vals[xi] * k * powk / distance_ij);
                    mom_ders[i*umax_num_neigh_atoms + jj][1] += NeighVect[1] / distance_ij * coeffs[idx] * mult0 *
                                                                (rq_chebyshev_ders2r[xi] * powk
                                                                 - rq_chebyshev_vals[xi] * k * powk / distance_ij);
                    mom_ders[i*umax_num_neigh_atoms + jj][2] += NeighVect[2] / distance_ij * coeffs[idx] * mult0 *
                                                                (rq_chebyshev_ders2r[xi] * powk
                                                                 - rq_chebyshev_vals[xi] * k * powk / distance_ij);
                    
                    if (alpha_index_basic[i][1] != 0) {
                        mom_ders[i*umax_num_neigh_atoms + jj][0] += coeffs[idx] * rq_chebyshev_vals[xi] * powk * alpha_index_basic[i][1]
                            * auto_coords_powers_[alpha_index_basic[i][1]-1][0]
                            * pow1
                            * pow2;
                    }
                    if (alpha_index_basic[i][2] != 0) {
                        mom_ders[i*umax_num_neigh_atoms + jj][1] += coeffs[idx] * rq_chebyshev_vals[xi] * powk * alpha_index_basic[i][2]
                            * pow0
                            * auto_coords_powers_[alpha_index_basic[i][2]-1][1]
                            * pow2;
                    }
                    if (alpha_index_basic[i][3] != 0) {
                        mom_ders[i*umax_num_neigh_atoms + jj][2] += coeffs[idx] * rq_chebyshev_vals[xi] * powk * alpha_index_basic[i][3]
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
            
            for (int tmp_type_outer=0; tmp_type_outer<ntypes; tmp_type_outer++) {
                for (int q=0; q<num_mus4moms[alpha_index_times[i][0]]; q++) {
                    const int mu = mus4moms_ptr[alpha_index_times[i][0]*max_num_mus4mom + q];
                    for (int xi=0; xi<chebyshev_size; xi++) {
                        const int idx = (type_central*ntypes + tmp_type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                        mom_ders2coeffs[alpha_index_times[i][3]*num_coeffs + idx] += val2
                            * mom_ders2coeffs[alpha_index_times[i][0]*num_coeffs + idx]
                            * val1;
                    }
                }
                for (int q=0; q<num_mus4moms[alpha_index_times[i][1]]; q++) {
                    const int mu = mus4moms_ptr[alpha_index_times[i][1]*max_num_mus4mom + q];
                    for (int xi=0; xi<chebyshev_size; xi++) {
                        const int idx = (type_central*ntypes + tmp_type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                        mom_ders2coeffs[alpha_index_times[i][3]*num_coeffs + idx] += val2
                            * val0
                            * mom_ders2coeffs[alpha_index_times[i][1]*num_coeffs + idx];
                    }
                }
            }
            

            for (int jj=0; jj<numneigh[ii]; jj++) 
            {
                mom_ders[alpha_index_times[i][3]*umax_num_neigh_atoms + jj][0] += val2 * 
                    ( mom_ders[alpha_index_times[i][0]*umax_num_neigh_atoms + jj][0] * val1
                      + val0 * mom_ders[alpha_index_times[i][1]*umax_num_neigh_atoms + jj][0] );
                mom_ders[alpha_index_times[i][3]*umax_num_neigh_atoms + jj][1] += val2 *
                    ( mom_ders[alpha_index_times[i][0]*umax_num_neigh_atoms + jj][1] * val1
                      + val0 * mom_ders[alpha_index_times[i][1]*umax_num_neigh_atoms + jj][1] );
                mom_ders[alpha_index_times[i][3]*umax_num_neigh_atoms + jj][2] += val2 *
                    ( mom_ders[alpha_index_times[i][0]*umax_num_neigh_atoms + jj][2] * val1
                      + val0 * mom_ders[alpha_index_times[i][1]*umax_num_neigh_atoms + jj][2] );
            }
        }

        for (int i=0; i<alpha_scalar_moments; i++)
        {
            mtp_basis_val[ii*alpha_scalar_moments + i] = mom_vals[alpha_moment_mapping[i]];
            for (int jj=0; jj<numneigh[ii]; jj++)
                for (int a=0; a<3; a++)
                    mtp_basis_der[ii*alpha_scalar_moments*umax_num_neigh_atoms
                                  + i*umax_num_neigh_atoms
                                  + jj][a] = mom_ders[alpha_moment_mapping[i]*umax_num_neigh_atoms + jj][a];
            
            for (int idx=0; idx<num_coeffs; idx++)
                mtp_basis_der2coeffs[ii*alpha_scalar_moments*num_coeffs + i*num_coeffs + idx] = mom_ders2coeffs[alpha_moment_mapping[i]*num_coeffs + idx];
        }

    }
}


template <typename CoordType>
__host__
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
    CoordType rmax,
    CoordType rmin,
    CoordType *moms_vals,
    CoordType *moms_ders,
    CoordType *moms_ders2coeffs)
{
    const int block_size_x = 256;
    const int grid_size_x = (inum - 1) / block_size_x + 1;

    double *d_mtp_basis_val;
    double *d_mtp_basis_der;
    double *d_mtp_basis_der2coeffs;
    double *d_coeffs;
    int *d_alpha_index_basic;
    int *d_alpha_index_times;
    int *d_alpha_moment_mapping;
    int *d_num_mus4moms;
    int *d_mus4moms_ptr;
    int *d_ilist;
    int *d_numneigh;
    int *d_firstneigh;
    CoordType *d_relative_coords;
    int *d_types;
    CoordType *d_moms_vals;
    CoordType *d_moms_ders;
    CoordType *d_moms_ders2coeffs;
    CHECK( cudaMalloc((void**)&d_mtp_basis_val, sizeof(CoordType) * inum * alpha_scalar_moments) );
    CHECK( cudaMalloc((void**)&d_mtp_basis_der, sizeof(CoordType) * inum * alpha_scalar_moments * umax_num_neigh_atoms * 3) );
    CHECK( cudaMalloc((void**)&d_mtp_basis_der2coeffs, sizeof(CoordType) * inum * alpha_scalar_moments * ntypes * ntypes * nmus * chebyshev_size) );
    CHECK( cudaMalloc((void**)&d_coeffs, sizeof(CoordType) * ntypes * ntypes * nmus * chebyshev_size) );
    CHECK( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4) );
    CHECK( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * alpha_index_times_count * 4) );
    CHECK( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * alpha_scalar_moments) );
    CHECK( cudaMalloc((void**)&d_num_mus4moms, sizeof(int) * alpha_moments_count) );
    CHECK( cudaMalloc((void**)&d_mus4moms_ptr, sizeof(int) * alpha_moments_count* max_num_mus4mom) );
    CHECK( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
    CHECK( cudaMalloc((void**)&d_numneigh, sizeof(int) * inum) );
    CHECK( cudaMalloc((void**)&d_firstneigh, sizeof(int) * inum * umax_num_neigh_atoms) );
    CHECK( cudaMalloc((void**)&d_relative_coords, sizeof(CoordType) * inum * umax_num_neigh_atoms * 3) );
    CHECK( cudaMalloc((void**)&d_types, sizeof(int) * inum) );
    CHECK( cudaMalloc((void**)&d_moms_vals, sizeof(CoordType) * inum * alpha_moments_count) );
    CHECK( cudaMalloc((void**)&d_moms_ders, sizeof(CoordType) * inum * alpha_moments_count * umax_num_neigh_atoms * 3) );
    CHECK( cudaMalloc((void**)&d_moms_ders2coeffs, sizeof(CoordType) * inum * alpha_moments_count * ntypes * ntypes * nmus * chebyshev_size) );

    CHECK( cudaMemcpy(d_mtp_basis_val, mtp_basis_val, sizeof(CoordType) * inum * alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_mtp_basis_der, (CoordType*)mtp_basis_der, sizeof(CoordType) * inum * alpha_scalar_moments * umax_num_neigh_atoms * 3, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_mtp_basis_der2coeffs, mtp_basis_der2coeffs, sizeof(CoordType) * inum * alpha_scalar_moments * ntypes * ntypes * nmus * chebyshev_size, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_coeffs, coeffs, sizeof(double) * ntypes * ntypes * nmus * chebyshev_size, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_alpha_index_basic, (CoordType*)alpha_index_basic, sizeof(int) * alpha_index_basic_count * 4, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_alpha_index_times, (CoordType*)alpha_index_times, sizeof(int) * alpha_index_times_count * 4, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_alpha_moment_mapping, alpha_moment_mapping, sizeof(int) * alpha_scalar_moments, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_num_mus4moms, num_mus4moms, sizeof(int) * alpha_moments_count, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_mus4moms_ptr, mus4moms_ptr, sizeof(int) * alpha_moments_count * max_num_mus4mom, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_ilist, ilist, sizeof(int) * inum, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_numneigh, numneigh, sizeof(int) * inum, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_firstneigh, firstneigh, sizeof(int) * inum * umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_relative_coords, relative_coords, sizeof(CoordType) * inum * umax_num_neigh_atoms * 3, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_types, types, sizeof(int) * inum, cudaMemcpyHostToDevice) );

    find_mtp_basis_val_der_cuda_kernel<CoordType> KERNEL_ARG2(grid_size_x, block_size_x) (
        d_mtp_basis_val,
        (CoordType (*)[3])d_mtp_basis_der,
        d_mtp_basis_der2coeffs,
        chebyshev_size,
        d_coeffs,
        alpha_moments_count,
        alpha_index_basic_count,
        (int (*)[4])d_alpha_index_basic,
        alpha_index_times_count,
        (int (*)[4])d_alpha_index_times,
        alpha_scalar_moments,
        d_alpha_moment_mapping,
        max_num_mus4mom,
        d_num_mus4moms,
        d_mus4moms_ptr,
        nmus,
        inum,
        d_ilist,
        d_numneigh,
        d_firstneigh,
        (CoordType (*)[3])d_relative_coords,
        d_types,
        ntypes,
        umax_num_neigh_atoms,
        rmax,
        rmin,
        d_moms_vals,
        d_moms_ders,
        d_moms_ders2coeffs);

    CHECK( cudaFree(d_mtp_basis_val) );
    CHECK( cudaFree(d_mtp_basis_der) );
    CHECK( cudaFree(d_mtp_basis_der2coeffs) );
    CHECK( cudaFree(d_coeffs) );
    CHECK( cudaFree(d_alpha_index_basic) );
    CHECK( cudaFree(d_alpha_index_times) );
    CHECK( cudaFree(d_alpha_moment_mapping) );
    CHECK( cudaFree(d_num_mus4moms) );
    CHECK( cudaFree(d_mus4moms_ptr) );
    CHECK( cudaFree(d_ilist) );
    CHECK( cudaFree(d_numneigh) );
    CHECK( cudaFree(d_firstneigh) );
    CHECK( cudaFree(d_relative_coords) );
    CHECK( cudaFree(d_types) );
    CHECK( cudaFree(d_moms_vals) );
    CHECK( cudaFree(d_moms_ders) );
    CHECK( cudaFree(d_moms_ders2coeffs) );

    CHECK( cudaPeekAtLastError() );
    CHECK( cudaDeviceSynchronize() );
    CHECK( cudaPeekAtLastError() );
}


};  // namespace : mtpr
};  // namespace : ai2pot


#endif
