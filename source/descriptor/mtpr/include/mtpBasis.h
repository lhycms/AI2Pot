#ifndef AI2POT_MTPR_MTP_BASIS_H
#define AI2POT_MTPR_MTP_BASIS_H

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "./basis.h"
#include "./mtpParam.h"


#ifdef USE_CUDA
extern void find_mtp_basis_val_der_cuda_launcher<float>();
extern void find_mtp_basis_val_der_cuda_launcher<double>();
#endif

namespace ai2pot {
namespace mtpr {

template <typename CoordType>
class MtpBasis {
public:
    static void find_val_der(
        CoordType *mtp_basis_val,
        CoordType (*mtp_basis_der)[3],
        CoordType *mtp_basis_der2coeffs,
        bool calculate_der2xyz,
        bool calculate_der2coeffs,
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

    static void find_der_backward(
        CoordType *out_der,
        CoordType *out_der2coeffs,
        bool calculate_der2xyz,
        bool calculate_der2coeffs,
        CoordType *grad_output,
        CoordType *mtp_basis_der,
        CoordType *mtp_basis_der2coeffs,
        int alpha_scalar_moments,
        int inum,
        int umax_num_neighs,
        int num_coeffs);
};  // classs : MtpBasis


template <typename CoordType>
void MtpBasis<CoordType>::find_val_der(
    CoordType *mtp_basis_val,
    CoordType (*mtp_basis_der)[3],
    CoordType *mtp_basis_der2coeffs,
    bool calculate_der2xyz,
    bool calculate_der2coeffs,
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
    //memset(mtp_basis_val, 0, sizeof(CoordType) * inum * alpha_scalar_moments);
    //memset(mtp_basis_der, 0, sizeof(CoordType) * inum * alpha_scalar_moments * umax_num_neigh_atoms * 3);
    //memset(mtp_basis_der2coeffs, 0, sizeof(CoordType) * inum * alpha_scalar_moments * ntypes * ntypes * nmus * chebyshev_size);

    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    CoordType *mom_ders2coeffs;
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    if (calculate_der2xyz)
        mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);
    if (calculate_der2coeffs)
        mom_ders2coeffs = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count * ntypes * ntypes * nmus * chebyshev_size);
    //memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
    //memset(mom_ders, 0, sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);
    //memset(mom_ders2coeffs, 0, sizeof(CoordType) * alpha_moments_count * ntypes * ntypes * nmus * chebyshev_size);

    int max_alpha_index_basic = 0;
    for (int ii=0; ii<alpha_index_basic_count; ii++) {
        int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][3];
        if (now_alpha_index_basic > max_alpha_index_basic) 
            max_alpha_index_basic = now_alpha_index_basic;
    }
    max_alpha_index_basic++;

    CoordType *auto_dist_powers_;
    CoordType (*auto_coords_powers_)[3];
    auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic);
    auto_coords_powers_ = (CoordType (*)[3])malloc(sizeof(CoordType) * max_alpha_index_basic * 3);
    //memset(auto_dist_powers_, 0, sizeof(CoordType) * max_alpha_index_basic);
    //memset(auto_coords_powers_, 0, sizeof(CoordType) * max_alpha_index_basic * 3);
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    int type_central;
    int type_outer;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    RQ_Chebyshev<CoordType>* p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);

    // Step 2.
    for (int ii=0; ii<inum; ii++)
    {
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        if (calculate_der2xyz)
            memset(mom_ders, 0, sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);
        if (calculate_der2coeffs)
            memset(mom_ders2coeffs, 0, sizeof(CoordType) * alpha_moments_count * num_coeffs);

        type_central = types[ilist[ii]];
        if (type_central >= ntypes)
            throw MtpException("Too few types in the MTP potential.");

        for (int jj=0; jj<numneigh[ii]; jj++) 
        {
            type_outer = types[firstneigh[ii*umax_num_neigh_atoms + jj]];
            if (type_outer >= ntypes)
                throw MtpException("Too few types in the MTP potential.");
            for (int a=0; a<3; a++)
                NeighbVect[a] = relative_coords[ii*umax_num_neigh_atoms + jj][a];
            distance_ij = std::sqrt( std::pow(NeighbVect[0], 2) + 
                                     std::pow(NeighbVect[1], 2) + 
                                     std::pow(NeighbVect[2], 2) );
            if (distance_ij > rmax)
                continue;
            distance_ij_inv = 1 / distance_ij;
            p_RadialBasis->build(distance_ij);

            auto_dist_powers_[0] = 1;
            for (int a=0; a<3; a++)
                auto_coords_powers_[0][a] = 1;
            for (int k=1; k<max_alpha_index_basic; k++) {
                auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
                for (int a=0; a<3; a++)
                    auto_coords_powers_[k][a] = auto_coords_powers_[k-1][a] * NeighbVect[a];
            }

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
                    int idx = type_central*ntypes*nmus*chebyshev_size + type_outer*nmus*chebyshev_size + mu*chebyshev_size + xi;
                    mom_vals[i] += coeffs[idx] * p_RadialBasis->vals()[xi] * powk * mult0;
                    if (calculate_der2coeffs)
                        mom_ders2coeffs[i*num_coeffs + idx] += p_RadialBasis->vals()[xi] * powk * mult0;
                    
                    CoordType shuffle_mom_ders_part = distance_ij_inv *
                                    ( coeffs[idx] * p_RadialBasis->ders2r()[xi] * powk * mult0
                                    - coeffs[idx] * p_RadialBasis->vals()[xi] * k * powk * distance_ij_inv * mult0 );
                    if (calculate_der2xyz) {
                        mom_ders[i*umax_num_neigh_atoms + jj][0] += NeighbVect[0] * shuffle_mom_ders_part;
                        mom_ders[i*umax_num_neigh_atoms + jj][1] += NeighbVect[1] * shuffle_mom_ders_part;
                        mom_ders[i*umax_num_neigh_atoms + jj][2] += NeighbVect[2] * shuffle_mom_ders_part;

                        if (alpha_index_basic[i][1] != 0) {
                            mom_ders[i*umax_num_neigh_atoms + jj][0] += coeffs[idx] * p_RadialBasis->vals()[xi] * powk * alpha_index_basic[i][1]
                                            * auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
                                            * pow1
                                            * pow2;
                        }
                        if (alpha_index_basic[i][2] != 0) {
                            mom_ders[i*umax_num_neigh_atoms + jj][1] += coeffs[idx] * p_RadialBasis->vals()[xi] * powk * alpha_index_basic[i][2]
                                            * pow0
                                            * auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
                                            * pow2;
                        }
                        if (alpha_index_basic[i][3] != 0) {
                            mom_ders[i*umax_num_neigh_atoms + jj][2] += coeffs[idx] * p_RadialBasis->vals()[xi] * powk * alpha_index_basic[i][3]
                                            * pow0
                                            * pow1
                                            * auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
                        }
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

            if (calculate_der2coeffs) {
                for (int tmp_type_central=0; tmp_type_central<ntypes; tmp_type_central++) {
                    for (int tmp_type_outer=0; tmp_type_outer<ntypes; tmp_type_outer++) {
                        for (int q=0; q<num_mus4moms[alpha_index_times[i][0]]; q++) {
                            for (int xi=0; xi<chebyshev_size; xi++) {
                                int idx0 = (tmp_type_central*ntypes + tmp_type_outer)*nmus*chebyshev_size 
                                        + mus4moms_ptr[alpha_index_times[i][0]*max_num_mus4mom + q]*chebyshev_size
                                        + xi;

                                mom_ders2coeffs[alpha_index_times[i][3]*num_coeffs + idx0] += val2
                                    * mom_ders2coeffs[alpha_index_times[i][0]*num_coeffs + idx0]
                                    * val1;
                            }
                        }
                        for (int q=0; q<num_mus4moms[alpha_index_times[i][1]]; q++) {
                            for (int xi=0; xi<chebyshev_size; xi++) {
                                int idx1 = (tmp_type_central*ntypes + tmp_type_outer)*nmus*chebyshev_size
                                        + mus4moms_ptr[alpha_index_times[i][1]*max_num_mus4mom + q]*chebyshev_size
                                        + xi;
                                mom_ders2coeffs[alpha_index_times[i][3]*num_coeffs + idx1] += val2
                                    * val0
                                    * mom_ders2coeffs[alpha_index_times[i][1]*num_coeffs + idx1];
                            }
                        }
                    }
                }
            }
            
            if (calculate_der2xyz) {
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
        }


        for (int i=0; i<alpha_scalar_moments; i++)
        {
            mtp_basis_val[ii*alpha_scalar_moments + i] = mom_vals[alpha_moment_mapping[i]];

            if (calculate_der2xyz){
                for (int jj=0; jj<numneigh[ii]; jj++)
                    for (int a=0; a<3; a++)
                        mtp_basis_der[ii*alpha_scalar_moments*umax_num_neigh_atoms + i*umax_num_neigh_atoms + jj][a] = mom_ders[alpha_moment_mapping[i]*umax_num_neigh_atoms + jj][a];
            }
            if (calculate_der2coeffs) {
                for (int idx=0; idx<num_coeffs; idx++)
                    mtp_basis_der2coeffs[ii*alpha_scalar_moments*num_coeffs + i*num_coeffs + idx] = mom_ders2coeffs[alpha_moment_mapping[i]*num_coeffs + idx];
            }
        }
    }
    
    // Step . Free memory
    free(mom_vals);
    if (calculate_der2xyz)
        free(mom_ders);
    if (calculate_der2coeffs)
        free(mom_ders2coeffs);
    free(auto_dist_powers_);
    free(auto_coords_powers_);
    delete p_RadialBasis;
}


template <typename CoordType>
void MtpBasis<CoordType>::find_der_backward(
        CoordType *out_der,
        CoordType *out_der2coeffs,
        bool calculate_der2xyz,
        bool calculate_der2coeffs,
        CoordType *grad_output,
        CoordType *mtp_basis_der,
        CoordType *mtp_basis_der2coeffs,
        int alpha_scalar_moments,
        int inum,
        int umax_num_neighs,
        int num_coeffs)
{
    if ( (!calculate_der2xyz) && (!calculate_der2coeffs) )
        return;
    for (int ii=0; ii<inum; ii++) {
        for (int ee=0; ee<alpha_scalar_moments; ee++) {
            if (calculate_der2xyz) {
                for (int jj=0; jj<umax_num_neighs; jj++) {
                    out_der[ii*umax_num_neighs*3 + jj*3 + 0] += grad_output[ii*alpha_scalar_moments + ee]
                        * mtp_basis_der[ii*alpha_scalar_moments*umax_num_neighs*3 + ee*umax_num_neighs*3 + jj*3 + 0];
                    out_der[ii*umax_num_neighs*3 + jj*3 + 1] += grad_output[ii*alpha_scalar_moments + ee]
                        * mtp_basis_der[ii*alpha_scalar_moments*umax_num_neighs*3 + ee*umax_num_neighs*3 + jj*3 + 1];
                    out_der[ii*umax_num_neighs*3 + jj*3 + 2] += grad_output[ii*alpha_scalar_moments + ee]
                        * mtp_basis_der[ii*alpha_scalar_moments*umax_num_neighs*3 + ee*umax_num_neighs*3 + jj*3 + 2];
                }
            }

            if (calculate_der2coeffs) {
                for (int kk=0; kk<num_coeffs; kk++)
                    out_der2coeffs[kk] += grad_output[ii*alpha_scalar_moments + ee]
                        * mtp_basis_der2coeffs[ii*alpha_scalar_moments*num_coeffs + ee*num_coeffs + kk];
            }
        }
    }
}

};  // namespace : mtpr
};  // namespace : ai2pot

#endif
