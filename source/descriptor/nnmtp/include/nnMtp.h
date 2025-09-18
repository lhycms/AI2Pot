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

#ifndef AI2POT_NNMTP_NNMTP_H
#define AI2POT_NNMTP_NNMTP_H

#include <stdio.h>
#include <stdlib.h>
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
#include <omp.h>
#endif

#include "./basis.h"
#include "./mtpBasis.h"
#include "./nnMtpUtils.h"
#include "./nnMtpLoss.h"

namespace ai2pot {
namespace nnmtp {


template <typename CoordType>
class NNMtp {
public:
    static void find_ef(
        CoordType &etot,
        CoordType (*force)[3],
        int chebyshev_size,
        int num_neurons,
        CoordType *coeffs,
        CoordType *w0,
        CoordType *w1,
        CoordType *type_bias,
        const int alpha_moments_count,
        const int alpha_index_basic_count,
        const int (*alpha_index_basic)[4],
        const int alpha_index_times_count,
        const int (*alpha_index_times)[4],
        const int alpha_scalar_moments,
        const int *alpha_moment_mapping,
        int nmus,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        CoordType (*rcs)[3],
        int *types,
        int ntypes,
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin);

    static void find_efv(
        CoordType &etot,
        CoordType (*force)[3],
        CoordType *virial,
        int chebyshev_size,
        int num_neurons,
        CoordType *coeffs,
        CoordType *w0,
        CoordType *w1,
        CoordType *type_bias,
        const int alpha_moments_count,
        const int alpha_index_basic_count,
        const int (*alpha_index_basic)[4],
        const int alpha_index_times_count,
        const int (*alpha_index_times)[4],
        const int alpha_scalar_moments,
        const int *alpha_moment_mapping,
        int nmus,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        CoordType (*rcs)[3],
        int *types,
        int ntypes,
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin);


    static void find_descriptors(
        CoordType *all_descriptors,
        int chebyshev_size,
        int num_neurons,
        CoordType *coeffs,
        CoordType *w0,
        CoordType *w1,
        CoordType *type_bias,
        const int alpha_moments_count,
        const int alpha_index_basic_count,
        const int (*alpha_index_basic)[4],
        const int alpha_index_times_count,
        const int (*alpha_index_times)[4],
        const int alpha_scalar_moments,
        const int *alpha_moment_mapping,
        int nmus,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        CoordType (*relative_coords)[3],
        int *types,
        int ntypes,
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin);

    static void find_e_sites(
        CoordType *e_sites,
        int chebyshev_size,
        int num_neurons,
        CoordType *coeffs,
        CoordType *w0,
        CoordType *w1,
        CoordType *type_bias,
        const int alpha_moments_count,
        const int alpha_index_basic_count,
        const int (*alpha_index_basic)[4],
        const int alpha_index_times_count,
        const int (*alpha_index_times)[4],
        const int alpha_scalar_moments,
        const int *alpha_moment_mapping,
        int nmus,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        CoordType (*rcs)[3],
        int *types,
        int ntypes,
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin);
    
    static void find_e_sites_backward(
        CoordType *e_sites_der2coeffs,
        CoordType *e_sites_der2w0,
        CoordType *e_sites_der2w1,
        CoordType *e_sites_der2type_bias,
        int chebyshev_size,
        int num_neurons,
        CoordType *coeffs,
        CoordType *w0,
        CoordType *w1,
        CoordType *type_bias,
        const int alpha_moments_count,
        const int alpha_index_basic_count,
        const int (*alpha_index_basic)[4],
        const int alpha_index_times_count,
        const int (*alpha_index_times)[4],
        const int alpha_scalar_moments,
        const int *alpha_moment_mapping,
        int nmus,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        CoordType (*rcs)[3],
        int *types,
        int ntypes,
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin);
};  // class : NNMtp



template <typename CoordType>
void NNMtp<CoordType>::find_ef(
    CoordType &etot,
    CoordType (*force)[3],
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*rcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    CoordType *e_site_der2mom;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    // Step 2.
    etot = 0;
    memset(force, 0, sizeof(CoordType) * (inum+nghost) * 3);

#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
#pragma omp parallel private(mom_vals, mom_ders, e_site_der2mom)
{
#endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
    e_site_der2mom = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);

    int center_idx;
    int type_central;
    CoordType *type_central_w0;
    CoordType *type_central_w1;
    int neigh_idx;
    CoordType neigh_vec[3];
    CoordType distance_ij;

    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++)
    {
        center_idx = ilist[ii];
        type_central = types[center_idx];
        type_central_w0 = &w0[type_central*num_neurons*alpha_scalar_moments];
        type_central_w1 = &w1[type_central*num_neurons];
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
        memset(e_site_der2mom, 0, sizeof(CoordType) * alpha_moments_count);
        CoordType e_site = 0;

        MomsValDer<CoordType>::find_val_der(
            mom_vals,
            mom_ders,
            chebyshev_size,
            coeffs,
            alpha_moments_count,
            alpha_index_basic_count,
            alpha_index_basic,
            alpha_index_times_count,
            alpha_index_times,
            nmus,
            ilist[ii],
            numneigh[ii],
            &firstneigh[ii*umax_num_neigh_atoms],
            &rcs[ii*umax_num_neigh_atoms],
            types,
            ntypes,
            umax_num_neigh_atoms,
            rmax,
            rmin);

        // NNMtp Energy
        e_site = type_bias[type_central];
        for (int p=0; p<num_neurons; p++)
        {
            CoordType hidden_val = 0;
            CoordType activated_hidden_val = 0;
            for (int k=0; k<alpha_scalar_moments; k++) {
                //printf("** %.15lf, %.15lf\n", type_central_w0[p*alpha_scalar_moments + k], mom_vals[alpha_moment_mapping[k]]);
                hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
            }
            TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
            e_site += type_central_w1[p] * activated_hidden_val;
        }
        #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
        #pragma omp atomic
        #endif
        etot += e_site;

        
        // NNMtp Energy derivatives w.r.t. xyz
        for (int p=0; p<num_neurons; p++)
        {
            CoordType hidden_val = 0;
            CoordType activated_hidden_der = 0;
            for (int k=0; k<alpha_scalar_moments; k++)
                hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
            TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
            for (int k=0; k<alpha_scalar_moments; k++)
                e_site_der2mom[alpha_moment_mapping[k]] += type_central_w1[p]
                                                           * activated_hidden_der
                                                           * type_central_w0[p*alpha_scalar_moments + k];
        }

        for (int i=alpha_index_times_count-1; i>=0; i--) {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];

            e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2
                                                       * val0;
            e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2
                                                       * val1;
        }

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            neigh_vec[0] = rcs[ii*umax_num_neigh_atoms+jj][0];
            neigh_vec[1] = rcs[ii*umax_num_neigh_atoms+jj][1];
            neigh_vec[2] = rcs[ii*umax_num_neigh_atoms+jj][2];
            distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                     + std::pow(neigh_vec[1], 2)
                                     + std::pow(neigh_vec[2], 2) );
            if (distance_ij > rmax)
                continue;
            
            for (int aa=0; aa<3; aa++) {
                CoordType e_site_ders_ija = 0;
                // eders_ija
                for (int i=0; i<alpha_index_basic_count; i++)
                    e_site_ders_ija += e_site_der2mom[i] * mom_ders[i*umax_num_neigh_atoms + jj][aa];
                
                // Force
                #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                #pragma omp atomic
                #endif
                force[center_idx][aa] += e_site_ders_ija;
                #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                #pragma omp atomic
                #endif
                force[neigh_idx][aa] -= e_site_ders_ija;
            }
        }
    }

    // Step . Free
    free(mom_vals);
    free(mom_ders);
    free(e_site_der2mom);
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
}
#endif
}


template <typename CoordType>
void NNMtp<CoordType>::find_efv(
    CoordType &etot,
    CoordType (*force)[3],
    CoordType *virial,
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*rcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    CoordType *e_site_der2mom;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    // Step 2.
    etot = 0;
    memset(force, 0, sizeof(CoordType) * (inum+nghost) * 3);

#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
#pragma omp parallel private(mom_vals, mom_ders, e_site_der2mom)
{
#endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
    e_site_der2mom = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);

    int center_idx;
    int type_central;
    CoordType *type_central_w0;
    CoordType *type_central_w1;
    int neigh_idx;
    CoordType neigh_vec[3];
    CoordType distance_ij;

    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++)
    {
        center_idx = ilist[ii];
        type_central = types[center_idx];
        type_central_w0 = &w0[type_central*num_neurons*alpha_scalar_moments];
        type_central_w1 = &w1[type_central*num_neurons];
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
        memset(e_site_der2mom, 0, sizeof(CoordType) * alpha_moments_count);
        CoordType e_site = 0;

        MomsValDer<CoordType>::find_val_der(
            mom_vals,
            mom_ders,
            chebyshev_size,
            coeffs,
            alpha_moments_count,
            alpha_index_basic_count,
            alpha_index_basic,
            alpha_index_times_count,
            alpha_index_times,
            nmus,
            ilist[ii],
            numneigh[ii],
            &firstneigh[ii*umax_num_neigh_atoms],
            &rcs[ii*umax_num_neigh_atoms],
            types,
            ntypes,
            umax_num_neigh_atoms,
            rmax,
            rmin);

        // NNMtp Energy
        e_site = type_bias[type_central];
        for (int p=0; p<num_neurons; p++)
        {
            CoordType hidden_val = 0;
            CoordType activated_hidden_val = 0;
            for (int k=0; k<alpha_scalar_moments; k++) {
                //printf("** %.15lf, %.15lf\n", type_central_w0[p*alpha_scalar_moments + k], mom_vals[alpha_moment_mapping[k]]);
                hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
            }
            TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
            e_site += type_central_w1[p] * activated_hidden_val;
        }
        #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
        #pragma omp atomic
        #endif
        etot += e_site;

        
        // NNMtp Energy derivatives w.r.t. xyz
        for (int p=0; p<num_neurons; p++)
        {
            CoordType hidden_val = 0;
            CoordType activated_hidden_der = 0;
            for (int k=0; k<alpha_scalar_moments; k++)
                hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
            TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
            for (int k=0; k<alpha_scalar_moments; k++)
                e_site_der2mom[alpha_moment_mapping[k]] += type_central_w1[p]
                                                           * activated_hidden_der
                                                           * type_central_w0[p*alpha_scalar_moments + k];
        }

        for (int i=alpha_index_times_count-1; i>=0; i--) {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];

            e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2
                                                       * val0;
            e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2
                                                       * val1;
        }

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            neigh_vec[0] = rcs[ii*umax_num_neigh_atoms+jj][0];
            neigh_vec[1] = rcs[ii*umax_num_neigh_atoms+jj][1];
            neigh_vec[2] = rcs[ii*umax_num_neigh_atoms+jj][2];
            distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                     + std::pow(neigh_vec[1], 2)
                                     + std::pow(neigh_vec[2], 2) );
            if (distance_ij > rmax)
                continue;
            
            for (int aa=0; aa<3; aa++) {
                CoordType e_site_ders_ija = 0;
                // eders_ija
                for (int i=0; i<alpha_index_basic_count; i++)
                    e_site_ders_ija += e_site_der2mom[i] * mom_ders[i*umax_num_neigh_atoms + jj][aa];
                
                // Force
                #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                #pragma omp atomic
                #endif
                force[center_idx][aa] += e_site_ders_ija;
                #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                #pragma omp atomic
                #endif
                force[neigh_idx][aa] -= e_site_ders_ija;

                // Virial
                for (int bb=0; bb<3; bb++) {
                    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
                    #pragma omp atomic
                    #endif
                    virial[aa*3 + bb] -= e_site_ders_ija * neigh_vec[bb];
                }
            }
        }
    }

    // Step . Free
    free(mom_vals);
    free(mom_ders);
    free(e_site_der2mom);
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
}
#endif
}



template <typename CoordType>
void NNMtp<CoordType>::find_descriptors(
    CoordType *all_descriptors,
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*relative_coords)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    memset(all_descriptors, 0, sizeof(CoordType) * inum * alpha_scalar_moments);

    // Step 2.
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
#pragma omp parallel private(mom_vals, mom_ders)
{
#endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
    
    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++)
    {
        MomsValDer<CoordType>::find_val_der(
            mom_vals,
            mom_ders,
            chebyshev_size,
            coeffs,
            alpha_moments_count,
            alpha_index_basic_count,
            alpha_index_basic,
            alpha_index_times_count,
            alpha_index_times,
            nmus,
            ilist[ii],
            numneigh[ii],
            &firstneigh[ii*umax_num_neigh_atoms],
            &relative_coords[ii*umax_num_neigh_atoms],
            types,
            ntypes,
            umax_num_neigh_atoms,
            rmax,
            rmin);

        for (int i=0; i<alpha_scalar_moments; i++)
            all_descriptors[ii*alpha_scalar_moments + i] = mom_vals[alpha_moment_mapping[i]];
    }

    free(mom_vals);
    free(mom_ders);
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
}
#endif
}


template <typename CoordType>
void NNMtp<CoordType>::find_e_sites(
    CoordType *e_sites,
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*rcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    memset(e_sites, 0.0, sizeof(CoordType) * inum);

    // Step 2.
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
#pragma omp parallel private(mom_vals, mom_ders)
{
#endif
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);

    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++)
    {
        center_idx = ilist[ii];
        type_central = types[center_idx];
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
        CoordType *type_central_w0 = &w0[type_central * num_neurons * alpha_scalar_moments];
        CoordType *type_central_w1 = &w1[type_central * num_neurons];

        MomsValDer<CoordType>::find_val_der(
            mom_vals,
            mom_ders,
            chebyshev_size,
            coeffs,
            alpha_moments_count,
            alpha_index_basic_count,
            alpha_index_basic,
            alpha_index_times_count,
            alpha_index_times,
            nmus,
            ilist[ii],
            numneigh[ii],
            &firstneigh[ii*umax_num_neigh_atoms],
            &rcs[ii*umax_num_neigh_atoms],
            types,
            ntypes,
            umax_num_neigh_atoms,
            rmax,
            rmin);

        // NN Energy
        e_sites[ii] = type_bias[type_central];
        for (int p=0; p<num_neurons; p++) {
            CoordType hidden_val = 0;
            CoordType activated_hidden_val = 0;
            for (int k=0; k<alpha_scalar_moments; k++)
                hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
            TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
            e_sites[ii] += type_central_w1[p] * activated_hidden_val;
        }
    }

    // Step . Free
    free(mom_vals);
    free(mom_ders);
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
}
#endif
}


template <typename CoordType>
void NNMtp<CoordType>::find_e_sites_backward(
    CoordType *e_sites_der2coeffs,
    CoordType *e_sites_der2w0,
    CoordType *e_sites_der2w1,
    CoordType *e_sites_der2type_bias,
    int chebyshev_size,
    int num_neurons,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    const int alpha_moments_count,
    const int alpha_index_basic_count,
    const int (*alpha_index_basic)[4],
    const int alpha_index_times_count,
    const int (*alpha_index_times)[4],
    const int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int nmus,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*rcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType *e_site_der2mom;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    int max_alpha_index_basic = 0;
    for (int ii=0; ii<alpha_index_basic_count; ii++) {
        int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][3];
        if (now_alpha_index_basic > max_alpha_index_basic)
            max_alpha_index_basic = now_alpha_index_basic;
    }
    max_alpha_index_basic++;

#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
#pragma omp parallel private(mom_vals, e_site_der2mom)
{
#endif
    // Step 2.
    int center_idx;
    int type_central;
    int neigh_idx;
    int type_outer;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    CoordType *auto_dist_powers_;
    CoordType (*auto_coords_powers_)[3];
    RQ_Chebyshev<CoordType> *p_RadialBasis;

    // Step 3. 
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    e_site_der2mom = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic);
    auto_coords_powers_ = (CoordType (*)[3])malloc(sizeof(CoordType) * max_alpha_index_basic * 3);
    p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);

    #if defined(USE_OPENMP) or defined(__INTELLISENSE__)
    #pragma omp for schedule(static)
    #endif
    for (int ii=0; ii<inum; ii++) {
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(e_site_der2mom, 0, sizeof(CoordType) * alpha_moments_count);
        center_idx = ilist[ii];
        type_central = types[center_idx];
        CoordType *type_central_w0 = &w0[type_central*num_neurons*alpha_scalar_moments];
        CoordType *type_central_w1 = &w1[type_central*num_neurons];

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            neigh_vec[0] = rcs[ii*umax_num_neigh_atoms + jj][0];
            neigh_vec[1] = rcs[ii*umax_num_neigh_atoms + jj][1];
            neigh_vec[2] = rcs[ii*umax_num_neigh_atoms + jj][2];
            distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                     + std::pow(neigh_vec[1], 2)
                                     + std::pow(neigh_vec[2], 2) );
            if (distance_ij > rmax)
                continue;
            distance_ij_inv = 1.0 / distance_ij;
            p_RadialBasis->build(distance_ij);

            auto_dist_powers_[0] = 1.0;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[0][aa] = 1.0;
            for (int k=1; k<max_alpha_index_basic; k++) {
                auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
                for (int aa=0; aa<3; aa++)
                    auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * neigh_vec[aa];
            }

            for (int i=0; i<alpha_index_basic_count; i++) {
                int mu = alpha_index_basic[i][0];
                int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
                CoordType powk = 1.0 / auto_dist_powers_[k];
                CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
                CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
                CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
                CoordType mult0 = pow0 * pow1 * pow2;

                for (int xi=0; xi<chebyshev_size; xi++) {
                    int idx = (type_central*ntypes + type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                    CoordType A = p_RadialBasis->vals()[xi];
                    CoordType B = mult0;
                    CoordType C = powk;
                    mom_vals[i] += coeffs[idx] * A * B * C;
                }
            }
        }

        for (int i=0; i<alpha_index_times_count; i++) {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];
            mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
        }

        // Step 4. Backward
        for (int p=0; p<num_neurons; p++) {
            CoordType hidden_val = 0.0;
            CoordType activated_hidden_der = 0.0;
            for (int k=0; k<alpha_scalar_moments; k++)
                hidden_val += type_central_w0[p*alpha_scalar_moments+k] * mom_vals[alpha_moment_mapping[k]];
            TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
            
            for (int k=0; k<alpha_scalar_moments; k++)
                e_site_der2mom[alpha_moment_mapping[k]] += type_central_w1[p]
                                                           * activated_hidden_der
                                                           * type_central_w0[p*alpha_scalar_moments + k];
        }

        for (int i=alpha_index_times_count-1; i>=0; i--) {
            CoordType val0 = mom_vals[alpha_index_times[i][0]];
            CoordType val1 = mom_vals[alpha_index_times[i][1]];
            CoordType val2 = alpha_index_times[i][2];

            e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2 * val1;
            e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                       * val2 * val0;
        }

        // Step 5.
        for (int jj=0; jj<numneigh[ii]; jj++)
        {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            neigh_vec[0] = rcs[ii*umax_num_neigh_atoms + jj][0];
            neigh_vec[1] = rcs[ii*umax_num_neigh_atoms + jj][1];
            neigh_vec[2] = rcs[ii*umax_num_neigh_atoms + jj][2];
            distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                     + std::pow(neigh_vec[1], 2)
                                     + std::pow(neigh_vec[2], 2) );
            if (distance_ij > rmax)
                continue;
            distance_ij_inv = 1.0 / distance_ij;
            p_RadialBasis->build(distance_ij);

            auto_dist_powers_[0] = 1.0;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[0][aa] = 1.0;
            for (int k=1; k<max_alpha_index_basic; k++) {
                auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
                for (int aa=0; aa<3; aa++)
                    auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * neigh_vec[aa];
            }

            for (int i=0; i<alpha_index_basic_count; i++) {
                int mu = alpha_index_basic[i][0];
                int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
                CoordType powk = 1.0 / auto_dist_powers_[k];
                CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
                CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
                CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
                CoordType mult0 = pow0 * pow1 * pow2;

                for (int xi=0; xi<chebyshev_size; xi++) {
                    int idx = (type_central*ntypes+type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                    CoordType A = p_RadialBasis->vals()[xi];
                    CoordType B = mult0;
                    CoordType C = powk;

                    e_sites_der2coeffs[ii*num_coeffs + idx] += e_site_der2mom[i] * A * B * C;
                }
            }
        }

        for (int p=0; p<num_neurons; p++) {
            CoordType hidden_val = 0.0;
            CoordType activated_hidden_val = 0.0;
            CoordType activated_hidden_der = 0.0;
            for (int k=0; k<alpha_scalar_moments; k++)
                hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
            TanhActivationFunc<CoordType>::find_val(activated_hidden_val, hidden_val);
            TanhActivationFunc<CoordType>::find_der(activated_hidden_der, hidden_val);
            
            e_sites_der2w1[ii*num_neurons + p] = activated_hidden_val;

            for (int k=0; k<alpha_scalar_moments; k++)
                e_sites_der2w0[ii*num_neurons*alpha_scalar_moments + p*alpha_scalar_moments + k] += type_central_w1[p]
                                                                                                    * activated_hidden_der
                                                                                                    * mom_vals[alpha_moment_mapping[k]];
        }

        e_sites_der2type_bias[ii*ntypes + type_central] += 1.0;
    }


    // Step . Free
    free(mom_vals);
    free(e_site_der2mom);
    free(auto_dist_powers_);
    free(auto_coords_powers_);
    delete p_RadialBasis;
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
}
#endif
}



};  // namespace : nnmtp
};  // namespace : ai2pot


#endif
