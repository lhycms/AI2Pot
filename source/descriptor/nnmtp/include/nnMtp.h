#ifndef AI2POT_NNMTP_NNMTP_H
#define AI2POT_NNMTP_NNMTP_H

#include <stdio.h>
#include <stdlib.h>
#if defined(USE_OPENMP) or defined(__INTELLISENSE__)
#include <omp.h>
#endif

#include "./mtpBasis.h"
#include "./nnMtpUtils.h"


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
        CoordType rmin,
        CoordType zbl_rmax,
        CoordType zbl_rmin,
        CoordType *zbl_cks,
        CoordType *zbl_dks);

    static void find_ef_loss();

    static void find_ef_loss_backward();

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
    CoordType rmin,
    CoordType zbl_rmax,
    CoordType zbl_rmin,
    CoordType *zbl_cks,
    CoordType *zbl_dks)
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
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_index_basic_count * 3);
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
        type_central_w0 = &w0[type_central*num_neurons*alpha_moments_count];
        type_central_w1 = &w1[type_central*num_neurons];
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_index_basic_count * 3);
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
            for (int k=0; k<alpha_scalar_moments; k++)
                hidden_val += type_central_w0[p*alpha_scalar_moments + k] * mom_vals[alpha_moment_mapping[k]];
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
                e_site_der2mom[alpha_moment_mapping[k]] += type_central_w1[p];
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


};  // namespace : nnmtp
};  // namespace : ai2pot


#endif