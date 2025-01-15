#ifndef AI2POT_LINEAR_MTP_H
#define AI2POT_LINEAR_MTP_H

#include <stdio.h>
#include <stdlib.h>

#include "./basis.h"
#include "./mtpParam.h"
#include "./mtpBasis.h"

namespace ai2pot {
namespace mtpr {


template <typename CoordType>
class LinearMtp {
public:
    static void find_efv(
        CoordType &etot,
        int chebyshev_size, 
        CoordType *coeffs,
        CoordType *linear_coeffs,
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
        int umax_num_neigh_atoms,
        CoordType rmax,
        CoordType rmin);
};  // class : MtpBasisToE


template <typename CoordType>
void LinearMtp<CoordType>::find_efv(
    CoordType &etot,
    int chebyshev_size,
    CoordType *coeffs,
    CoordType *linear_coeffs,
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
    int umax_num_neigh_atoms,
    CoordType rmax,
    CoordType rmin)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    CoordType *e_site_der2mom;
    CoordType (*e_site_ders)[3];
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
    e_site_der2mom = (CoordType *)malloc(sizeof(CoordType) * alpha_scalar_moments);
    e_site_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * umax_num_neigh_atoms * 3);

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
    auto_coords_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic * 3);
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    int type_central;
    int type_outer;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    RQ_Chebyshev<CoordType> *p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);


    // Step 2.
    etot = 0;
    for (int ii=0; ii<inum; ii++)
    {
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_index_basic_count * 3);
        memset(e_site_der2mom, 0, sizeof(CoordType) * alpha_scalar_moments);
        memset(e_site_ders, 0, sizeof(CoordType) * umax_num_neigh_atoms * 3);

        type_central = types[ilist[ii]];
        MomValDer<CoordType>::find_val_der_one(
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
            firstneigh[ii*umax_num_neigh_atoms],
            relative_coords[ii*umax_num_neigh_atoms],
            types,
            ntypes,
            umax_num_neigh_atoms,
            rmax,
            rmin);

        // Linear Energy
        CoordType e_site = type_bias[type_central];
        for (int i=0; i<alpha_scalar_moments; i++)
            e_site += linear_coeffs[i] * mom_vals[alpha_moment_mapping[i]];
        etot += e_site;

        // Linear Energy derivative w.r.t. xyz
        for (int i=0; i<alpha_scalar_moments; i++)
            e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i];
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
        for (int i=0; i<alpha_index_basic_count; i++)
            for (int jj=0; jj<numneigh[ii]; jj++)
                for (int a=0; a<3; a++)
                    e_site_ders[jj][a] = e_site_der2mom[i] * mom_ders[i*umax_num_neigh_atoms+jj][a];

        // Linear Force
        

        // Linear Virial

    }

    // Step . Free
    free(mom_vals);
    free(mom_ders);
    free(auto_dist_powers_);
    free(auto_coords_powers_);
    free(e_site_der2mom);
    free(e_site_ders);
    delete p_RadialBasis;
}

};  // namespace : mtpr
};  // namespace : ai2pot
#endif