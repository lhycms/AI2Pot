#ifndef AI2POT_MTP_BASIS_TO_EFV_H
#define AI2POT_MTP_BASIS_TO_EFV_H

#include <stdio.h>
#include <stdlib.h>

#include "./basis.h"
#include "./mtpParam.h"
#include "./mtpBasis.h"

namespace ai2pot {
namespace mtpr {


template <typename CoordType>
class MtpBasisToEFV {
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
        CoordType rmin);
};  // class : MtpBasisToE


template <typename CoordType>
void MtpBasisToEFV<CoordType>::find_efv(
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
    CoordType rmin)
{
    // Step 1.
    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_moments_count * 3);

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
    for (int ii=0; ii<inum; ii++)
    {
        type_centra = types[ilist[ii]];
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
            max_num_mus4mom,
            num_mus4moms,
            mus4moms_ptr,
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

        // Linear
        etot += type_bias[type_central];
        for (int i=0; i<alpha_scalar_moments; i++)
            etot += linear_coeffs[i] * mom_vals[alpha_moment_mapping[i]];
    }

    // Step . Free
    free(mom_vals);
    free(mom_ders);
    free(auto_dist_powers_);
    free(auto_coords_powers_);
    delete p_RadialBasis;
}

};  // namespace : mtpr
};  // namespace : ai2pot
#endif