#ifndef AI2POT_MTP_BASIS_GRAD_H
#define AI2POT_MTP_BASIS_GRAD_H
#include <stdlib.h>
#include <cstring>
#include <cassert>

#include "./basis.h"


namespace ai2pot {
namespace mtpr {


template <typename CoordType>
class MtpBasisGrad {
public:
    static void find_val_der(
        CoordType *(mbg_val)[3],
        CoordType (*mbg_der)[3][3],
        CoordType *mbg_der2coeffs,
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

    static void find_der_backward();
};



template <typename CoordType>
void MtpBasisGrad<CoordType>::find_val_der(
    CoordType *(mbg_val)[3],
    CoordType (*mbg_der)[3][3],
    CoordType *mbg_der2coeffs,
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
    memset(mbg_val, 0, sizeof(CoordType) * inum * alpha_scalar_moments * umax_num_neigh_atoms * 3);
    memset(mbg_der, 0, sizeof(CoordType) * inum * alpha_scalar_moments * umax_num_neigh_atoms * 3 * 3);
    memset(mbg_der2coeffs, 0, sizeof(CoordType) * inum * alpha_scalar_moments * umax_num_neigh_atoms * 3 * ntypes * ntypes * nmus * chebyshev_size);

    CoordType *mom_vals;
    CoordType (*mom_ders)[3];
    mom_vals = (CoordType*)malloc(sizeof(CoordType) * alpha_moments_count);
    mom_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);
    //memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
    //memset(mom_ders, 0, sizeof(CoordType) * alpha_moments_count * umax_num_neigh_atoms * 3);

    int max_alpha_index_basic = 0;
    for (int ii=0; ii<alpha_index_basic_count; ii++) {
        int now_alpha_index_basic = alpha_index_basic[ii][1] + alpha_index_basic[ii][2] + alpha_index_basic[ii][3];
        if (now_alpha_index_basic > max_alpha_index_basic)
            max_alpha_index_basic = now_alpha_index_basic;
    }
    max_alpha_index_basic++;

    CoordType *auto_dist_powers_;
    CoordType *(auto_coords_powers_)[3];
    auto_dist_powers_ = (CoordType*)malloc(sizeof(CoordType) * max_alpha_index_basic);
    auto_coords_powers_ = (CoordType (*)[3])malloc(sizeof(CoordType) * max_alpha_index_basic * 3);
    //memset(auto_dist_powers_, 0, sizeof(CoordType) * max_alpha_index_basic);
    //memset(auto_coords_powers_, 0, sizeof(CoordType) * max_alpha_index_basic * 3);
    CoordType NeighVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    int type_central;
    int type_outer;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

    RQ_Chebyshev<CoordType>* p_RadialBasis = new RQ_Chebyshev<CoordType>(chebyshev_size, rmax, rmin);

    // Step 2.
    for (int ii=0; ii<inum; ii++) {
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_moment_mapping * umax_num_neigh_atoms * 3);
        type_central = types[ilist[ii]];

        for (int jj=0; jj<numneigh[ii]; jj++)
        {
            type_outer = types[firstneigh[ii*umax_num_neigh_atoms + jj]];
            for (int a=0; a<3; a++)
                NeighVect[a] = relative_coords[ii*umax_num_neigh_atoms + jj][a];
            distance_ij = std::sqrt( std::pow(NeighVect[0], 2) 
                                     + std::pow(NeighVect[1], 2)
                                     + std::pow(NeighVect[2], 2) );
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
                    auto_coords_powers_[k][a] = auto_coords_powers_[k-1][a] * NeighVect[a];
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
                    int idx = (type_central*ntypes + type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                    
                    CoordType A = p_RadialBasis->vals()[xi];
                    CoordType B = mult0;
                    CoordType C = powk;
                    CoordType Ad[3] = p_RadialBasis->ders2r()[xi];
                    CoordType Bd[3] = ;
                    CoordType Cd[3] = ;
                    CoordType Add[3] = p_RadialBasis->ders2r_ders2r()[xi];
                    CoordType Bdd[3] = ;
                    CoordType Cdd[3] = ;

                    mom_vals[i] += coeffs[idx] * A * B * C;
                    mom_ders[i*umax_num_neigh_atoms*3 + jj][0] += coeffs[idx] * Ad;
                    mom_ders[i*umax_num_neigh_atoms*3 + jj][1] += ;
                    mom_ders[i*umax_num_neigh_atoms*3 + jj][2] += ;
                }
            }
            
        }
    }

    // Step . Free
    free(mom_vals);
    free(mom_ders);
}



};  // namespace : mtpr
};  // namespace : ai2pot


#endif