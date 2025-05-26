#ifndef AI2POT_LINEAR_MTP_CUH
#define AI2POT_LINEAR_MTP_CUH
#include <math.h>

#include "./mtp_utilities.cuh"
#include "./mtpBasis.cuh"


namespace ai2pot {
namespace mtpr {


template <typename CoordType>
static __host__ __device__
void find_efv_atom(CoordType &etot,
                    CoordType (*force)[3],
                    CoordType *virial,
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
                    int silist,
                    int snumneigh,
                    int *sfirstneigh,
                    CoordType (*srcs)[3],
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



template <typename CoordType>
__host__ __device__ 
void find_efv_atom(CoordType &etot,
                   CoordType (*force)[3],
                   CoordType *virial,
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
                   int silist,
                   int snumneigh,
                   int *sfirstneigh,
                   CoordType (*srcs)[3],
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
    // Step 1. Init temp array
    CoordType mom_vals[MAX_ALPHA_MOMENTS_COUNT] = {0.};
    CoordType e_site_der2mom[MAX_ALPHA_MOMENTS_COUNT] = {0.};

    CoordType auto_dist_powers_[MAX_ALPHA_INDEX_BASIC] = {0.};
    CoordType auto_coords_powers_[MAX_ALPHA_INDEX_BASIC][3] = {{0.}};

    int center_idx;
    int type_central;
    int Zi;
    int neigh_idx;
    int type_outer;
    int Zj;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    CoordType NeighbVect[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;
    
    center_idx = silist;
    type_central = types[center_idx];
    Zi = type_map[type_central];

    
    // Step 2. Calculate mom_vals
    // Step 2.1.
    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        Zj = type_map[type_outer];
        for (int aa=0; aa<3; aa++)
            NeighbVect[aa] = srcs[jj][aa];
        
        distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                 + std::pow(NeighbVect[1], 2)
                                 + std::pow(NeighbVect[2], 2) );
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;
        CoordType rq_chebyshev_vals[MAX_CHEBYSHEV_SIZE] = {0.0};
        CoordType rq_chebyshev_ders2r[MAX_CHEBYSHEV_SIZE] = {0.0};
        find_rq_chebyshev<CoordType>(rq_chebyshev_vals,
                                     rq_chebyshev_ders2r,
                                     chebyshev_size,
                                     rmax,
                                     rmin,
                                     distance_ij);
        

        auto_dist_powers_[0] = 1.0;
        for (int aa=0; aa<3; aa++)
            auto_dist_powers_[0][aa] = NeighbVect[aa];
        for (int k=1; k<MAX_ALPHA_INDEX_BASIC; k++) {
            auto_dist_powers_[k] = auto_dist_powers_[k-1] * distance_ij;
            for (int aa=0; aa<3; aa++)
                auto_coords_powers_[k][aa] = auto_coords_powers_[k-1][aa] * NeighbVect[aa];
        }

        for (int i=0; i<alpha_index_basic_count; i++)
        {
            int mu = alpha_index_basic[i][0];
            int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
            CoordType powk = 1.0 / auto_dist_powers_[k];
            CoordType pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
            CoordType pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
            CoordType pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];
            CoordType mult0 = pow0 * pow1 * pow2;

            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx = (type_central*ntypes + type_outer)*nmus*chebyshev_size + mu*chebyshev_size + xi;
                CoordType A = rq_chebyshev_vals[xi];
                CoordType B = mult0;
                CoordType C = powk;
                mom_vals[i] += coeffs[idx] * A * B * C;
            }
        }
    }
    
    // Step 2.2.
    for (int i=0; i<alpha_index_times_count; i++)
    {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];
        mom_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
    }


    // Step 3. Calculate EFV for atom
    CoordType e_site = 0;
    for (int i=0; i<alpha_scalar_moments; i++)
        e_site += linear_coeffs[i] * mom_vals[alpha_moment_mapping[i]];
    atomicAdd(&etot, e_site);

    for (int i=0; i<alpha_scalar_moments; i++)
        e_site_der2mom[alpha_moment_mapping[i]] = linear_coeffs[i];
    
    for (int i=alpha_index_times_count; i>=0; i--) {
        CoordType val0 = mom_vals[alpha_index_times[i][0]];
        CoordType val1 = mom_vals[alpha_index_times[i][1]];
        CoordType val2 = alpha_index_times[i][2];

        e_site_der2mom[alpha_index_times[i][1]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val0;
        e_site_der2mom[alpha_index_times[i][0]] += e_site_der2mom[alpha_index_times[i][3]]
                                                   * val2 * val1;
    }

    for (int jj=0; jj<snumneigh; jj++) {
        
    }
}


};  // namespace : mtpr
};  // namespace : ai2pot


#endif