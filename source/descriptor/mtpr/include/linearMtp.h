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

#ifndef AI2POT_LINEAR_MTP_H
#define AI2POT_LINEAR_MTP_H

#include <stdio.h>
#include <stdlib.h>

#include "./basis.h"
#include "./mtpParam.h"
#include "./mtpBasis.h"
#include "./mtpLoss.h"

namespace ai2pot {
namespace mtpr {


template <typename CoordType>
class LinearMtp {
public:
    static void find_efv(
        CoordType &etot,
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
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        CoordType (*relative_coords)[3],
        int *types,
        int ntypes,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin);

    static void find_efv_backward();

    static void find_loss(
        CoordType &loss,
        CoordType e_weight,
        CoordType f_weight,
        CoordType v_weight,
        CoordType etot_dft,
        CoordType (*force_dft)[3],
        CoordType *virial_dft,
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
        int nghost,
        CoordType rmax,
        CoordType rmin);

    static void find_loss_backward();
};  // class : MtpBasisToE


template <typename CoordType>
void LinearMtp<CoordType>::find_efv(
    CoordType &etot,
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
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*relative_coords)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms,
    int nghost,
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
    e_site_der2mom = (CoordType *)malloc(sizeof(CoordType) * alpha_moments_count);
    e_site_ders = (CoordType (*)[3])malloc(sizeof(CoordType) * umax_num_neigh_atoms * 3);

    int type_central;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    CoordType NeighbVect[3] = {0.};
    CoordType distance_ij;

    // Step 2.
    etot = 0;
    CoordType e_site = 0;
    memset(force, 0, sizeof(CoordType) * (inum+nghost) * 3);
    memset(virial, 0, sizeof(CoordType) * 9);
    for (int ii=0; ii<inum; ii++)
    {
        memset(mom_vals, 0, sizeof(CoordType) * alpha_moments_count);
        memset(mom_ders, 0, sizeof(CoordType) * alpha_index_basic_count * umax_num_neigh_atoms * 3);
        memset(e_site_der2mom, 0, sizeof(CoordType) * alpha_moments_count);
        memset(e_site_ders, 0, sizeof(CoordType) * umax_num_neigh_atoms * 3);

        int center_idx = ilist[ii];
        type_central = types[ilist[ii]];
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

        // Linear Energy
        e_site = type_bias[type_central];
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
        
        for (int i=0; i<alpha_index_basic_count; i++) {
            for (int jj=0; jj<numneigh[ii]; jj++) {
                NeighbVect[0] = relative_coords[ii*umax_num_neigh_atoms+jj][0];
                NeighbVect[1] = relative_coords[ii*umax_num_neigh_atoms+jj][1];
                NeighbVect[2] = relative_coords[ii*umax_num_neigh_atoms+jj][2];
                distance_ij = std::sqrt( std::pow(NeighbVect[0], 2)
                                       + std::pow(NeighbVect[1], 2)
                                       + std::pow(NeighbVect[2], 2) );
                if (distance_ij > rmax)
                    continue;

                int neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
                for (int a=0; a<3; a++) {
                    e_site_ders[jj][a] = e_site_der2mom[i] * mom_ders[i*umax_num_neigh_atoms+jj][a];
                    // Linear Force
                    force[center_idx][a] += e_site_ders[jj][a];
                    force[neigh_idx][a] -= e_site_ders[jj][a];

                    // Linear Virial
                    for (int b=0; b<3; b++)
                        virial[a*3 + b] -= e_site_ders[jj][a] * relative_coords[ii*umax_num_neigh_atoms + jj][b];
                }
            }
        }

    }

    // Step . Free
    free(mom_vals);
    free(mom_ders);
    free(e_site_der2mom);
    free(e_site_ders);
}


template <typename CoordType>
void LinearMtp<CoordType>::find_loss(
    CoordType &loss,
    CoordType e_weight,
    CoordType f_weight,
    CoordType v_weight,
    CoordType etot_dft,
    CoordType (*force_dft)[3],
    CoordType *virial_dft,
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
    int nghost,
    CoordType rmax,
    CoordType rmin)
{
    CoordType etot;
    CoordType (*force)[3];
    CoordType *virial;

    force = (CoordType (*)[3])malloc(sizeof(CoordType) * inum * 3);
    virial = (CoordType*)malloc(sizeof(CoordType) * 9);
    
    find_efv(
        etot,
        force,
        virial,
        chebyshev_size,
        coeffs,
        linear_coeffs,
        type_bias,
        alpha_moments_count,
        alpha_index_basic_count,
        alpha_index_basic,
        alpha_index_times_count,
        alpha_index_times,
        alpha_scalar_moments,
        alpha_moment_mapping,
        nmus,
        inum,
        ilist,
        numneigh,
        firstneigh,
        relative_coords,
        types,
        ntypes,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);
    MtpLoss<CoordType>::find_loss(
        loss,
        e_weight,
        f_weight,
        v_weight,
        etot,
        etot_dft,
        force,
        force_dft,
        virial,
        virial_dft,
        inum);
    

    free(force);
    free(virial);
}


};  // namespace : mtpr
};  // namespace : ai2pot
#endif