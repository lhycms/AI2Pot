#ifndef MATERSDK_ENVMATRIX_H
#define MATERSDK_ENVMATRIX_H
#include <cstring>
#include "./se.h"

namespace matersdk {
namespace deepPotSE {

template <typename CoordType>
class EnvMatrix {
public:
    static void find_value_deriv(
        CoordType* tilde_r,
        CoordType* tilde_r_deriv,
        int inum, 
        int* ilist,
        int* numneigh,
        int* firstneigh,
        CoordType* relative_coords,
        int* types,
        int ntypes,
        int* umax_num_neigh_atoms_lst,
        CoordType rcut,
        CoordType rcut_smooth);
};  // class : EnvMatrix


/**
 * @brief Find the value and deriv wrt. R_{ij} of Environment Matrix
 * 
 * @tparam CoordType 
 * @param tilde_r : value of Environment Matrix.
 *          len = inum * sum(umax_num_neigh_atoms_lst) * 4
 * @param tilde_r_deriv : deriv wrt. R{ij} of Environment Matrix
 *          len = inum * sum(umax_num_neigh_atoms_lst) * 4 * 3
 * @param inum : number of primitive cell atoms
 * @param ilist : len = inum
 * @param numneigh : len = inum
 * @param firstneigh : len = inum * umax_num_neigh_atoms
 * @param relative_coords : len = inum * nmax_num_neigh_atoms * 3
 * @param types : len = inum, depends on `firstneigh`
 * @param ntypes : number of element kinds
 * @param umax_num_neigh_atoms_lst : len = ntypes. User specified `umax_num_neigh_atoms` for all elements.
 * 
 */
template <typename CoordType>
void EnvMatrix<CoordType>::find_value_deriv(
        CoordType* tilde_r,
        CoordType* tilde_r_deriv,
        int inum, 
        int* ilist,
        int* numneigh,
        int* firstneigh,
        CoordType* relative_coords,
        int* types,
        int ntypes,
        int* umax_num_neigh_atoms_lst,
        CoordType rcut,
        CoordType rcut_smooth)
{   
    // Step 1. 
    // Step 1.1. Init tilde_r, tilde_r_deriv
    int umax_num_neigh_atoms = 0;
    for (int ii=0; ii<ntypes; ii++)
        umax_num_neigh_atoms += umax_num_neigh_atoms_lst[ii];
    memset(tilde_r, 0.0, sizeof(CoordType) * inum * umax_num_neigh_atoms * 4);
    memset(tilde_r_deriv, 0.0, sizeof(CoordType) * inum * umax_num_neigh_atoms * 4 * 3);
    
    // Step 1.2. 
    CoordType tmp_distance_ij;
    CoordType tmp_distance_ij_recip;
    CoordType tilde_s_value;
    CoordType tilde_x_value;
    CoordType tilde_y_value;
    CoordType tilde_z_value;
    SwitchFunc<CoordType> switch_func(rcut, rcut_smooth);
    int* nstart_idxs = (int*)malloc(sizeof(int) * ntypes);
    memset(nstart_idxs, 0, sizeof(int) * ntypes);
    for (int ii=0; ii<ntypes; ii++)
        for (int jj=0; jj<ii; jj++)
            nstart_idxs[ii] += umax_num_neigh_atoms_lst[jj];
    int* nloop_idxs = (int*)malloc(sizeof(int) * ntypes);
    memset(nloop_idxs, 0, sizeof(int) * ntypes);
    CoordType* tmp_diff_cart_coords = (CoordType*)malloc(sizeof(CoordType) * 3);

    // Step 2. 
    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<ntypes; jj++)
            nloop_idxs[jj] = 0;

        for (int jj=0; jj<numneigh[ii]; jj++) {
            int tmp_neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            int tmp_neigh_type = types[tmp_neigh_idx];
            tmp_diff_cart_coords[0] = relative_coords[ii*umax_num_neigh_atoms*3 + jj*3 + 0];
            tmp_diff_cart_coords[1] = relative_coords[ii*umax_num_neigh_atoms*3 + jj*3 + 1];
            tmp_diff_cart_coords[2] = relative_coords[ii*umax_num_neigh_atoms*3 + jj*3 + 2];
            tmp_distance_ij = std::sqrt( 
                std::pow(tmp_diff_cart_coords[0], 2) + 
                std::pow(tmp_diff_cart_coords[1], 2) +
                std::pow(tmp_diff_cart_coords[2], 2));
            tmp_distance_ij_recip = recip<CoordType>(tmp_distance_ij);
            tilde_s_value = smooth_func(tmp_distance_ij, rcut, rcut_smooth);
            tilde_x_value = tilde_s_value * tmp_distance_ij_recip * tmp_diff_cart_coords[0];
            tilde_y_value = tilde_s_value * tmp_distance_ij_recip * tmp_diff_cart_coords[1];
            tilde_z_value = tilde_s_value * tmp_distance_ij_recip * tmp_diff_cart_coords[2];

            // Step 2.1. Value of EnvMatrix
            tilde_r[ii*umax_num_neigh_atoms*4 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4 + 0] = tilde_s_value;
            tilde_r[ii*umax_num_neigh_atoms*4 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4 + 1] = tilde_x_value;
            tilde_r[ii*umax_num_neigh_atoms*4 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4 + 2] = tilde_y_value;
            tilde_r[ii*umax_num_neigh_atoms*4 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4 + 3] = tilde_z_value;

            // Step 2.2. Deriv wrt. R_{ij} of EnvMatrix
            // Step 2.2.1. switch_func * 1/r_ij
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 0*3 + 0] = (
                std::pow(tmp_distance_ij_recip, 2) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[0] -
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[0]);
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 0*3 + 1] = (
                std::pow(tmp_distance_ij_recip, 2) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[1] -
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[1]);      
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 0*3 + 2] = (
                std::pow(tmp_distance_ij_recip, 2) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[2] -
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[2]);

            // Step 2.2.2. switch_func * 1/r_ij * x_ij * 1/r_ij
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 1*3 + 0] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * std::pow(tmp_diff_cart_coords[0], 2) + 
                std::pow(tmp_distance_ij_recip, 2) * switch_func.get_result(tmp_distance_ij) - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * std::pow(tmp_diff_cart_coords[0], 2));
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 1*3 + 1] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[0] * tmp_diff_cart_coords[1] - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[0] * tmp_diff_cart_coords[1]);
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 1*3 + 2] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[0] * tmp_diff_cart_coords[2] - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[0] * tmp_diff_cart_coords[2]);

            // Step 2.2.3. switch_func * 1/r_ij * y_ij * 1/r_ij
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 2*3 + 0] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[1] * tmp_diff_cart_coords[0] - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[1] * tmp_diff_cart_coords[0]);
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 2*3 + 1] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * std::pow(tmp_diff_cart_coords[1], 2) + 
                std::pow(tmp_distance_ij_recip, 2) * switch_func.get_result(tmp_distance_ij) - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * std::pow(tmp_diff_cart_coords[1], 2));
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 2*3 + 2] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[1] * tmp_diff_cart_coords[2] - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[1] * tmp_diff_cart_coords[2]);

            // Step 2.2.4. switch_func * 1/r_ij * z_ij * 1/r_ij
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 3*3 + 0] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[2] * tmp_diff_cart_coords[0] - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[2] * tmp_diff_cart_coords[0]);
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 3*3 + 1] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * tmp_diff_cart_coords[2] * tmp_diff_cart_coords[1] - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * tmp_diff_cart_coords[2] * tmp_diff_cart_coords[1]);
            tilde_r_deriv[ii*umax_num_neigh_atoms*4*3 + (nstart_idxs[tmp_neigh_type]+nloop_idxs[tmp_neigh_type])*4*3 + 3*3 + 2] = (
                std::pow(tmp_distance_ij_recip, 3) * switch_func.get_deriv2r(tmp_distance_ij) * std::pow(tmp_diff_cart_coords[2], 2) + 
                std::pow(tmp_distance_ij_recip, 2) * switch_func.get_result(tmp_distance_ij) - 
                2 * std::pow(tmp_distance_ij_recip, 4) * switch_func.get_result(tmp_distance_ij) * std::pow(tmp_diff_cart_coords[2], 2));

            // Step 2.3. 
            nloop_idxs[tmp_neigh_type]++;
        }
    }

    // Step . Free memory
    free(nstart_idxs);
    free(nloop_idxs);
    free(tmp_diff_cart_coords);
}


}   // namespace : deepPotSE
}   // namespace : matersdk


#endif