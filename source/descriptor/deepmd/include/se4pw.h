#ifndef MATERSDK_SE4PW_H
#define MATERSDK_SE4PW_H
#include "./se.h"

namespace matersdk {
namespace deepPotSE {


template <typename CoordType>
class Se4pw {
public:
    static void generate(
        CoordType* tilde_r,
        CoordType* tilde_r_deriv,
        CoordType* relative_coords,
        int inum,
        int* ilist,
        int* numneigh,
        int* firstneigh,   // shape = inum * tot_num_neigh_atoms
        CoordType* x,      // shape = supercell_inum * 3
        int* types, // starts from 0.
        int ntypes, // starts from 0. e.g. 2
        int* num_neigh_atoms_lst,
        CoordType rcut,
        CoordType rcut_smooth);
    
    // Compatible with Fortran. 0 stands for none atom. indices starts from 1
    static void get_prim_indices_from_matersdk(
        int* prim_indices,
        int inum,
        int* ilist,
        int* numneigh,
        int* firstneigh,   // shape = inum * tot_num_neigh_atoms
        int* types,     // starts from 0.
        int ntypes,    // starts from 0. e.g. 2
        int* num_neigh_atoms_lst);

    // Compatible with Fortran. 0 stands for none atom. indices starts from 1
    static void get_prim_indices_from_lmp();
};


template <typename CoordType>
void Se4pw<CoordType>::generate(
        CoordType* tilde_r,
        CoordType* tilde_r_deriv,
        CoordType* relative_coords,
        int inum,
        int* ilist,
        int* numneigh,
        int* firstneigh,   
        CoordType* x,
        int* types, // starts from 0.
        int ntypes, // starts from 0. e.g. 2
        int* num_neigh_atoms_lst,
        CoordType rcut,
        CoordType rcut_smooth)
{
    // Step 1.
    // Step 1.1. $\tilde{R}$ = (s(r_{ji}), x_{ji}, y_{ji}, z_{ji})
    //  = (tilde_s_value, tilde_x_value, tilde_y_value, tilde_z_value)
    CoordType tilde_s_value;
    CoordType tilde_x_value;
    CoordType tilde_y_value;
    CoordType tilde_z_value;
    CoordType distance_ji;
    CoordType distance_ji_recip;
    int center_atom_idx;
    int neigh_atom_idx;
    CoordType* center_cart_coords = (CoordType*)malloc(sizeof(CoordType) * 3);
    CoordType* neigh_cart_coords = (CoordType*)malloc(sizeof(CoordType) * 3);
    CoordType* diff_cart_coords = (CoordType*)malloc(sizeof(CoordType) * 3);
    SwitchFunc<CoordType> switch_func(rcut, rcut_smooth);

    int* nstart_idxs = (int*)malloc(sizeof(int) * ntypes);
    for (int ii=0; ii<ntypes; ii++)
        nstart_idxs[ii] = 0;
    for (int ii=0; ii<ntypes; ii++)
        for (int jj=0; jj<ii; jj++)
            nstart_idxs[ii] += num_neigh_atoms_lst[jj];
    int* nloop_idxs = (int*)malloc(sizeof(int) * ntypes);

    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<ntypes; ii++)
        tot_num_neigh_atoms += num_neigh_atoms_lst[ii];

    memset(tilde_r, 0, sizeof(CoordType)*inum*tot_num_neigh_atoms*4);
    memset(tilde_r_deriv, 0, sizeof(CoordType)*inum*tot_num_neigh_atoms*4*3);
    memset(relative_coords, 0, sizeof(CoordType)*inum*tot_num_neigh_atoms*3);
    
    // Step 2. Populate `tilde_s/x/y/z`
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        center_cart_coords[0] = x[center_atom_idx*3 + 0];
        center_cart_coords[1] = x[center_atom_idx*3 + 1];
        center_cart_coords[2] = x[center_atom_idx*3 + 2];
        for (int jj=0; jj<ntypes; jj++)
            nloop_idxs[jj] = 0;
            
        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_atom_idx = firstneigh[ii*tot_num_neigh_atoms+jj];
            
            // Step 3.1.1. 计算计算 1/r (`distance_ji_recip`), s(r_ji) (`tilde_s_value`)
            neigh_cart_coords[0] = x[neigh_atom_idx*3 + 0];
            neigh_cart_coords[1] = x[neigh_atom_idx*3 + 1];
            neigh_cart_coords[2] = x[neigh_atom_idx*3 + 2];
            for (int kk=0; kk<3; kk++)
                diff_cart_coords[kk] = neigh_cart_coords[kk] - center_cart_coords[kk];
            distance_ji = vec3Operation::norm(diff_cart_coords);
            distance_ji_recip = recip<CoordType>(distance_ji);
            tilde_s_value = smooth_func(distance_ji, rcut, rcut_smooth);

            // Step 3.1.2. 计算 `x_ji_s`, `y_ji_s`, `z_ji_s` 
            tilde_x_value = tilde_s_value * distance_ji_recip * diff_cart_coords[0];
            tilde_y_value = tilde_s_value * distance_ji_recip * diff_cart_coords[1];
            tilde_z_value = tilde_s_value * distance_ji_recip * diff_cart_coords[2];
            
            // Step 3.1.3. Assignment
            int kk = types[neigh_atom_idx]; // Note: So `types` must starts from 0.
            //for (kk=0; kk<ntypes; kk++) 
            //   if (types[neigh_atom_idx] == kk)
            //       break;

            tilde_r[0 + (nstart_idxs[kk]+nloop_idxs[kk])*4 + ii*tot_num_neigh_atoms*4] = tilde_s_value;
            tilde_r[1 + (nstart_idxs[kk]+nloop_idxs[kk])*4 + ii*tot_num_neigh_atoms*4] = tilde_x_value;
            tilde_r[2 + (nstart_idxs[kk]+nloop_idxs[kk])*4 + ii*tot_num_neigh_atoms*4] = tilde_y_value;
            tilde_r[3 + (nstart_idxs[kk]+nloop_idxs[kk])*4 + ii*tot_num_neigh_atoms*4] = tilde_z_value;

            /*
                1. smooth func = s(r) = \frac{1}{r} \cdot switch_func
                2. s(r) = \frac{1}{r} \cdot switch_func -- 需要分步求导
            */
            // Step 3.2.1. s(r) = switchFunc(r) * $\frac{1}{r}$
            tilde_r_deriv[0 + 0*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                switch_func.get_result(distance_ji) * diff_cart_coords[0] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coords[0] * std::pow(distance_ji_recip, 2)
            );
            tilde_r_deriv[1 + 0*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                switch_func.get_result(distance_ji) * diff_cart_coords[1] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coords[1] * std::pow(distance_ji_recip, 2)
            );
            tilde_r_deriv[2 + 0*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                switch_func.get_result(distance_ji) * diff_cart_coords[2] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coords[2] * std::pow(distance_ji_recip, 2)
            );

            // Step 3.2.2. s(r)x/r = switchFunc(r) * $\frac{x}{r^2}$
            tilde_r_deriv[0 + 1*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * std::pow(diff_cart_coords[0], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coords[0], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            tilde_r_deriv[1 + 1*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * diff_cart_coords[0] * diff_cart_coords[1] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coords[0] * diff_cart_coords[1] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            tilde_r_deriv[2 + 1*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * diff_cart_coords[0] * diff_cart_coords[2] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coords[0] * diff_cart_coords[2] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );

            // Step 3.2.3. s(r)y/r = switchFunc(r) * $\frac{y}{r^2}$
            tilde_r_deriv[0 + 2*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * diff_cart_coords[1] * diff_cart_coords[0] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coords[1] * diff_cart_coords[0] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            tilde_r_deriv[1 + 2*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * std::pow(diff_cart_coords[1], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coords[1], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            tilde_r_deriv[2 + 2*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * diff_cart_coords[1] * diff_cart_coords[2] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coords[1] * diff_cart_coords[2] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );

            // Step 3.2.4. s(r)z/r = switchFunc(r) * $\frac{z}{r^2}$
            tilde_r_deriv[0 + 3*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * diff_cart_coords[2] * diff_cart_coords[0] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coords[2] * diff_cart_coords[0] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            tilde_r_deriv[1 + 3*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * diff_cart_coords[2] * diff_cart_coords[1] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coords[2] * diff_cart_coords[1] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            tilde_r_deriv[2 + 3*3 + (nstart_idxs[kk]+nloop_idxs[kk])*4*3 + ii*tot_num_neigh_atoms*4*3] = (
                2 * std::pow(diff_cart_coords[2], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coords[2], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );

            // Step 3.3.1. 
            relative_coords[0 + (nstart_idxs[kk]+nloop_idxs[kk])*3 + ii*tot_num_neigh_atoms*3] = diff_cart_coords[0];
            relative_coords[1 + (nstart_idxs[kk]+nloop_idxs[kk])*3 + ii*tot_num_neigh_atoms*3] = diff_cart_coords[1];
            relative_coords[2 + (nstart_idxs[kk]+nloop_idxs[kk])*3 + ii*tot_num_neigh_atoms*3] = diff_cart_coords[2];

            nloop_idxs[kk]++;
        }
    }


    // Step. Free Memory
    free(center_cart_coords);
    free(neigh_cart_coords);
    free(diff_cart_coords);
    free(nstart_idxs);
    free(nloop_idxs);
}




template <typename CoordType>
void Se4pw<CoordType>::get_prim_indices_from_matersdk(
        int* prim_indices,
        int inum,
        int* ilist,
        int* numneigh,
        int* firstneigh,
        int* types,
        int ntypes,
        int* num_neigh_atoms_lst)
{
    // Step 1. 
    int center_atom_idx;
    int neigh_atom_idx;
    int* nstart_idxs = (int*)malloc(sizeof(int) * ntypes);
    for (int ii=0; ii<ntypes; ii++)
        nstart_idxs[ii] = 0;
    for (int ii=0; ii<ntypes; ii++)
        for (int jj=0; jj<ii; jj++)
            nstart_idxs[ii] += num_neigh_atoms_lst[jj];
    int* nloop_idxs = (int*)malloc(sizeof(int) * ntypes);
    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<ntypes; ii++)
        tot_num_neigh_atoms += num_neigh_atoms_lst[ii];
    memset(prim_indices, -1, sizeof(int)*inum*tot_num_neigh_atoms);

    // Step 2. Populate `prim_indices`
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        for (int jj=0; jj<ntypes; jj++)
            nloop_idxs[jj] = 0;

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_atom_idx = firstneigh[ii*tot_num_neigh_atoms + jj];
            
            int kk = types[neigh_atom_idx];
            
            prim_indices[(nstart_idxs[kk]+nloop_idxs[kk]) + ii*tot_num_neigh_atoms] = neigh_atom_idx % inum;
            nloop_idxs[kk]++;
        }
    }

    // Step . Free memory
    free(nstart_idxs);
    free(nloop_idxs);
}


};
};


#endif