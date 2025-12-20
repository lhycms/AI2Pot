#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

#include "../include/nep.cuh"
#include "../include/nepLoss.cuh"
#include "../include/nep_utilities.cuh"
#include "../../../../nblist/include/structure.h"
#include "../../../../nblist/include/neighborList.h"

typedef double real;


class NepTest : public ::testing::Test
{
protected:
    // Parameters
    int chebyshev_size;
    int n_radial_basis;
    int n_angular_basis;
    int l_max;
    int num_neurons;
    int num_descriptors;
    int ntypes;
    int batch_size;
    int natoms_pad;
    
    real *coeffs;
    real *w0;
    real *w1;
    real *type_bias;
    real *q_scaler;
    real rmax;
    real rmin;

    // Structure && Nblist
    int num_atoms;
    real basis_vectors[3][3];
    int atomic_numbers[12];
    real frac_coords[12][3];
    real rcut;
    real bin_size_xyz[3];
    bool pbc_xyz[3];
    int type_map[2];

    int umax_num_neigh_atoms;
    int *binum;
    int *bilist;
    int *bnumneigh;
    int *bfirstneigh;
    real *brcs;
    int *btypes;
    int nghost;
    ai2pot::Structure<real> structure;
    ai2pot::NeighborList<real> neighbor_list;

    // Target
    real *betot;
    real *betot_;
    real (*bforce)[3];
    real (*bforce_)[3];
    real *bvirial;
    real *bvirial_;
    real *bdescriptors;


    // Loss
    real e_weight;
    real f_weight;
    real v_weight;
    real *bloss_der2coeffs;
    real *bloss_der2w0;
    real *bloss_der2w1;
    real *bloss_der2type_bias;

    real *betot_dft;
    real (*bforce_dft)[3];
    real *bvirial_dft;


    static void SetUpTestSuite() {
        std::cout << "NepTest (TestSuite) is setting up..." << std::endl;
    }

    static void TearDownTestSuite() {
        std::cout << "NepTest (TestSuite) is tearing down..." << std::endl;
    }

    void SetUp() override {
        // Parameters
        chebyshev_size = 8;
        n_radial_basis = 6;
        n_angular_basis = 4;
        l_max = 3;
        num_neurons = 30;
        num_descriptors = ai2pot::nep::NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
        ntypes = 2;
        batch_size = 1;
        natoms_pad = 12;

        coeffs = (real*)malloc(sizeof(real) * ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size);
        for (int ii=0; ii<ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size; ii++)
            coeffs[ii] = 0.01 + 0.001 * ii;
        w0 = (real*)malloc(sizeof(real) * ntypes * num_neurons * num_descriptors);
        for (int ii=0; ii<ntypes * num_neurons * num_descriptors; ii++)
            w0[ii] = 0.1 + 0.001 * ii;
        w1 = (real*)malloc(sizeof(real) * ntypes * num_neurons);
        for (int ii=0; ii<ntypes * num_neurons; ii++)
            w1[ii] = 0.1 + 0.001 * ii;
        type_bias = (real*)malloc(sizeof(real) * ntypes);
        type_bias[0] = -0.1;
        type_bias[1] = -0.2;
        q_scaler = (real*)malloc(sizeof(real) * num_descriptors);
        for (int k=0; k<num_descriptors; k++)
            q_scaler[k] = 1.0;

        // Structure && Nblist
        num_atoms = 12;
        rmax = 5.0;
        rmin = 0.0;
        rcut = rmax;
        basis_vectors[0][0] = 3.1903157348;
        basis_vectors[0][1] = 5.5257885468;
        basis_vectors[0][2] = 0.0000000000;
        basis_vectors[1][0] = -6.3806307800;
        basis_vectors[1][1] = 0.0000000000;
        basis_vectors[1][2] = 0.0000000000;
        basis_vectors[2][0] = 0.0000000000;
        basis_vectors[2][1] = 0.0000000000;
        basis_vectors[2][2] = 23.1297687334;

        // 42: 0;  16: 1
        atomic_numbers[0] = 0;
        atomic_numbers[1] = 1;
        atomic_numbers[2] = 1;
        atomic_numbers[3] = 0;
        atomic_numbers[4] = 1;
        atomic_numbers[5] = 1;
        atomic_numbers[6] = 0;
        atomic_numbers[7] = 1;
        atomic_numbers[8] = 1;
        atomic_numbers[9] = 0; 
        atomic_numbers[10] = 1;
        atomic_numbers[11] = 1;

        frac_coords[0][0] = 0.333333333333;
        frac_coords[0][1] = 0.166666666667;
        frac_coords[0][2] = 0.500000000000;
        frac_coords[1][0] = 0.166666666667;
        frac_coords[1][1] = 0.333333333333;
        frac_coords[1][2] = 0.432343276548;
        frac_coords[2][0] = 0.166666666667;
        frac_coords[2][1] = 0.333333333333;
        frac_coords[2][2] = 0.567656723452;
        frac_coords[3][0] = 0.333333333333;
        frac_coords[3][1] = 0.666666666667;
        frac_coords[3][2] = 0.500000000000;
        frac_coords[4][0] = 0.166666666667;
        frac_coords[4][1] = 0.833333333333;
        frac_coords[4][2] = 0.432343276548;
        frac_coords[5][0] = 0.166666666667;
        frac_coords[5][1] = 0.833333333333;
        frac_coords[5][2] = 0.567656723452;
        frac_coords[6][0] = 0.833333333333;
        frac_coords[6][1] = 0.166666666667;
        frac_coords[6][2] = 0.500000000000;
        frac_coords[7][0] = 0.666666666667;
        frac_coords[7][1] = 0.333333333333;
        frac_coords[7][2] = 0.432343276548;
        frac_coords[8][0] = 0.666666666667;
        frac_coords[8][1] = 0.333333333333;
        frac_coords[8][2] = 0.567656723452;
        frac_coords[9][0] = 0.833333333333;
        frac_coords[9][1] = 0.666666666667;
        frac_coords[9][2] = 0.500000000000;
        frac_coords[10][0] = 0.666666666667;
        frac_coords[10][1] = 0.833333333333;
        frac_coords[10][2] = 0.432343276548;
        frac_coords[11][0] = 0.666666666667;
        frac_coords[11][1] = 0.833333333333;
        frac_coords[11][2] = 0.567656723452;

        bin_size_xyz[0] = 2.5;
        bin_size_xyz[1] = 2.5;
        bin_size_xyz[2] = 2.5;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;
        
        type_map[0] = 42;
        type_map[1] = 16;

        structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);
        umax_num_neigh_atoms = 20;
        binum = (int*)malloc(sizeof(int) * batch_size);
        binum[0] = 12;
        bilist = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        bnumneigh = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        bfirstneigh = (int*)malloc(sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms);
        brcs = (real*)malloc(sizeof(real) * batch_size * natoms_pad * umax_num_neigh_atoms * 3);
        btypes = (int*)malloc(sizeof(int) * batch_size * (natoms_pad+nghost));
        for (int bb=0; bb<batch_size; bb++)
            neighbor_list.find_info4mlff(
                binum[bb],
                &bilist[bb*natoms_pad],
                &bnumneigh[bb*natoms_pad],
                &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms],
                &brcs[bb*natoms_pad*umax_num_neigh_atoms*3],
                &btypes[bb*(natoms_pad+nghost)],
                nghost,
                umax_num_neigh_atoms);

        // Target
        betot = (real*)malloc(sizeof(real)*batch_size);
        memset(betot, 0.0, sizeof(real)*batch_size);
        bforce = (real (*)[3])malloc(sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        memset(bforce, 0.0, sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        bvirial = (real*)malloc(sizeof(real)*batch_size*9);
        memset(bvirial, 0.0, sizeof(real)*batch_size*9);

        betot_ = (real*)malloc(sizeof(real)*batch_size);
        memset(betot_, 0.0, sizeof(real)*batch_size);
        bforce_ = (real (*)[3])malloc(sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        memset(bforce_, 0.0, sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        bvirial_ = (real*)malloc(sizeof(real)*batch_size*9);
        memset(bvirial_, 0.0, sizeof(real)*batch_size*9);

        // Loss
        e_weight = 0.1;
        f_weight = 0.2;
        v_weight = 0.3;
        int num_coeffs = ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size;
        bloss_der2coeffs = (real*)malloc(sizeof(real)*batch_size*num_coeffs);
        memset(bloss_der2coeffs, 0, sizeof(real)*batch_size*num_coeffs);
        bloss_der2w0 = (real*)malloc(sizeof(real)*batch_size*ntypes*num_neurons*num_descriptors);
        memset(bloss_der2w0, 0, sizeof(real)*batch_size*ntypes*num_neurons*num_descriptors);
        bloss_der2w1 = (real*)malloc(sizeof(real)*batch_size*ntypes*num_neurons);
        memset(bloss_der2w1, 0, sizeof(real)*batch_size*ntypes*num_neurons);
        bloss_der2type_bias = (real*)malloc(sizeof(real)*batch_size*ntypes);
        memset(bloss_der2type_bias, 0, sizeof(real)*batch_size*ntypes);

        betot_dft = (real*)malloc(sizeof(real)*batch_size);
        memset(betot_dft, 0, sizeof(real)*batch_size);
        bforce_dft = (real (*)[3])malloc(sizeof(real)*batch_size*natoms_pad*3);
        memset(bforce_dft, 0, sizeof(real)*batch_size*natoms_pad*3);
        bvirial_dft = (real*)malloc(sizeof(real)*batch_size*9);
        memset(bvirial_dft, 0, sizeof(real)*batch_size*9);

        bdescriptors = (real*)malloc(sizeof(real) * batch_size * natoms_pad * num_descriptors);
        memset(bdescriptors, 0, sizeof(real) * batch_size * natoms_pad * num_descriptors);
    }

    void TearDown() override {
        // Parameters
        free(coeffs);
        free(w0);
        free(w1);
        free(type_bias);
        free(q_scaler);

        // Nblist
        free(binum);
        free(bilist);
        free(bnumneigh);
        free(bfirstneigh);
        free(brcs);
        free(btypes);

        // Target
        free(betot);
        free(bforce);
        free(bvirial);
        free(betot_);
        free(bforce_);
        free(bvirial_);

        // Loss
        free(bloss_der2coeffs);   
        free(bloss_der2w0);
        free(bloss_der2w1);
        free(bloss_der2type_bias);
        free(betot_dft);
        free(bforce_dft);
        free(bvirial_dft);
        free(bdescriptors);
    }
};  // class : NepTest


TEST_F(NepTest, find_ef_accuracy) {
    real delta = 1e-7;
    int center_idx_modify = 1;
    int direction_idx_modify = 2;

    ai2pot::nep::find_ef_launcher(
        betot,
        bforce,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        q_scaler);

    // *** delta
    real cart_coords[natoms_pad][3] = {0.};
    for (int ii=0; ii<natoms_pad; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction_idx_modify] += delta;
    structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);
    neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);

    for (int bb=0; bb<batch_size; bb++)
        neighbor_list.find_info4mlff(
            binum[bb],
            &bilist[bb*natoms_pad],
            &bnumneigh[bb*natoms_pad],
            &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms],
            &brcs[bb*natoms_pad*umax_num_neigh_atoms*3],
            &btypes[bb*(natoms_pad+nghost)],
            nghost,
            umax_num_neigh_atoms);

    ai2pot::nep::find_ef_launcher(
        betot_,
        bforce_,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        q_scaler);

printf("1.1. energy = %.15lf\n", betot[0]);
printf("1.1. bforce[%d][%d][%d] calculated by custom code = %.15lf\n", 0, center_idx_modify, direction_idx_modify, bforce[0*(natoms_pad+nghost) + center_idx_modify][direction_idx_modify]);
printf("1.2. energy = %.15lf\n", betot_[0]);
printf("1.2. bforce[%d][%d][%d] calculated by finite difference method = %.15lf\n", 0, center_idx_modify, direction_idx_modify, -(betot_[0] - betot[0]) / delta);
printf("\n\nfind_efv_launcher:\n");
printf("\t2.1. Energy = %.15f\n", betot[0]);
printf("\t2.2. Force =\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, bforce[0*(natoms_pad+nghost) + ii][0], bforce[0*(natoms_pad+nghost) + ii][1], bforce[0*(natoms_pad+nghost) + ii][2]);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
