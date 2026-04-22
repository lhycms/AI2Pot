#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../include/nep_utilities.h"
#include "../include/nep_cpu_launcher.h"
#include "../include/nepLoss_cpu_launcher.h"


typedef double real;


class NepCPULauncher : public ::testing::Test
{
protected:
    int n_radial_basis;
    int n_angular_basis;
    real rmax_radial;
    real rmax_angular;
    int chebyshev_size;
    int num_neurons;
    int ntypes;
    int num_descriptors;
    int l_max;

    real *coeffs;
    real *w0;
    real *b0;
    real *w1;
    real *type_bias;
    real rmax;
    real rmin;

    int num_atoms;
    real basis_vectors[3][3];
    int atomic_numbers[12];
    real frac_coords[12][3];
    real rcut;
    real bin_size_xyz[3];
    bool pbc_xyz[3];
    int type_map[2];

    int batch_size;
    int natoms_pad;
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


    real *betot;
    real *betot_;
    real (*bforce)[3];
    real (*bforce_)[3];
    real *bvirial;
    real *bvirial_;

    // Loss derivatives
    real *bloss_der2coeffs;
    real *bloss_der2w0;
    real *bloss_der2b0;
    real *bloss_der2w1;
    real *bloss_der2type_bias;

    // Loss
    real e_weight;
    real f_weight;
    real v_weight;
    real *betot_dft;
    real (*bforce_dft)[3];
    real *bvirial_dft;

    // Descriptors
    real *bdescriptors;

    // q factors
    real *q_scaler;

    static void SetUpTestSuite() {
        std::cout << "NepCPULauncher (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NepCPULauncher (TestSuite) is tearding down...\n";
    }

    void SetUp() override {
        n_radial_basis = 6;
        n_angular_basis = 4;
        l_max = 3;
        chebyshev_size = 8;
        num_neurons = 30;
        ntypes = 2;
        num_descriptors = ai2pot::nep::NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);

        rmax_radial = 5.0;
        rmax_angular = 4.0;

        coeffs = (real*)malloc(sizeof(real) * ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size);
        for (int ii=0; ii<ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size; ii++)
            coeffs[ii] = 0.01 + 0.001 * ii;
        w0 = (real*)malloc(sizeof(real) * ntypes * num_neurons * num_descriptors);
        for (int ii=0; ii<ntypes * num_neurons * num_descriptors; ii++)
            w0[ii] = 0.1 + 0.001 * ii;
        b0 = (real*)malloc(sizeof(real)*ntypes*num_neurons);
        for (int ii=0; ii<ntypes*num_neurons; ii++)
            b0[ii] = 0.1 + 0.001 * ii;
        w1 = (real*)malloc(sizeof(real) * ntypes * num_neurons);
        for (int ii=0; ii<ntypes * num_neurons; ii++)
            w1[ii] = 0.1 + 0.001 * ii;
        type_bias = (real*)malloc(sizeof(real) * ntypes);
        type_bias[0] = -0.1;
        type_bias[1] = -0.2;

        rmax = 5.0;
        rmin = 0.0;

        // Establish neighbor list
        num_atoms = 12;
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

        rcut = 5.0;
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
        batch_size = 1;
        natoms_pad = 12;
        umax_num_neigh_atoms = 20;
        nghost = 0;
        binum = (int*)malloc(sizeof(int) * batch_size);
        bilist = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        bnumneigh = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        bfirstneigh = (int*)malloc(sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms);
        brcs = (real*)malloc(sizeof(real) * batch_size * natoms_pad * umax_num_neigh_atoms * 3);
        btypes = (int*)malloc(sizeof(int) * batch_size * (natoms_pad + nghost));
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

        betot = (real *)malloc(sizeof(real) * batch_size);
        memset(betot, 0.0, batch_size);
        bforce = (real (*)[3])malloc(sizeof(real) * batch_size * (natoms_pad+nghost) * 3);
        memset(bforce, 0.0, sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        bvirial = (real*)malloc(sizeof(real) * batch_size * 9);
        memset(bvirial, 0.0, sizeof(real)*batch_size*9);

        betot_ = (real *)malloc(sizeof(real) * batch_size);
        memset(betot_, 0.0, batch_size);
        bforce_ = (real (*)[3])malloc(sizeof(real) * batch_size * (natoms_pad+nghost) * 3);
        memset(bforce_, 0.0, sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        bvirial_ = (real*)malloc(sizeof(real) * batch_size * 9);
        memset(bvirial_, 0.0, sizeof(real) * batch_size * 9);

        // Loss derivative
        bloss_der2coeffs = (real*)malloc(sizeof(real) * batch_size * ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size);
        memset(bloss_der2coeffs, 0, sizeof(real) * batch_size * ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size);
        bloss_der2w0 = (real*)malloc(sizeof(real) * batch_size * ntypes * num_neurons * num_descriptors);
        memset(bloss_der2w0, 0, sizeof(real) * batch_size * ntypes * num_neurons * num_descriptors);
        bloss_der2b0 = (real*)malloc(sizeof(real) * batch_size * ntypes * num_neurons);
        memset(bloss_der2b0, 0, sizeof(real) * batch_size * ntypes * num_neurons);
        bloss_der2w1 = (real*)malloc(sizeof(real) * batch_size * ntypes * num_neurons);
        memset(bloss_der2w1, 0, sizeof(real) * batch_size * ntypes * num_neurons);
        bloss_der2type_bias = (real*)malloc(sizeof(real) * batch_size * ntypes);
        memset(bloss_der2type_bias, 0, sizeof(real) * batch_size * ntypes);

        // Loss
        e_weight = 0.1;
        f_weight = 0.2;
        v_weight = 0.3;
        betot_dft = (real*)malloc(sizeof(real) * batch_size);
        memset(betot_dft, 0, sizeof(real)*batch_size);
        bforce_dft = (real (*)[3])malloc(sizeof(real) * batch_size * natoms_pad * 3);
        memset(bforce_dft, 0, sizeof(real)*batch_size*natoms_pad*3);
        bvirial_dft = (real*)malloc(sizeof(real) * batch_size * 9);
        memset(bvirial_dft, 0, sizeof(real)*batch_size*9);

        // Descriptors
        bdescriptors = (real*)malloc(sizeof(real) * batch_size * natoms_pad * num_descriptors);
        memset(bdescriptors, 0, sizeof(real) * batch_size * natoms_pad * num_descriptors);

        // 
        q_scaler = (real*)malloc(sizeof(real) * num_descriptors);
        for (int ii=0; ii<num_descriptors; ii++) {
            q_scaler[ii] = 0.67 + 0.05 * ii;
        }
    }

    void TearDown() override {
        free(binum);
        free(bilist);
        free(bnumneigh);
        free(bfirstneigh);
        free(brcs);
        free(btypes);
    
        free(coeffs);
        free(w0);
        free(b0);
        free(w1);
        free(type_bias);

        free(betot);
        free(bforce);
        free(bvirial);
        free(betot_);
        free(bforce_);
        free(bvirial_);

        free(bloss_der2coeffs);
        free(bloss_der2w0);
        free(bloss_der2b0);
        free(bloss_der2w1);
        free(bloss_der2type_bias);
        free(betot_dft);
        free(bforce_dft);
        free(bvirial_dft);

        free(bdescriptors);
        free(q_scaler);
    }
};  // class : NepCPULauncher


TEST_F(NepCPULauncher, find_ef_cpu_launcher) {
    int center_idx_modify = 1;
    int direction1_idx_modify = 2;

    ai2pot::nep::find_ef_cpu_launcher<real>(
        betot,
        bforce,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        coeffs,
        w0,
        b0,
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
        rmax_radial,
        rmax_angular,
        q_scaler);

printf("1.1. energy = %.15lf\n", betot[0]);
printf("1.2. force[%d][%d] calculated by custom code = %.15lf\n", center_idx_modify, direction1_idx_modify, bforce[center_idx_modify][direction1_idx_modify]);
printf("1.3. force=\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, bforce[ii][0], bforce[ii][1], bforce[ii][2]);
}


TEST_F(NepCPULauncher, find_efv_cpu_launcher) {
    int center_idx_modify = 1;
    int direction1_idx_modify = 2;

    ai2pot::nep::find_efv_cpu_launcher<real>(
        betot,
        bforce,
        bvirial,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        coeffs,
        w0,
        b0,
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
        rmax_radial,
        rmax_angular,
        q_scaler);

printf("1.1. energy = %.15lf\n", betot[0]);
printf("1.2. force[%d][%d] calculated by custom code = %.15lf\n", center_idx_modify, direction1_idx_modify, bforce[center_idx_modify][direction1_idx_modify]);
printf("1.3. force=\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, bforce[ii][0], bforce[ii][1], bforce[ii][2]);
printf("1.4. virial:\n\t[");
for (int aa=0; aa<3; aa++)
    for (int bb=0; bb<3; bb++)
        printf("%.8lf, ", bvirial[aa*3+bb]);
printf("]\n");
}


TEST_F(NepCPULauncher, find_ef_loss_backward_cpu_launcher)
{
    ai2pot::nep::find_ef_cpu_launcher<real>(
        betot,
        bforce,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        coeffs,
        w0,
        b0,
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
        rmax_radial,
        rmax_angular,
        q_scaler);

    ai2pot::nep::find_ef_loss_backward_cpu_launcher<real>(
        bloss_der2coeffs,
        bloss_der2w0,
        bloss_der2b0,
        bloss_der2w1,
        bloss_der2type_bias,
        e_weight,
        f_weight,
        betot,
        betot_dft,
        bforce,
        bforce_dft,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        coeffs,
        w0,
        b0,
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
        rmax_radial,
        rmax_angular,
        q_scaler);

printf("***+++ %d\n", ntypes*ntypes*(n_radial_basis+n_angular_basis)*chebyshev_size);
printf("1. bloss_der2coeffs:\n");
for (int ii=0; ii<ntypes*ntypes*(n_radial_basis+n_angular_basis)*chebyshev_size; ii++)
    printf("%.15f, ", bloss_der2coeffs[ii]);
printf("\n\n");

printf("2. bloss_der2w0:\n");
for (int ii=0; ii<ntypes*num_neurons*num_descriptors; ii++)
    printf("%.15f, ", bloss_der2w0[ii]);
printf("\n\n");

printf("3. bloss_der2w1:\n");
for (int ii=0; ii<ntypes*num_neurons; ii++)
    printf("%.15f, ", bloss_der2w1[ii]);
printf("\n\n");

printf("4. bloss_der2type_bias:\n");
for (int ii=0; ii<ntypes; ii++)
    printf("%.15f, ", bloss_der2type_bias[ii]);
printf("\n\n");
}


TEST_F(NepCPULauncher, find_descriptors_cpu_launcher)
{
    ai2pot::nep::find_descriptors_cpu_launcher<real>(
        bdescriptors,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        coeffs,
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
        rmax_radial,
        rmax_angular);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
