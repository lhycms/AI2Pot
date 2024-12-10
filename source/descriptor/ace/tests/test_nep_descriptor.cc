#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>

#include "../../../nblist/include/neighborList.h"
#include "../include/sinlm.h"
#include "../include/nep_descriptor.h"


class PseudoWignerTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite() {
        std::cout << "PseudoWignerTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "PseudoWignerTest (TestSuite) is tearing down...\n";;
    }

    void SetUp() override {
    }

    void TearDown() override {
    }
};  // class : PseudoWignerTest


class NepDescriptorTest : public ::testing::Test
{
protected:
    int num_descriptors_r;
    int num_descriptors_a;
    int num_l;
    int l_max;
    int num_s_a;
    double lambda_val;
    double rmax_r, rmin_r;
    int n_r_max, n_r_basis;
    int max_body;
    double rmax_a, rmin_a;
    int n_a_max, n_a_basis;
    int l_3b_max;
    int l_4b_max;
    int l_5b_max;

    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];
    int nghost;
    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> neighbor_list;
    int inum;
    int *ilist;
    int *numneigh;
    int *firstneigh;
    double *rcs;
    int ntypes;
    int *types;
    int umax_num_neighs;

    double *s_val_r;
    double *s_val_r_delta;
    double *s_val_a;
    double *s_val_a_delta;
    double *s_der2xyz_r;
    double *s_der2xyz_r_delta;
    double *s_der2xyz_a;
    double *s_der2xyz_a_delta;
    double *s_der2coeffs_r;
    double *s_der2coeffs_r_delta;
    double *s_der2coeffs_a;
    double *s_der2coeffs_a_delta;
    double delta;

    double *coeffs_r;
    double *coeffs_r_delta;
    double *coeffs_a;
    double *coeffs_a_delta;


    static void SetUpTestSuite() {
        std::cout << "NepDescriptorTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NepDescriptorTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        max_body = 5;
        rmax_r = 5.0;
        rmin_r = 2.0;
        n_r_max = 10;
        n_r_basis = 8;
        rmax_a = 5.0;
        rmin_a = 2.0;
        n_a_max = 8;
        n_a_basis = 6;
        l_3b_max = 4;
        l_4b_max = 2;
        l_5b_max = 1;
        
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
        // 42: 0; 16: 1
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

        structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = ai2pot::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, false);

        ntypes = 2;
        umax_num_neighs = 19;
        inum = neighbor_list.get_num_center_atoms();
        ilist = (int*)malloc(inum * sizeof(int));
        numneigh = (int*)malloc(inum * sizeof(int));
        firstneigh = (int*)malloc(inum * umax_num_neighs * sizeof(int));
        rcs = (double*)malloc(inum * umax_num_neighs * 3 * sizeof(double));
        types = (int*)malloc(inum * sizeof(int));
        neighbor_list.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neighs);

        delta = 0.001;
        s_val_r = (double*)malloc(n_r_max * sizeof(double));
        memset(s_val_r, 0, n_r_max * sizeof(double));
        s_val_a = (double*)malloc(n_a_max * 24 * sizeof(double));
        memset(s_val_a, 0, n_a_max * 24 * sizeof(double));
        s_der2coeffs_r = (double*)malloc(n_r_max * ntypes * ntypes * n_r_basis * sizeof(double));
        memset(s_der2coeffs_r, 0, n_r_max * ntypes * ntypes * n_r_basis * sizeof(double));
        s_der2coeffs_a = (double*)malloc(n_a_max * 24 * ntypes * ntypes * n_a_basis * sizeof(double));
        memset(s_der2coeffs_a, 0, n_a_max * 24 * ntypes * ntypes * n_a_basis * sizeof(double));
        s_der2xyz_r = (double*)malloc(n_r_max * umax_num_neighs * 3 * sizeof(double));
        memset(s_der2xyz_r, 0, n_r_max * umax_num_neighs * 3 * sizeof(double));
        s_der2xyz_a = (double*)malloc(n_a_max * 24 * umax_num_neighs * 3 * sizeof(double));
        memset(s_der2xyz_a, 0, n_a_max * 24 * umax_num_neighs * 3 * sizeof(double));
        s_val_r_delta = (double*)malloc(n_r_max * sizeof(double));
        memset(s_val_r_delta, 0, n_r_max * sizeof(double));
        s_val_a_delta = (double*)malloc(n_a_max * 24 * sizeof(double));
        memset(s_val_a_delta, 0, n_a_max * 24 * sizeof(double));
        s_der2coeffs_r_delta = (double*)malloc(n_r_max * ntypes * ntypes * n_r_basis * sizeof(double));
        memset(s_der2coeffs_r_delta, 0, n_r_max * ntypes * ntypes * n_r_basis * sizeof(double));
        s_der2coeffs_a_delta = (double*)malloc(n_a_max * 24 * ntypes * ntypes * n_a_basis * sizeof(double));
        memset(s_der2coeffs_a_delta, 0, n_a_max * 24 * ntypes * ntypes * n_a_basis * sizeof(double));
        s_der2xyz_r_delta = (double*)malloc(n_r_max * umax_num_neighs * 3 * sizeof(double));
        memset(s_der2xyz_r_delta, 0, n_r_max * umax_num_neighs * 3 * sizeof(double));
        s_der2xyz_a_delta = (double*)malloc(n_a_max * 24 * umax_num_neighs * 3 * sizeof(double));
        memset(s_der2xyz_a_delta, 0, n_a_max * 24 * umax_num_neighs * 3 * sizeof(double));

        coeffs_r = (double*)malloc(ntypes * ntypes * n_r_max * n_r_basis * sizeof(double));
        coeffs_a = (double*)malloc(ntypes * ntypes * n_a_max * n_a_basis * sizeof(double));
        for (int ii=0; ii<ntypes * ntypes * n_r_max * n_r_basis; ii++)
            coeffs_r[ii] = 1 + 0.03 * ii;
        for (int ii=0; ii<ntypes * ntypes * n_a_max * n_a_basis; ii++)
            coeffs_a[ii] = 1 + 0.04 * ii;
        coeffs_r_delta = (double*)malloc(ntypes * ntypes * n_r_max * n_r_basis * sizeof(double));
        coeffs_a_delta = (double*)malloc(ntypes * ntypes * n_a_max * n_a_basis * sizeof(double));
    }

    void TearDown() override {
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);
    }
};  // class : NepDescriptorTest




TEST_F(PseudoWignerTest, init) {
    ai2pot::ace::PseudoWigner<double> pwigner;
    printf("1. Value of C3B[%d] = %g\n", 0, pwigner.C3B()[0]);
}


TEST_F(NepDescriptorTest, find_hyperparams)
{
    ai2pot::ace::NepDescriptor<double>::find_hyperparams(
        num_descriptors_r,
        num_descriptors_a,
        num_l,
        l_max,
        num_s_a,
        max_body,
        n_r_max,
        n_a_max,
        l_3b_max,
        l_4b_max,
        l_5b_max);

printf("1. num_r_descriptors = %d\n", num_descriptors_r);
printf("2. num_a_descriptors = %d\n", num_descriptors_a);
printf("3. num_l = %d\n", num_l);
printf("4. max_l = %d\n", l_max);
printf("5. num_s_a = %d\n", num_s_a);
}


TEST_F(NepDescriptorTest, accum_3b_descriptor_one) {
    ai2pot::ace::NepDescriptor<double>::find_hyperparams(
        num_descriptors_r,
        num_descriptors_a,
        num_l,
        l_max,
        num_s_a,
        max_body,
        n_r_max,
        n_a_max,
        l_3b_max,
        l_4b_max,
        l_5b_max);

    // ai2pot::ace::NepDescriptor<double>::find_descriptor_val_der_atom();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
