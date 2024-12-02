#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include "../include/sinlm.h"
#include "../include/radialBasis.h"
#include "../../../nblist/include/neighborList.h"


class BlmTest : public ::testing::Test {
protected:
    double val1, val2;
    double *rc1, *rc2;
    double *der2xyz1, *der2xyz2;
    double delta;

    static void SetUpTestSuite() {
        std::cout << "BlmTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "BlmTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rc1 = (double*)malloc(sizeof(double) * 3);
        rc1[0] = 1.37429;
        rc1[1] = 3.98653;
        rc1[2] = 4.09821;
        rc2 = (double*)malloc(sizeof(double) * 3);
        der2xyz1 = (double*)malloc(sizeof(double) * 3);
        der2xyz2 = (double*)malloc(sizeof(double) * 3);
        delta = 0.001;
    }

    void TearDown() override {
        free(rc1);
        free(rc2);
        free(der2xyz1);
        free(der2xyz2);
    }
};  // class : BlmTest


class SinlmTest : public ::testing::Test {
protected:
    double lambda_val;
    double rmax_r, rmin_r;
    int n_r_max, n_r_basis;
    int max_body;
    double rmax_a, rmin_a;
    int n_a_max, n_a_basis;
    int l_3b_max;
    ai2pot::ace::Gn<double> *ptr_gn_r;
    ai2pot::ace::Gn<double> *ptr_gn_a;

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
        std::cout << "SinlmTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "SinlmTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        lambda_val = 5.0;
        rmax_r = 5.0;
        rmin_r = 2.0;
        n_r_max = 10;
        n_r_basis = 8;
        max_body = 5;
        rmax_a = 5.0;
        rmin_a = 2.0;
        n_a_max = 8;
        n_a_basis = 6;
        l_3b_max = 4;
        ptr_gn_r = new ai2pot::ace::Gn<double>(n_r_basis, rmax_r, rmin_r, lambda_val);
        ptr_gn_a = new ai2pot::ace::Gn<double>(n_a_basis, rmax_a, rmin_a, lambda_val);

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
        neighbor_list = ai2pot::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, true);

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
        delete ptr_gn_r;
        delete ptr_gn_a;
            
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);

        free(s_val_r);
        free(s_val_r_delta);
        free(s_val_a);
        free(s_val_a_delta);
        free(s_der2coeffs_r);
        free(s_der2coeffs_r_delta);
        free(s_der2coeffs_a);
        free(s_der2coeffs_a_delta);
        free(s_der2xyz_r);
        free(s_der2xyz_r_delta);
        free(s_der2xyz_a);
        free(s_der2xyz_a_delta);

        free(coeffs_r);
        free(coeffs_r_delta);
        free(coeffs_a);
        free(coeffs_a_delta);
    }
};  // class : SinlmTest




TEST_F(BlmTest, b10_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b10(val1, der2xyz1, rc1);
    ai2pot::ace::b10(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b10_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b10(val1, der2xyz1, rc1);
    ai2pot::ace::b10(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b10_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b10(val1, der2xyz1, rc1);
    ai2pot::ace::b10(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b11_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b11(val1, der2xyz1, rc1);
    ai2pot::ace::b11(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b11_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b11(val1, der2xyz1, rc1);
    ai2pot::ace::b11(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b11_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b11(val1, der2xyz1, rc1);
    ai2pot::ace::b11(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b12_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b12(val1, der2xyz1, rc1);
    ai2pot::ace::b12(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b12_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b12(val1, der2xyz1, rc1);
    ai2pot::ace::b12(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b12_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b12(val1, der2xyz1, rc1);
    ai2pot::ace::b12(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b20_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b20(val1, der2xyz1, rc1);
    ai2pot::ace::b20(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b20_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b20(val1, der2xyz1, rc1);
    ai2pot::ace::b20(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b20_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b20(val1, der2xyz1, rc1);
    ai2pot::ace::b20(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b21_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b21(val1, der2xyz1, rc1);
    ai2pot::ace::b21(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b21_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b21(val1, der2xyz1, rc1);
    ai2pot::ace::b21(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b21_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b21(val1, der2xyz1, rc1);
    ai2pot::ace::b21(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b22_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b22(val1, der2xyz1, rc1);
    ai2pot::ace::b22(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b22_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b22(val1, der2xyz1, rc1);
    ai2pot::ace::b22(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b22_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b22(val1, der2xyz1, rc1);
    ai2pot::ace::b22(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b23_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b23(val1, der2xyz1, rc1);
    ai2pot::ace::b23(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b23_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b23(val1, der2xyz1, rc1);
    ai2pot::ace::b23(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b23_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b23(val1, der2xyz1, rc1);
    ai2pot::ace::b23(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b24_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b24(val1, der2xyz1, rc1);
    ai2pot::ace::b24(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b24_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b24(val1, der2xyz1, rc1);
    ai2pot::ace::b24(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b24_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b24(val1, der2xyz1, rc1);
    ai2pot::ace::b24(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2xz= %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b30_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b30(val1, der2xyz1, rc1);
    ai2pot::ace::b30(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b30_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b30(val1, der2xyz1, rc1);
    ai2pot::ace::b30(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b30_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b30(val1, der2xyz1, rc1);
    ai2pot::ace::b30(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b31_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b31(val1, der2xyz1, rc1);
    ai2pot::ace::b31(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b31_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b31(val1, der2xyz1, rc1);
    ai2pot::ace::b31(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b31_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b31(val1, der2xyz1, rc1);
    ai2pot::ace::b31(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b32_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b32(val1, der2xyz1, rc1);
    ai2pot::ace::b32(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b32_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b32(val1, der2xyz1, rc1);
    ai2pot::ace::b32(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b32_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b32(val1, der2xyz1, rc1);
    ai2pot::ace::b32(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b33_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b33(val1, der2xyz1, rc1);
    ai2pot::ace::b33(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b33_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b33(val1, der2xyz1, rc1);
    ai2pot::ace::b33(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b33_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b33(val1, der2xyz1, rc1);
    ai2pot::ace::b33(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b34_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b34(val1, der2xyz1, rc1);
    ai2pot::ace::b34(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b34_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b34(val1, der2xyz1, rc1);
    ai2pot::ace::b34(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b34_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b34(val1, der2xyz1, rc1);
    ai2pot::ace::b34(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b35_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b35(val1, der2xyz1, rc1);
    ai2pot::ace::b35(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b35_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b35(val1, der2xyz1, rc1);
    ai2pot::ace::b35(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b35_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b35(val1, der2xyz1, rc1);
    ai2pot::ace::b35(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b36_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b36(val1, der2xyz1, rc1);
    ai2pot::ace::b36(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b36_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b36(val1, der2xyz1, rc1);
    ai2pot::ace::b36(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b36_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b36(val1, der2xyz1, rc1);
    ai2pot::ace::b36(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b40_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b40(val1, der2xyz1, rc1);
    ai2pot::ace::b40(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b40_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b40(val1, der2xyz1, rc1);
    ai2pot::ace::b40(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b40_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b40(val1, der2xyz1, rc1);
    ai2pot::ace::b40(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b41_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b41(val1, der2xyz1, rc1);
    ai2pot::ace::b41(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b41_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b41(val1, der2xyz1, rc1);
    ai2pot::ace::b41(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b41_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b41(val1, der2xyz1, rc1);
    ai2pot::ace::b41(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b42_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b42(val1, der2xyz1, rc1);
    ai2pot::ace::b42(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b42_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b42(val1, der2xyz1, rc1);
    ai2pot::ace::b42(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b42_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b42(val1, der2xyz1, rc1);
    ai2pot::ace::b42(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b43_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b43(val1, der2xyz1, rc1);
    ai2pot::ace::b43(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b43_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b43(val1, der2xyz1, rc1);
    ai2pot::ace::b43(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b43_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b43(val1, der2xyz1, rc1);
    ai2pot::ace::b43(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b44_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b44(val1, der2xyz1, rc1);
    ai2pot::ace::b44(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b44_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b44(val1, der2xyz1, rc1);
    ai2pot::ace::b44(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b44_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b44(val1, der2xyz1, rc1);
    ai2pot::ace::b44(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b45_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b45(val1, der2xyz1, rc1);
    ai2pot::ace::b45(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b45_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b45(val1, der2xyz1, rc1);
    ai2pot::ace::b45(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b45_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b45(val1, der2xyz1, rc1);
    ai2pot::ace::b45(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b46_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b46(val1, der2xyz1, rc1);
    ai2pot::ace::b46(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b46_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b46(val1, der2xyz1, rc1);
    ai2pot::ace::b46(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b46_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b46(val1, der2xyz1, rc1);
    ai2pot::ace::b46(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b47_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b47(val1, der2xyz1, rc1);
    ai2pot::ace::b47(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b47_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b47(val1, der2xyz1, rc1);
    ai2pot::ace::b47(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b47_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b47(val1, der2xyz1, rc1);
    ai2pot::ace::b47(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b48_der2x) {
    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    ai2pot::ace::b48(val1, der2xyz1, rc1);
    ai2pot::ace::b48(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2x = %g\n", der2xyz1[0]);
    printf("\t2. Numerical der2x = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b48_der2y) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    ai2pot::ace::b48(val1, der2xyz1, rc1);
    ai2pot::ace::b48(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2y = %g\n", der2xyz1[1]);
    printf("\t2. Numerical der2y = %g\n", (val2 - val1) / delta);
}

TEST_F(BlmTest, b48_der2z) {
    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    ai2pot::ace::b48(val1, der2xyz1, rc1);
    ai2pot::ace::b48(val2, der2xyz2, rc2);
    printf("\t1. Analytic der2z = %g\n", der2xyz1[2]);
    printf("\t2. Numerical der2z = %g\n", (val2 - val1) / delta);
}


TEST_F(SinlmTest, default_constructor)
{
    ai2pot::ace::Sinlm<double> sinlm;
    ASSERT_EQ(sinlm.n_r_max(), 0);
    ASSERT_EQ(sinlm.n_r_basis(), 0);
    ASSERT_EQ(sinlm.n_a_max(), 0);
    ASSERT_EQ(sinlm.n_a_basis(), 0);
    ASSERT_EQ(sinlm.l_3b_max(), 0);
    ASSERT_EQ(sinlm.num_s_a(), 0);
}


TEST_F(SinlmTest, constructor_1)
{
    ai2pot::ace::Sinlm<double> sinlm(lambda_val, 
                                     rmax_r, 
                                     rmin_r, 
                                     n_r_max, 
                                     n_r_basis, 
                                     max_body, 
                                     rmax_a, 
                                     rmin_a, 
                                     n_a_max, 
                                     n_a_basis, 
                                     l_3b_max);
    ASSERT_EQ(sinlm.n_r_max(), n_r_max);
    ASSERT_EQ(sinlm.n_r_basis(), n_r_basis);
    ASSERT_EQ(sinlm.max_body(), max_body);
    ASSERT_EQ(sinlm.n_a_max(), n_a_max);
    ASSERT_EQ(sinlm.n_a_basis(), n_a_basis);
    ASSERT_EQ(sinlm.l_3b_max(), l_3b_max);
    ASSERT_EQ(sinlm.num_s_a(), 24);
}


TEST_F(SinlmTest, constructor_2)
{
    ai2pot::ace::Gn<double> gn_r(n_r_basis, rmax_r, rmin_r, lambda_val);
    ai2pot::ace::Gn<double> gn_a(n_a_basis, rmax_a, rmin_a, lambda_val);
    ai2pot::ace::Sinlm<double> sinlm(&gn_r, &gn_a, n_r_max, max_body, n_a_max, l_3b_max);
    ASSERT_EQ(sinlm.n_r_max(), n_r_max);
    ASSERT_EQ(sinlm.n_r_basis(), n_r_basis);
    ASSERT_EQ(sinlm.max_body(), max_body);
    ASSERT_EQ(sinlm.n_a_max(), n_a_max);
    ASSERT_EQ(sinlm.n_a_basis(), n_a_basis);
    ASSERT_EQ(sinlm.l_3b_max(), l_3b_max);
    ASSERT_EQ(sinlm.num_s_a(), 24);
}

TEST_F(SinlmTest, copy_constructor)
{
    ai2pot::ace::Gn<double> gn_r(n_r_basis, rmax_r, rmin_r, lambda_val);
    ai2pot::ace::Gn<double> gn_a(n_a_basis, rmax_a, rmin_a, lambda_val);
    ai2pot::ace::Sinlm<double> sinlm_1(&gn_r, &gn_a, n_r_max, max_body, n_a_max, l_3b_max);
    ai2pot::ace::Sinlm<double> sinlm_2(sinlm_1);
    ASSERT_EQ(sinlm_1.ptr_gn_r()->chebyshev_size(), sinlm_2.ptr_gn_r()->chebyshev_size());
    ASSERT_EQ(sinlm_1.ptr_gn_a()->chebyshev_size(), sinlm_2.ptr_gn_a()->chebyshev_size());
    ASSERT_EQ(sinlm_1.n_r_max(), sinlm_2.n_r_max());
    ASSERT_EQ(sinlm_1.n_r_basis(), sinlm_2.n_r_basis());
    ASSERT_EQ(sinlm_1.max_body(), sinlm_2.max_body());
    ASSERT_EQ(sinlm_1.n_a_max(), sinlm_2.n_a_max());
    ASSERT_EQ(sinlm_1.n_a_basis(), sinlm_2.n_a_basis());
    ASSERT_EQ(sinlm_1.l_3b_max(), sinlm_2.l_3b_max());
    ASSERT_EQ(sinlm_1.num_s_a(), sinlm_2.num_s_a());
}

TEST_F(SinlmTest, copy_constructor_move)
{
    ai2pot::ace::Gn<double> gn_r(n_r_basis, rmax_r, rmin_r, lambda_val);
    ai2pot::ace::Gn<double> gn_a(n_a_basis, rmax_a, rmin_a, lambda_val);
    ai2pot::ace::Sinlm<double> sinlm_1(&gn_r, &gn_a, n_r_max, max_body, n_a_max, l_3b_max);
    ai2pot::ace::Sinlm<double> sinlm_2(std::move(sinlm_1));
    ASSERT_EQ(sinlm_1.ptr_gn_r(), nullptr);
    ASSERT_EQ(sinlm_1.ptr_gn_a(), nullptr);
}

TEST_F(SinlmTest, assignment_operator)
{
    ai2pot::ace::Gn<double> gn_r(n_r_basis, rmax_r, rmin_r, lambda_val);
    ai2pot::ace::Gn<double> gn_a(n_a_basis, rmax_a, rmin_a, lambda_val);
    ai2pot::ace::Sinlm<double> sinlm_1(&gn_r, &gn_a, n_r_max, max_body, n_a_max, l_3b_max);
    ai2pot::ace::Sinlm<double> sinlm_2(&gn_r, &gn_a, n_r_max+1, max_body, n_a_max, l_3b_max);
    sinlm_2 = sinlm_1;
    ASSERT_EQ(sinlm_1.ptr_gn_r()->chebyshev_size(), sinlm_2.ptr_gn_r()->chebyshev_size());
    ASSERT_EQ(sinlm_1.ptr_gn_a()->chebyshev_size(), sinlm_2.ptr_gn_a()->chebyshev_size());
    ASSERT_EQ(sinlm_1.n_r_max(), sinlm_2.n_r_max());
    ASSERT_EQ(sinlm_1.n_r_basis(), sinlm_2.n_r_basis());
    ASSERT_EQ(sinlm_1.max_body(), sinlm_2.max_body());
    ASSERT_EQ(sinlm_1.n_a_max(), sinlm_2.n_a_max());
    ASSERT_EQ(sinlm_1.n_a_basis(), sinlm_2.n_a_basis());
    ASSERT_EQ(sinlm_1.l_3b_max(), sinlm_2.l_3b_max());
    ASSERT_EQ(sinlm_1.num_s_a(), sinlm_2.num_s_a());
}

TEST_F(SinlmTest, assignment_operator_move)
{
    ai2pot::ace::Gn<double> gn_r(n_r_basis, rmax_r, rmin_r, lambda_val);
    ai2pot::ace::Gn<double> gn_a(n_a_basis, rmax_a, rmin_a, lambda_val);
    ai2pot::ace::Sinlm<double> sinlm_1(&gn_r, &gn_a, n_r_max, max_body, n_a_max, l_3b_max);
    ai2pot::ace::Sinlm<double> sinlm_2(&gn_r, &gn_a, n_r_max+1, max_body, n_a_max, l_3b_max);
    sinlm_2 = std::move(sinlm_1);
    ASSERT_EQ(sinlm_1.ptr_gn_r(), nullptr);
    ASSERT_EQ(sinlm_1.ptr_gn_a(), nullptr);
}


TEST_F(SinlmTest, find_val_der_r_one_der2xyz) 
{
    int center_idx_modify = 0;
    int neigh_idx_modify = 1;
    int direction_modify = 0;
    int kk_modify = 0;
    int cidx = ilist[center_idx_modify];
    int nidx = firstneigh[center_idx_modify * umax_num_neighs + neigh_idx_modify];
    int itype = types[cidx];
    int jtype = types[nidx];

    ai2pot::ace::Sinlm<double> sinlm(lambda_val,
                                     rmax_r,
                                     rmin_r,
                                     n_r_max,
                                     n_r_basis,
                                     max_body,
                                     rmax_a,
                                     rmin_a,
                                     n_a_max,
                                     n_a_basis,
                                     l_3b_max);
    sinlm.accum_val_der_r_one(s_val_r[kk_modify],
                              &s_der2xyz_r[kk_modify*umax_num_neighs*3 + neigh_idx_modify*3 + 0],
                              &s_der2coeffs_r[kk_modify*ntypes*ntypes*n_r_basis + (itype*ntypes+jtype)*n_r_basis + 0],
                              &rcs[center_idx_modify*umax_num_neighs*3 + neigh_idx_modify*3 + 0],
                              &coeffs_r[(itype*ntypes+jtype)*n_r_max*n_r_basis + kk_modify*n_r_basis + 0]);
    
    rcs[center_idx_modify*umax_num_neighs*3 + neigh_idx_modify*3 + direction_modify] += delta;
    ai2pot::ace::Sinlm<double> sinlm_delta(lambda_val,
                                        rmax_r,
                                        rmin_r,
                                        n_r_max,
                                        n_r_basis,
                                        max_body,
                                        rmax_a,
                                        rmin_a,
                                        n_a_max,
                                        n_a_basis,
                                        l_3b_max);
    sinlm_delta.accum_val_der_r_one(s_val_r_delta[kk_modify],
                                    &s_der2xyz_r[kk_modify*umax_num_neighs*3 + neigh_idx_modify*3 + 0],
                                    &s_der2coeffs_r[kk_modify*ntypes*ntypes*n_r_basis + (itype*ntypes+jtype)*n_r_basis + 0],
                                    &rcs[center_idx_modify*umax_num_neighs*3 + neigh_idx_modify*3 + 0],
                                    &coeffs_r[(itype*ntypes+jtype)*n_r_max*n_r_basis + kk_modify*n_r_basis + 0]);
printf("1.1. Sinlm_one value = %g\n", s_val_r[kk_modify]);
printf("1.2. Sinlm_one value = %g\n", s_val_r_delta[kk_modify]);
printf("2.1. Sinlm_one derivative w.r.t. rcs[%d][%d][%d] calculated with custom code = %g\n", 
                                                                center_idx_modify, 
                                                                neigh_idx_modify, 
                                                                direction_modify, 
                                                                s_der2xyz_r[kk_modify*umax_num_neighs*3 + neigh_idx_modify*3 + direction_modify]);
printf("2.2. Sinlm_one derivative w.r.t. rcs[%d][%d][%d] calculated with finite difference method = %g\n",
                                                                center_idx_modify, 
                                                                neigh_idx_modify, 
                                                                direction_modify, 
                                                                (s_val_r_delta[kk_modify] - s_val_r[kk_modify]) / delta);
}





int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
