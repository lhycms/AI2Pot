#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include "../include/sinlm.h"
#include "../include/radialBasis.h"


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
        delta = 0.0001;
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

    static void SetUpTestSuite() {
        std::cout << "SinlmTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "SinlmTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        lambda_val = 5.0;
        rmax_r = 8.0;
        rmin_r = 2.0;
        n_r_max = 10;
        n_r_basis = 8;
        max_body = 5;
        rmax_a = 6.0;
        rmin_a = 2.0;
        n_a_max = 8;
        n_a_basis = 8;
        l_3b_max = 4;
        ptr_gn_r = new ai2pot::ace::Gn<double>(n_r_basis, rmax_r, rmin_r, lambda_val);
        ptr_gn_a = new ai2pot::ace::Gn<double>(n_a_basis, rmax_a, rmin_a, lambda_val);
    }

    void TearDown() override {
        delete ptr_gn_r;
        delete ptr_gn_a;
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


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
