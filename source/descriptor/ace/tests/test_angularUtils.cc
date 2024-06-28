#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include "../include/angularUtils.h"


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


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}