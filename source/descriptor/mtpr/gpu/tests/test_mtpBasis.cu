#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#include "../include/mtp_utilities.cuh"
#include "../include/mtpBasis.cuh"


class BasisUtilitesTest : public ::testing::Test 
{
protected:
    double rmax;
    double rmin;
    double distance_ij;

    double h_switch_func_val;
    double h_switch_func_der2r;

    int chebyshev_size;
    double *h_rb_chebyshev_vals;
    double *h_rb_chebyshev_ders2r;

    double *h_rq_chebyshev_vals;
    double *h_rq_chebyshev_ders2r;

    static void SetUpTestSuite() {
        std::cout << "BasisUtilitiesTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "BasisUtilitiesTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rmax = 5.0;
        rmin = 2.0;
        distance_ij = 3.14;

        chebyshev_size = 8;
        h_rb_chebyshev_vals = (double*)malloc(chebyshev_size * sizeof(double));
        h_rb_chebyshev_ders2r = (double*)malloc(chebyshev_size * sizeof(double));

        h_rq_chebyshev_vals = (double*)malloc(chebyshev_size * sizeof(double));
        h_rq_chebyshev_ders2r = (double*)malloc(chebyshev_size * sizeof(double));
    }

    void TearDown() override {
        free(h_rb_chebyshev_vals);
        free(h_rb_chebyshev_ders2r);

        free(h_rq_chebyshev_vals);
        free(h_rq_chebyshev_ders2r);
    }
};  // class : BasisUtilitiesTest


TEST_F(BasisUtilitesTest, switch_func_test) {
    distance_ij = 3.14;
    find_switch_func(h_switch_func_val,
                     h_switch_func_der2r,
                     rmax,
                     rmin, 
                     distance_ij);
printf("BasisUtilitiesTest.switch_func_test:\n");
printf("\t1. h_switch_func_val = %g\n", h_switch_func_val);
printf("\t2. h_switch_func_der2r = %g\n", h_switch_func_der2r);
}


TEST_F(BasisUtilitesTest, rb_chebyshev_test) {
    distance_ij = 3.14;
    find_rb_chebyshev(h_rb_chebyshev_vals,
                      h_rb_chebyshev_ders2r,
                      chebyshev_size,
                      rmax,
                      rmin,
                      distance_ij);
printf("BasisUtilitiesTest.rb_chebyshev_test:\n");
printf("\t1. rb_chebyshev_vals = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rb_chebyshev_vals[ii]);
printf("\n");
printf("\t1. rb_chebyshev_der2r = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rb_chebyshev_ders2r[ii]);
printf("\n");
}


TEST_F(BasisUtilitesTest, rq_chebyshev_test) {
    distance_ij = 3.14;
    find_rq_chebyshev(h_rq_chebyshev_vals,
                      h_rq_chebyshev_ders2r,
                      chebyshev_size,
                      rmax,
                      rmin,
                      distance_ij);
printf("BasisUtilitiesTest.rq_chebyshev_test:\n");
printf("\t1. rq_chebyshev_vals = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rq_chebyshev_vals[ii]);
printf("\n");
printf("\t1. rq_chebyshev_der2r = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rq_chebyshev_ders2r[ii]);
printf("\n");
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}