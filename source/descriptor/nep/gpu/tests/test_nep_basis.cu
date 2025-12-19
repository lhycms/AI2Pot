#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/nep_basis.cuh"


class NepBasisTest : public ::testing::Test 
{
protected:
    double distance_ij;
    double rmax;
    double rmin;
    double *switch_func_val_ptr;
    double *switch_func_der2r_ptr;

    int chebyshev_size;
    double *rb_chebyshev_vals;
    double *rb_chebyshev_ders2r;
    double *rq_chebyshev_vals;
    double *rq_chebyshev_ders2r;

    static void SetUpTestSuite() {
        std::cout << "NepBasisTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NepBasisTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rmax = 5.0;
        rmin = 2.0;

        switch_func_val_ptr = (double*)malloc(sizeof(double));
        memset(switch_func_val_ptr, 0, sizeof(double));
        switch_func_der2r_ptr = (double*)malloc(sizeof(double));
        memset(switch_func_der2r_ptr, 0, sizeof(double));

        chebyshev_size  = 8;
        rb_chebyshev_vals = (double*)malloc(sizeof(double) * chebyshev_size);
        memset(rb_chebyshev_vals, 0, sizeof(double)*chebyshev_size);
        rb_chebyshev_ders2r = (double*)malloc(sizeof(double) * chebyshev_size);
        memset(rb_chebyshev_ders2r, 0, sizeof(double)*chebyshev_size);

        rq_chebyshev_vals = (double*)malloc(sizeof(double) * chebyshev_size);
        memset(rq_chebyshev_vals, 0, sizeof(double)*chebyshev_size);
        rq_chebyshev_ders2r = (double*)malloc(sizeof(double) * chebyshev_size);
        memset(rq_chebyshev_ders2r, 0, sizeof(double)*chebyshev_size);
    }

    void TearDown() override {
        free(switch_func_val_ptr);
        free(switch_func_der2r_ptr);

        free(rb_chebyshev_vals);
        free(rb_chebyshev_ders2r);
        free(rq_chebyshev_vals);
        free(rq_chebyshev_ders2r);
    }
};  // class : NepBasisTest


TEST_F(NepBasisTest, switch_func_test) {
    distance_ij = 3.14;

    ai2pot::nep::find_switch_func(
        *switch_func_val_ptr,
        *switch_func_der2r_ptr,
        rmax,
        rmin,
        distance_ij);
printf("NepBasisTest.switch_func_test:\n");
printf("\t1. switch_func_val = %g\n", *switch_func_val_ptr);
printf("\t2. switch_func_der2r = %g\n", *switch_func_der2r_ptr);
}


TEST_F(NepBasisTest, rb_chebyshev_test) {
    distance_ij = 3.14;

    ai2pot::nep::find_rb_chebyshev(
        rb_chebyshev_vals,
        rb_chebyshev_ders2r,
        chebyshev_size,
        rmax,
        rmin,
        distance_ij);

printf("NepBasisTest.rb_chebyshev_test:\n");
printf("\t1. rb_chebyshev_vals = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", rb_chebyshev_vals[ii]);
printf("\n");
printf("\t1. rb_chebyshev_der2r = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", rb_chebyshev_ders2r[ii]);
printf("\n");
}


TEST_F(NepBasisTest, rq_chebyshev_test) {
    distance_ij = 3.14;

    ai2pot::nep::find_rq_chebyshev(
        rq_chebyshev_vals,
        rq_chebyshev_ders2r,
        chebyshev_size,
        rmax,
        rmin,
        distance_ij);

printf("NepBasisTest.rq_chebyshev_test:\n");
printf("\t1. rq_chebyshev_vals = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", rq_chebyshev_vals[ii]);
printf("\n");
printf("\t1. rq_chebyshev_der2r = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", rq_chebyshev_ders2r[ii]);
printf("\n");
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
