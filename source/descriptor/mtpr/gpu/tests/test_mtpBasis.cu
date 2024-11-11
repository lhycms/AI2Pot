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
    double h_switch_func_val;
    double d_switch_func_val;
    double h_switch_func_der2r;
    double d_switch_func_der2r;

    int chebyshev_size;
    double *h_rb_chebyshev_vals;
    double *d_rb_chebyshev_vals;
    double *h_rb_chebyshev_ders2r;
    double *d_rb_chebyshev_ders2r;

    double *h_rq_chebyshev_vals;
    double *d_rq_chebyshev_vals;
    double *h_rq_chebyshev_ders2r;
    double *d_rq_chebyshev_ders2r;

    static void SetUpTestSuite() {
        std::cout << "BasisUtilitiesTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "BasisUtilitiesTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        chebyshev_size = 8;
        h_rb_chebyshev_vals = (double*)malloc(chebyshev_size * sizeof(double));
        CHECK( cudaMalloc((void**)&d_rb_chebyshev_vals, chebyshev_size * sizeof(double)) );
        h_rb_chebyshev_ders2r = (double*)malloc(chebyshev_size * sizeof(double));
        CHECK( cudaMalloc((void**)&d_rb_chebyshev_ders2r, chebyshev_size * sizeof(double)) );

        h_rq_chebyshev_vals = (double*)malloc(chebyshev_size * sizeof(double));
        CHECK( cudaMalloc((void**)&d_rq_chebyshev_vals, chebyshev_size * sizeof(double)) );
        h_rq_chebyshev_ders2r = (double*)malloc(chebyshev_size * sizeof(double));
        CHECK( cudaMalloc((void**)&d_rq_chebyshev_ders2r, chebyshev_size * sizeof(double)) );
    }

    void TearDown() override {
        free(h_rb_chebyshev_vals);
        CHECK( cudaFree(d_rb_chebyshev_vals) );
        free(h_rb_chebyshev_ders2r);
        CHECK( cudaFree(d_rb_chebyshev_ders2r) );

        free(h_rq_chebyshev_vals);
        CHECK( cudaFree(d_rq_chebyshev_vals) );
        free(h_rq_chebyshev_ders2r);
        CHECK( cudaFree(d_rq_chebyshev_ders2r) );
    }
};  // class : BasisUtilitiesTest


TEST_F(BasisUtilitesTest, switch_func_test) {

}


TEST_F(BasisUtilitesTest, rb_chebyshev_test) {

}


TEST_F(BasisUtilitesTest, rq_chebyshev_test) {

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}