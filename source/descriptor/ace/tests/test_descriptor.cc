#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include "../include/descriptor.h"


class Descriptor2BTest : public ::testing::Test
{
protected:
    int size;
    double rmax;
    double rmin;
    double lambda_val;

    double val;
    double *ptr_der2coeffs;
    double der2r;
    double distance_ij;
    double *ptr_coeffs;

    double delta;
    double val_delta;
    double *ptr_der2coeffs_delta;
    double der2r_delta;
    double distance_ij_delta;
    double *ptr_coeffs_delta;

    static void SetUpTestSuite() {
        std::cout << "Descriptor2BTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "Descriptor2BTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        size = 8;
        rmax = 5.0;
        rmin = 2.0;
        lambda_val = 5.0;
        
        val = 0.0;
        ptr_der2coeffs = (double*)malloc(size * sizeof(double));
        der2r = 0.0;
        distance_ij = 3.14;
        ptr_coeffs = (double*)malloc(size * sizeof(double));
        for (int ii=0; ii<size; ii++)
            ptr_coeffs[ii] = 1.0 + 0.1 * ii;

        delta = 0.001;
        val_delta = 0.0;
        ptr_der2coeffs_delta = (double*)malloc(size * sizeof(double));
        der2r_delta = 0.0;
        ptr_coeffs_delta = (double*)malloc(size * sizeof(double));
    }

    void TearDown() override {
        free(ptr_der2coeffs);
        free(ptr_coeffs);
        free(ptr_der2coeffs_delta);
        free(ptr_coeffs_delta);
    }
};  // class : Descriptor2BTest

TEST_F(Descriptor2BTest, default_constructor) 
{
    ai2pot::ace::Descriptor2B<double> descriptor_2b;
    ASSERT_EQ(descriptor_2b.chebyshev_size(), 0);
    ASSERT_EQ(descriptor_2b.ptr_rq(), nullptr);
}

TEST_F(Descriptor2BTest, constructor_1) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b(size, rmax, rmin, lambda_val);
    ASSERT_EQ(descriptor_2b.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(Descriptor2BTest, constructor_2) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev(size, rmax, rmin, lambda_val);
    ai2pot::ace::Descriptor2B<double> descriptor_2b(&rq_cchebyshev);
    ASSERT_EQ(descriptor_2b.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(Descriptor2BTest, copy_constructor) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Descriptor2B<double> descriptor_2b_2(descriptor_2b_1);
    ASSERT_EQ(descriptor_2b_2.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(Descriptor2BTest, copy_constructor_move) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Descriptor2B<double> descriptor_2b_2(std::move(descriptor_2b_1));
    ASSERT_EQ(descriptor_2b_1.chebyshev_size(), 0);
    ASSERT_EQ(descriptor_2b_1.ptr_rq(), nullptr);
}

TEST_F(Descriptor2BTest, assignment_operator) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Descriptor2B<double> descriptor_2b_2(size+1, rmax, rmin, lambda_val);
    descriptor_2b_2 = descriptor_2b_1;
    ASSERT_EQ(descriptor_2b_1.chebyshev_size(), descriptor_2b_2.chebyshev_size());
    ASSERT_EQ(descriptor_2b_1.ptr_rq()->size(), descriptor_2b_2.ptr_rq()->size());
    ASSERT_EQ(descriptor_2b_1.ptr_rq()->rmax(), descriptor_2b_2.ptr_rq()->rmax());
    ASSERT_EQ(descriptor_2b_1.ptr_rq()->rmin(), descriptor_2b_2.ptr_rq()->rmin());
    ASSERT_EQ(descriptor_2b_1.ptr_rq()->lambda_val(), descriptor_2b_2.ptr_rq()->lambda_val());
    for (int ii=0; ii<descriptor_2b_1.chebyshev_size(); ii++) {
        ASSERT_EQ(descriptor_2b_1.ptr_rq()->vals()[ii], descriptor_2b_2.ptr_rq()->vals()[ii]);
        ASSERT_EQ(descriptor_2b_1.ptr_rq()->ders2r()[ii], descriptor_2b_2.ptr_rq()->ders2r()[ii]);
    }
}

TEST_F(Descriptor2BTest, assignment_operator_move) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Descriptor2B<double> descriptor_2b_2(size+1, rmax, rmin, lambda_val);
    descriptor_2b_2 = std::move(descriptor_2b_1);
    ASSERT_EQ(descriptor_2b_1.chebyshev_size(), 0);
    ASSERT_EQ(descriptor_2b_1.ptr_rq(), nullptr);
}

TEST_F(Descriptor2BTest, der_accuracy) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b(size, rmax, rmin, lambda_val);
    descriptor_2b.find_val_der2r(&val, 
                                 ptr_der2coeffs, 
                                 &der2r, 
                                 distance_ij, 
                                 ptr_coeffs);

printf("1. Derivative wrt. distance_ij:\n");
    distance_ij_delta = distance_ij + delta;
    for (int ii=0; ii<size; ii++)
        ptr_coeffs_delta[ii] = ptr_coeffs[ii];
    descriptor_2b.find_val_der2r(&val_delta,
                                 ptr_der2coeffs_delta,
                                 &der2r_delta,
                                 distance_ij_delta,
                                 ptr_coeffs_delta);
printf("\t1.1. Analytic derivative wrt. distance_ij = %g\n", der2r);
printf("\t1.2. Numerical derivative wrt. distance_ij = %g\n", (val_delta - val) / delta);


printf("2. Derivative wrt. coeffs:\n");
    distance_ij_delta = distance_ij;
    for (int ii=0; ii<size; ii++)
        ptr_coeffs_delta[ii] = ptr_coeffs[ii];
printf("\t2.1. Analytic derivate wrt. coeffs: ");
for (int ii=0; ii<descriptor_2b.chebyshev_size(); ii++) {
    printf("%g, ", ptr_der2coeffs[ii]);
}
printf("\n");
printf("\t2.2. Numerical derivate wrt. coeffs: ");
for (int ii=0; ii<descriptor_2b.chebyshev_size(); ii++) {
    ptr_coeffs_delta[ii] += delta;
    descriptor_2b.find_val_der2r(&val_delta,
                                 ptr_der2coeffs_delta,
                                 &der2r_delta,
                                 distance_ij_delta,
                                 ptr_coeffs_delta);
    printf("%g, ", (val_delta - val) / delta);
    ptr_coeffs_delta[ii] -= delta;
}
printf("\n");

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}