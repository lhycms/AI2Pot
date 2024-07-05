#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include "../include/radialBasis.h"


class SwitchFunctionTest : public ::testing::Test
{
protected:
    double rmax;
    double rmin;
    double distance_ij;

    static void SetUpTestSuite() {
        std::cout << "SwitchFunctionTest (TestSuite) is setting up...\n";
    }   

    static void TearDownTestSuite() {
        std::cout << "SwitchFunctionTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rmax = 5.0;
        rmin = 2.0;
        distance_ij = 3.14;
    }

    void TearDown() override {
    }
};  // class : SwitchFunction


class RB_CChebyshevTest: public ::testing::Test
{
protected:
    int size;
    double rmax;
    double rmin;
    double lambda_val;
    double distance_ij;

    static void SetUpTestSuite() {
        std::cout << "RB_CChebyshevTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "RB_CChebyshevTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        size = 8;
        rmax = 5.0;
        rmin = 2.0;
        lambda_val = 5;
        distance_ij = 3.14;
    }

    void TearDown() override {
    }
};  // class : RB_CChebyshevTest


class RQ_CChebyshevTest : public ::testing::Test
{
protected:
    int size;
    double rmax;
    double rmin;
    double lambda_val;
    double distance_ij;

    static void SetUpTestSuite() {
        std::cout << "RQ_CChebyshevTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "RQ_CChebyshevTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        size = 8;
        rmax = 5.0;
        rmin = 2.0;
        lambda_val = 5.0;
        distance_ij = 3.14;
    }

    void TearDown() override {
    }
};  // class : RQ_CChebyshevTest


class GnTest : public ::testing::Test
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
        std::cout << "GnTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "GnTest (TestSuite) is tearing down...\n";
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
};  // class : GnTest


TEST_F(SwitchFunctionTest, default_constructor) {
    ai2pot::ace::SwitchFunction<double> switch_func;

    ASSERT_DOUBLE_EQ(switch_func.rmin(), 0.0);
    ASSERT_DOUBLE_EQ(switch_func.rmax(), 0.0);
}

TEST_F(SwitchFunctionTest, constructor) {
    ai2pot::ace::SwitchFunction<double> switch_func(rmax, rmin);
    
    ASSERT_DOUBLE_EQ(switch_func.rmin(), rmin);
    ASSERT_DOUBLE_EQ(switch_func.rmax(), rmax);
}

TEST_F(SwitchFunctionTest, val) {
    ai2pot::ace::SwitchFunction<double> switch_func(rmax, rmin);

    ASSERT_DOUBLE_EQ(switch_func.val(rmin), 1);
    ASSERT_DOUBLE_EQ(switch_func.val(rmax), 0);
}

TEST_F(SwitchFunctionTest, der2r_accuracy) {
    ai2pot::ace::SwitchFunction<double> switch_func(rmax, rmin);

    double deriv2r_val = ( switch_func.val(distance_ij+0.0001) - switch_func.val(distance_ij) ) / 0.0001;
    printf("\t1. Numerical derivative: %lf\n", deriv2r_val);
    printf("\t2. Analytical derivative: %lf\n", switch_func.der2r(distance_ij));
}


TEST_F(RB_CChebyshevTest, default_constructor) {
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev;
    ASSERT_EQ(rb_cchebyshev.rmax(), 0.0);
    ASSERT_EQ(rb_cchebyshev.rmin(), 0.0);
    ASSERT_EQ(rb_cchebyshev.size(), 0);
    ASSERT_EQ(rb_cchebyshev.lambda_val(), 0.0);
}

TEST_F(RB_CChebyshevTest, constructor_1) {
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev(size, rmax, rmin, lambda_val);
    ASSERT_EQ(rb_cchebyshev.size(), size);
    ASSERT_EQ(rb_cchebyshev.rmax(), rmax);
    ASSERT_EQ(rb_cchebyshev.rmin(), rmin);
    ASSERT_EQ(rb_cchebyshev.lambda_val(), lambda_val);
}

TEST_F(RB_CChebyshevTest, copy_constructor) {
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev_1(size, rmax, rmin, lambda_val);
    rb_cchebyshev_1.build(distance_ij);
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev_2(rb_cchebyshev_1);
    ASSERT_EQ(rb_cchebyshev_1.size(), rb_cchebyshev_2.size());
    ASSERT_EQ(rb_cchebyshev_1.rmax(), rb_cchebyshev_2.rmax());
    ASSERT_EQ(rb_cchebyshev_1.rmin(), rb_cchebyshev_2.rmin());
    ASSERT_EQ(rb_cchebyshev_1.lambda_val(), rb_cchebyshev_2.lambda_val());
    for (int ii=0; ii<rb_cchebyshev_1.size(); ii++) {
        ASSERT_EQ(rb_cchebyshev_1.vals()[ii], rb_cchebyshev_2.vals()[ii]);
        ASSERT_EQ(rb_cchebyshev_1.ders2vv()[ii], rb_cchebyshev_2.ders2vv()[ii]);
        ASSERT_EQ(rb_cchebyshev_1.ders2r()[ii], rb_cchebyshev_2.ders2r()[ii]);
    }
}

TEST_F(RB_CChebyshevTest, copy_constructor_move) {
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev_1(size, rmax, rmin, lambda_val);
    rb_cchebyshev_1.build(distance_ij);
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev_2(std::move(rb_cchebyshev_1));
    ASSERT_EQ(rb_cchebyshev_1.vals(), nullptr);
    ASSERT_EQ(rb_cchebyshev_1.ders2vv(), nullptr);
    ASSERT_EQ(rb_cchebyshev_1.ders2r(), nullptr);
}

TEST_F(RB_CChebyshevTest, assignment_operator) {
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev_1(size, rmax, rmin, lambda_val);
    rb_cchebyshev_1.build(distance_ij);
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev_2;
    rb_cchebyshev_2 = rb_cchebyshev_1;
    ASSERT_EQ(rb_cchebyshev_1.size(), rb_cchebyshev_2.size());
    ASSERT_EQ(rb_cchebyshev_1.rmax(), rb_cchebyshev_2.rmax());
    ASSERT_EQ(rb_cchebyshev_1.rmin(), rb_cchebyshev_2.rmin());
    ASSERT_EQ(rb_cchebyshev_1.lambda_val(), rb_cchebyshev_2.lambda_val());
    for (int ii=0; ii<rb_cchebyshev_1.size(); ii++) {
        ASSERT_EQ(rb_cchebyshev_1.vals()[ii], rb_cchebyshev_2.vals()[ii]);
        ASSERT_EQ(rb_cchebyshev_1.ders2vv()[ii], rb_cchebyshev_2.ders2vv()[ii]);
        ASSERT_EQ(rb_cchebyshev_1.ders2r()[ii], rb_cchebyshev_2.ders2r()[ii]);
    }
}

TEST_F(RB_CChebyshevTest, assignement_operator_move) {
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev_1(size, rmax, rmin, lambda_val);
    rb_cchebyshev_1.build(distance_ij);
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev_2(size+1, rmax, rmin, lambda_val);
    rb_cchebyshev_2 = std::move(rb_cchebyshev_1);
    ASSERT_EQ(rb_cchebyshev_1.vals(), nullptr);
    ASSERT_EQ(rb_cchebyshev_1.ders2vv(), nullptr);
    ASSERT_EQ(rb_cchebyshev_1.ders2r(), nullptr);
}

TEST_F(RB_CChebyshevTest, ders2r_accuracy) {
    ai2pot::ace::RB_CChebyshev<double> rb_cchebyshev(size, rmax, rmin, lambda_val);
    rb_cchebyshev.build(distance_ij);
    ai2pot::ace::RB_CChebyshev<double> delta_rb_cchebyshev(size, rmax, rmin, lambda_val);
    delta_rb_cchebyshev.build(distance_ij+0.001);

printf("1.1. Values of rb_cchebyshev:\n\t");
for (int ii=0; ii<rb_cchebyshev.size(); ii++) {
    printf("%lf, ", rb_cchebyshev.vals()[ii]);
}
printf("\n");

printf("1.2. Values of delta_rb_cchebyshev:\n\t");
for (int ii=0; ii<delta_rb_cchebyshev.size(); ii++) {
    printf("%lf, ", delta_rb_cchebyshev.vals()[ii]);
}
printf("\n");

printf("2.1. Analytic ders2r:\n\t");
for (int ii=0; ii<rb_cchebyshev.size(); ii++) {
    printf("%lf, ", rb_cchebyshev.ders2r()[ii]);
}
printf("\n");

printf("2.2. Numerical ders2r:\n\t");
for (int ii=0; ii<rb_cchebyshev.size(); ii++) {
    printf("%lf, ", (delta_rb_cchebyshev.vals()[ii] - rb_cchebyshev.vals()[ii]) / 0.001);
}
printf("\n");
}


TEST_F(RQ_CChebyshevTest, default_constructor) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev;
    ASSERT_EQ(rq_cchebyshev.size(), 0);
    ASSERT_EQ(rq_cchebyshev.rmax(), 0.0);
    ASSERT_EQ(rq_cchebyshev.rmin(), 0.0);
    ASSERT_EQ(rq_cchebyshev.lambda_val(), 0.0);
    ASSERT_EQ(rq_cchebyshev.vals(), nullptr);
    ASSERT_EQ(rq_cchebyshev.ders2r(), nullptr);
}

TEST_F(RQ_CChebyshevTest, constructor_1) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev(size, rmax, rmin, lambda_val);
    ASSERT_EQ(rq_cchebyshev.size(), size);
    ASSERT_EQ(rq_cchebyshev.rmax(), rmax);
    ASSERT_EQ(rq_cchebyshev.rmin(), rmin);
    ASSERT_EQ(rq_cchebyshev.lambda_val(), lambda_val);
}

TEST_F(RQ_CChebyshevTest, copy_constructor) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev_1(size, rmax, rmin, lambda_val);
    rq_cchebyshev_1.build(distance_ij);
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev_2(rq_cchebyshev_1);
    ASSERT_EQ(rq_cchebyshev_1.size(), rq_cchebyshev_2.size());
    ASSERT_EQ(rq_cchebyshev_1.rmax(), rq_cchebyshev_2.rmax());
    ASSERT_EQ(rq_cchebyshev_1.rmin(), rq_cchebyshev_2.rmin());
    ASSERT_EQ(rq_cchebyshev_1.lambda_val(), rq_cchebyshev_2.lambda_val());
    for (int ii=0; ii<rq_cchebyshev_1.size(); ii++) {
        ASSERT_EQ(rq_cchebyshev_1.vals()[ii], rq_cchebyshev_2.vals()[ii]);
        ASSERT_EQ(rq_cchebyshev_1.ders2r()[ii], rq_cchebyshev_2.ders2r()[ii]);
    }
}

TEST_F(RQ_CChebyshevTest, copy_constructor_move) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev_1(size, rmax, rmin, lambda_val);
    rq_cchebyshev_1.build(distance_ij);
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev_2(std::move(rq_cchebyshev_1));
    ASSERT_EQ(rq_cchebyshev_1.size(), 0);
    ASSERT_EQ(rq_cchebyshev_1.rmax(), 0.0);
    ASSERT_EQ(rq_cchebyshev_1.rmin(), 0.0);
    ASSERT_EQ(rq_cchebyshev_1.lambda_val(), 0.0);
    ASSERT_EQ(rq_cchebyshev_1.vals(), nullptr);
    ASSERT_EQ(rq_cchebyshev_1.ders2r(), nullptr);
}

TEST_F(RQ_CChebyshevTest, assignment_operator) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev_1(size, rmax, rmin, lambda_val);
    rq_cchebyshev_1.build(distance_ij);
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev_2;
    rq_cchebyshev_2 = rq_cchebyshev_1;
    ASSERT_EQ(rq_cchebyshev_1.size(), rq_cchebyshev_2.size());
    ASSERT_EQ(rq_cchebyshev_1.rmax(), rq_cchebyshev_2.rmax());
    ASSERT_EQ(rq_cchebyshev_1.rmin(), rq_cchebyshev_2.rmin());
    ASSERT_EQ(rq_cchebyshev_1.lambda_val(), rq_cchebyshev_2.lambda_val());
    for (int ii=0; ii<rq_cchebyshev_1.size(); ii++) {
        ASSERT_EQ(rq_cchebyshev_1.vals()[ii], rq_cchebyshev_2.vals()[ii]);
        ASSERT_EQ(rq_cchebyshev_1.ders2r()[ii], rq_cchebyshev_2.ders2r()[ii]);
    }   
}

TEST_F(RQ_CChebyshevTest, assignment_operator_move) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev_1(size, rmax, rmin, lambda_val);
    rq_cchebyshev_1.build(distance_ij);
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev_2;
    rq_cchebyshev_2 = std::move(rq_cchebyshev_1);
    ASSERT_EQ(rq_cchebyshev_1.size(), 0);
    ASSERT_EQ(rq_cchebyshev_1.rmax(), 0.0);
    ASSERT_EQ(rq_cchebyshev_1.rmin(), 0.0);
    ASSERT_EQ(rq_cchebyshev_1.lambda_val(), 0.0);
    ASSERT_EQ(rq_cchebyshev_1.vals(), nullptr);
    ASSERT_EQ(rq_cchebyshev_1.ders2r(), nullptr);
}

TEST_F(RQ_CChebyshevTest, ders2r_accuracy) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev(size, rmax, rmin, lambda_val);
    rq_cchebyshev.build(distance_ij);
    ai2pot::ace::RQ_CChebyshev<double> delta_rq_cchebyshev(size, rmax, rmin, lambda_val);
    delta_rq_cchebyshev.build(distance_ij+0.001);

printf("1.1. Value of rq_cchbyshev:\n\t");
for (int ii=0; ii<rq_cchebyshev.size(); ii++) {
    printf("%lf, ", rq_cchebyshev.vals()[ii]);
}
printf("\n");

printf("1.2. Value of delta_rq_chebyshev:\n\t");
for (int ii=0; ii<rq_cchebyshev.size(); ii++) {
    printf("%lf, ", delta_rq_cchebyshev.vals()[ii]);
}
printf("\n");

printf("2.1. Numerical derivatives:\n\t");
for (int ii=0; ii<rq_cchebyshev.size(); ii++) {
    printf("%lf, ", (delta_rq_cchebyshev.vals()[ii] - rq_cchebyshev.vals()[ii]) / 0.001);
}
printf("\n");

printf("2.2. Analytical derivatives:\n\t");
for (int ii=0; ii<rq_cchebyshev.size(); ii++) {
    printf("%lf, ", rq_cchebyshev.ders2r()[ii]);
}
printf("\n");
}

TEST_F(GnTest, default_constructor) 
{
    ai2pot::ace::Gn<double> descriptor_2b;
    ASSERT_EQ(descriptor_2b.chebyshev_size(), 0);
    ASSERT_EQ(descriptor_2b.ptr_rq(), nullptr);
}

TEST_F(GnTest, constructor_1) {
    ai2pot::ace::Gn<double> descriptor_2b(size, rmax, rmin, lambda_val);
    ASSERT_EQ(descriptor_2b.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(GnTest, constructor_2) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev(size, rmax, rmin, lambda_val);
    ai2pot::ace::Gn<double> descriptor_2b(&rq_cchebyshev);
    ASSERT_EQ(descriptor_2b.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(GnTest, copy_constructor) {
    ai2pot::ace::Gn<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Gn<double> descriptor_2b_2(descriptor_2b_1);
    ASSERT_EQ(descriptor_2b_2.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(GnTest, copy_constructor_move) {
    ai2pot::ace::Gn<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Gn<double> descriptor_2b_2(std::move(descriptor_2b_1));
    ASSERT_EQ(descriptor_2b_1.chebyshev_size(), 0);
    ASSERT_EQ(descriptor_2b_1.ptr_rq(), nullptr);
}

TEST_F(GnTest, assignment_operator) {
    ai2pot::ace::Gn<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Gn<double> descriptor_2b_2(size+1, rmax, rmin, lambda_val);
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

TEST_F(GnTest, assignment_operator_move) {
    ai2pot::ace::Gn<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Gn<double> descriptor_2b_2(size+1, rmax, rmin, lambda_val);
    descriptor_2b_2 = std::move(descriptor_2b_1);
    ASSERT_EQ(descriptor_2b_1.chebyshev_size(), 0);
    ASSERT_EQ(descriptor_2b_1.ptr_rq(), nullptr);
}

TEST_F(GnTest, der_accuracy) {
    ai2pot::ace::Gn<double> descriptor_2b(size, rmax, rmin, lambda_val);
    descriptor_2b.find_val_der2r(val,
                                 ptr_der2coeffs,
                                 &der2r,
                                 distance_ij, 
                                 ptr_coeffs);

printf("1. Derivative wrt. distance_ij:\n");
    distance_ij_delta = distance_ij + delta;
    for (int ii=0; ii<size; ii++)
        ptr_coeffs_delta[ii] = ptr_coeffs[ii];
    descriptor_2b.find_val_der2r(val_delta,
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
    descriptor_2b.find_val_der2r(val_delta,
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
