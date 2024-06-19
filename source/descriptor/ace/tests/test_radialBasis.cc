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


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
