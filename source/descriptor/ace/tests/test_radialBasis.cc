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


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
