#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../include/mtpMRadial_de.h"
#include "../../../core/include/arrayUtils.h"
#include "../../../core/include/vec3Operation.h"


class SwitchFuncTest : public ::testing::Test {
protected:
    double rcut;
    double rcut_smooth;
    double rji;

    static void SetUpTestSuite() {
        std::cout << "SwitchFuncTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "SwitchFuncTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rcut = 3.5;
        rcut_smooth = 3.0;
        rji = 3.5;
    }

    void TearDown() override {

    }

};  // class : SwitchFuncTest



TEST_F(SwitchFuncTest, value_rcut) {
    rcut = 3.5;
    rcut_smooth = 3.0;
    rji = 3.5;
    matersdk::mtp::SwitchFunc<double> switch_func(rcut, rcut_smooth);

    double result = switch_func.get_result(rji);
    EXPECT_DOUBLE_EQ(result, 0);
}


TEST_F(SwitchFuncTest, value_random) {
    rcut = 3.5;
    rcut_smooth = 3.0;
    rji = 3.2;
    matersdk::mtp::SwitchFunc<double> switch_func(rcut, rcut_smooth);

    double result = switch_func.get_result(rji);
    EXPECT_FLOAT_EQ(result, 0.68256);
}


TEST_F(SwitchFuncTest, value_rcut_smooth) {
    rcut = 3.5;
    rcut_smooth = 3.0;
    rji = 3.0;
    matersdk::mtp::SwitchFunc<double> switch_func(rcut, rcut_smooth);

    double result = switch_func.get_result(rji);
    EXPECT_DOUBLE_EQ(result, 1);
}




class RadialBasisChebyshevTest : public ::testing::Test {
protected:
    double rcut;
    double rcut_smooth;
    int hmju;
    double rji;

    static void SetUpTestSuite() {
        std::cout << "RadialBasisChebyshevTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "RadialBasisChebyshevTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rcut = 6.0;
        rcut_smooth = 2.0;
        hmju = 0;
        rji = 3.0;
    }

    void TearDown() override {
    
    }
};  // class : RadialBasisChebyshevTest



TEST_F(RadialBasisChebyshevTest, constructor_default) {
    matersdk::mtp::RadialBasisChebyshev<double> rb;

    rb.show_in_value();
    rb.show_in_deriv();
}


TEST_F(RadialBasisChebyshevTest, constructor_1) {
    rcut = 6.0;
    rcut_smooth = 2.0;
    hmju = 3;
    rji = 3.0;
    matersdk::mtp::RadialBasisChebyshev<double> rb(rcut, rcut_smooth, hmju, rji);

    rb.show_in_value();
    rb.show_in_deriv();
}


TEST_F(RadialBasisChebyshevTest, get_info) {
    rcut = 6.0;
    rcut_smooth = 2.0;
    hmju = 3;
    rji = 3.0;
    matersdk::mtp::RadialBasisChebyshev<double> rb(rcut, rcut_smooth, hmju, rji);

    EXPECT_FLOAT_EQ(rcut, rb.get_rcut());
    EXPECT_FLOAT_EQ(rcut_smooth, rb.get_rcut_smooth());
    EXPECT_EQ(hmju, rb.get_hmju());
    EXPECT_FLOAT_EQ(rji, rb.get_rji());
}


TEST_F(RadialBasisChebyshevTest, get_chebyshev_vals) {
    rcut = 3.2;
    rcut_smooth = 3.0;
    hmju = 3;
    rji = 3.05;
    matersdk::mtp::RadialBasisChebyshev<double> rb(rcut, rcut_smooth, hmju, rji);

    const double* chebyshev_vals = rb.get_chebyshev_vals();
    printf("chebyshev_vals = [");
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        printf("%8f, ", chebyshev_vals[ii]);
    }
    printf("]\n");
}


TEST_F(RadialBasisChebyshevTest, get_chebyshev_ders) {
    rcut = 6.0;
    rcut_smooth = 2.0;
    hmju = 3;
    rji = 3.0;
    matersdk::mtp::RadialBasisChebyshev<double> rb(rcut, rcut_smooth, hmju, rji);

    const double* chebyshev_ders = rb.get_chebyshev_ders();
    printf("chebyshev_ders = [");
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        printf("%8f, ", chebyshev_ders[ii]);
    }
    printf("]\n");
}


TEST_F(RadialBasisChebyshevTest, get_rb_vals) {
    rcut = 6.0;
    rcut_smooth = 2.0;
    hmju = 3;
    rji = 3.0;
    matersdk::mtp::RadialBasisChebyshev<double> rb(rcut, rcut_smooth, hmju, rji);

    const double* rb_vals = rb.get_rb_vals();
    printf("rb_vals = [");
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        printf("%8f, ", rb_vals[ii]);
    }
    printf("]\n");
}


TEST_F(RadialBasisChebyshevTest, get_rb_ders) {
    rcut = 6.0;
    rcut_smooth = 2.0;
    hmju = 3;
    rji = 3.0;
    matersdk::mtp::RadialBasisChebyshev<double> rb(rcut, rcut_smooth, hmju, rji);

    const double* rb_ders = rb.get_rb_ders();
    printf("rb_ders = [");
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        printf("%8f, ", rb_ders[ii]);
    }
    printf("]\n");
}


TEST_F(RadialBasisChebyshevTest, check_deriv_from_definition_way1) {
    // Step 1. Declare parameters
    rcut = 3.2;
    rcut_smooth = 3.0;
    hmju = 8;
    rji = 3.05;
    double rji_end = 3.051;
    double delta = rji_end - rji;

    matersdk::mtp::RadialBasisChebyshev<double> rb(rcut, rcut_smooth, hmju, rji);
    matersdk::mtp::RadialBasisChebyshev<double> rb_end(rcut, rcut_smooth, hmju, rji_end);   
    
    double* chebyshev_ders_definition = (double*)malloc(sizeof(double) * (rb.get_hmju()+1));
    double* rb_ders_definition = (double*)malloc(sizeof(double) * (rb.get_hmju()+1));
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        chebyshev_ders_definition[ii] = 0;
        rb_ders_definition[ii] = 0;
    }

    // Step 2. Calculate ders according to definition
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        chebyshev_ders_definition[ii] = (rb_end.get_chebyshev_vals()[ii] - rb.get_chebyshev_vals()[ii]) / delta;
        rb_ders_definition[ii] = (rb_end.get_rb_vals()[ii] - rb.get_rb_vals()[ii]) / delta;
    }

    // Step 3. Compare derivs
    printf("1. Compare chebyshev_deriv:\n");
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        printf("%6f, ", rb.get_chebyshev_ders()[ii]);
    }
    printf("\n");
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        printf("%6f, ", chebyshev_ders_definition[ii]);
    }
    printf("\n");

    printf("2. Compare rb_deriv:\n");
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        printf("%6f, ", rb.get_rb_ders()[ii]);
    }
    printf("\n");
    for (int ii=0; ii<(rb.get_hmju()+1); ii++) {
        printf("%6f, ", rb_ders_definition[ii]);
    }
    printf("\n");
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
