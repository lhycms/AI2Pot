#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/nnMtpUtils.h"



typedef double real;

class NNMtpUtilsTest : public ::testing::Test
{
protected:
    real hidden_val;
    real activated_hidden_val;
    real activated_hidden_val_;
    real activated_hidden_der;
    
    static void SetUpTestSuite() {
        printf("NNMtpUtilsTest (TestSuite) is setting up...\n");
    }

    static void TearDownTestSuite() {
        printf("NNMtpUtilsTest (TestSuite) is tearing down...\n");
    }

    void SetUp() override {
        activated_hidden_val = 0;
        activated_hidden_val_ = 0;
        activated_hidden_der = 0;
        hidden_val = 0.1;
    }

    void TearDown() override {

    }
};  // class : NNMtpUtilsTest


TEST_F(NNMtpUtilsTest, der_accuracy) {
    real delta = 1e-5;
    ai2pot::nnmtp::TanhActivationFunc<real>::find_val(activated_hidden_val, hidden_val);
    ai2pot::nnmtp::TanhActivationFunc<real>::find_der(activated_hidden_der, hidden_val);
    ai2pot::nnmtp::TanhActivationFunc<real>::find_val(activated_hidden_val_, hidden_val+delta);

    printf("Tanh der accuracy:\n");
    printf("\t1. Derivative calculated by custom code = %.10lf\n", (activated_hidden_val_ - activated_hidden_val)/delta);
    printf("\t2. Derivative calculated by fdm = %.10lf\n", activated_hidden_der);

}   



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}