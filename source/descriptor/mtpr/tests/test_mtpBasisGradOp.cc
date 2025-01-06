#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>

#include "../include/mtpParamOp.h"
#include "../include/mtpBasisGradOp.h"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"


class MtpBasisGradOpTest : public ::testing::Test
{
protected:
    c10::TensorOptions int_options;
    c10::TensorOptions float_options;
    at::Tensor binum_tensor;
    at::Tensor bnumneigh_tensor;
    at::Tensor bfirstneigh_tensor;
    at::Tensor brcs_tensor;
    at::Tensor btypes_tensor;
    int nghost;
    int umax_num_neighs;

    

    static void SetUpTestSuite() {
        std::cout << "MtpBasisGradOpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpBasisGradOpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {

    }

    void TearDown() override {

    }
};  // class : MtpBasisGradOpTest


TEST_F(MtpBasisGradOpTest, der2coeff_accuracy) {

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
