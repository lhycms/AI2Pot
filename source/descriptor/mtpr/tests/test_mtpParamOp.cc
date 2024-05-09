#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>

#include "../include/mtpParamOp.h"


class MtpParamOpTest : public ::testing::Test {
protected:
    int mtp_level;

    static void SetUpTestSuite() {
        std::cout << "MtpParamOpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpParamOpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        mtp_level = 10;
    }

    void TearDown() override {

    }
};  // class : MtpParamOpTest


TEST_F(MtpParamOpTest, forward)
{
    torch::autograd::variable_list results = ai2pot::mtpr::MtpParamOp(mtp_level);
//std::cout << results[0] << std::endl;
//std::cout << results[1] << std::endl;
//std::cout << results[2] << std::endl;
//std::cout << results[3] << std::endl;
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}