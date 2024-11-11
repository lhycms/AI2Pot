#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>

#include "../include/mtp_utilities.cuh"


class MtpUtilitiesTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite() {
        std::cout << "MtpUtilitiesTest (TestSutie) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpUtilitiesTest (TestSuite) is tearing dow...\n";
    }

    void SetUp() override {
        
    }

    void TearDown() override {

    }
};  // class : MtpUtilitiesTest


TEST_F(MtpUtilitiesTest, mtp_hyperparameters) {
    printf("mtp_hyperparameters:\n");
    printf("\t1. MAX_CHEBYHSEV_SIZE = %d\n", MAX_CHEBYSHEV_SIZE);
    printf("\t2. MAX_NUM_MTP_BASIS = %d\n", MAX_NUM_MTP_BASIS);
    printf("\t3. ALPHA_MOMENTS_COUNT = %d\n", ALPHA_MOMENTS_COUNT);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
