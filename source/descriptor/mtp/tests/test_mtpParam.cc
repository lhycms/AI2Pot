#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include "../include/mtpParam.h"

class MtpParamTest : public ::testing::Test
{
protected:
    int mtp_level;
    int alpha_moments_count;
    int alpha_index_basic_count;
    std::vector<std::vector<int>> alpha_index_basic;
    int alpha_index_times_count;
    std::vector<std::vector<int>> alpha_index_times;
    int alpha_scalar_moments;
    std::vector<int> alpha_moment_mapping;

    static void SetUpTestSuite() {
        std::cout << "MtpParamTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpParamTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        mtp_level = 6;
    }

    void TearDown() override {}
};  // class : MtpParamTest


TEST_F(MtpParamTest, test_all)
{   
    matersdk::mtp::MtpParam::find_param(
        mtp_level,
        alpha_moments_count,
        alpha_index_basic_count,
        alpha_index_basic,
        alpha_index_times_count,
        alpha_index_times,
        alpha_scalar_moments,
        alpha_moment_mapping);
    
    std::cout << alpha_moment_mapping.size() << std::endl;
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}