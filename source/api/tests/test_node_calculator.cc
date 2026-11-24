#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include <stdio.h>
#include <iostream>
#include <string>

#include "../include/node_calculator.h"


class AI2PotNodeCalculatorTest : public ::testing::Test {
protected:
    std::string pt_path;
    torch::Device device = torch::kCPU;

    static void SetUpTestSuite() {
        std::cout << "AI2PotNodeCalculatorTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "AI2PotNodeCalculatorTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        pt_path = "/data/home/liuhanyu/ai2pot_paper_code/7.JCTC_1.2.2b/21_MTP_Training/ai2pot_libtorch.pt";
    }

    void TearDown() override {

    }
};  // class : AI2PotNodeCalculatortest


TEST_F(AI2PotNodeCalculatorTest, constructor)
{
    AI2PotNodeCalculator node_calculator = AI2PotNodeCalculator(pt_path, device);
    std::cout << node_calculator.potential_type << std::endl;
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
