#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>


class BlmTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        std::cout << "BlmTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "BlmTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {

    }

    void TearDown() override {
    }
};  // class : BlmTets


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv)
    return RUN_ALL_TESTS();
}