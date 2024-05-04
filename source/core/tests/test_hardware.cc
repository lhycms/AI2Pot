#include <gtest/gtest.h>
#include <iostream>

#include "../include/hardware.h"


class HardwareTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        std::cout << "Set up HardwareTest (TestSuite)...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "Tear down HardwareTest (TestSuite)...\n";
    }

    void SetUp() override {

    }

    void TearDown() override {

    }
};



TEST_F(HardwareTest, getNumProcessorsOnln) {
    EXPECT_EQ(getNumProcessorsOnln(), 28);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}