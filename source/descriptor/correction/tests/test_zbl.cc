#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>

#include "../include/zbl.h"


class ZBLTest : public ::testing::Test {
protected:
    int Zi;
    int Zj;
    double rmax;
    double rmin;
    double *ck;
    double *dk;

    static void SetUpTestSuite() {
        std::cout << "ZBLTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "ZBLTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        Zi = 16;
        Zj = 42;
        rmax = 5.0;
        rmin = 2.5;
        ck = (double*)malloc(sizeof(double) * 4);
        dk = (double*)malloc(sizeof(double) * 4);
        ck[0] = 1.1;
        ck[1] = 1.2;
        ck[2] = 1.3;
        ck[3] = 1.4;

        dk[0] = 2.1;
        dk[1] = 2.2;
        dk[2] = 2.3;
        dk[3] = 2.4;
    }

    void TearDown() override {
        free(ck);
        free(dk);
    }
};  // class : ZBLTest


TEST_F(ZBLTest, find_switch_func) {
    ai2pot::correction::ZBL<double> zbl(Zi, Zj, rmax, rmin, ck, dk);
    double result = zbl.find_switch_func(2.5);
    ASSERT_EQ(result, 1.0);
    result = zbl.find_switch_func(5.0);
    ASSERT_EQ(result, 0.0);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}