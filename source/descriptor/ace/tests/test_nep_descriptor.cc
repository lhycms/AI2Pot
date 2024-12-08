#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>

#include "../include/nep_descriptor.h"


class PseudoWignerTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite() {
        std::cout << "PseudoWignerTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "PseudoWignerTest (TestSuite) is tearing down...\n";;
    }

    void SetUp() override {
    }

    void TearDown() override {
    }
};  // class : PseudoWignerTest


class NepDescriptorTest : public ::testing::Test
{
protected:
    int num_descriptors_r;
    int num_descriptors_a;
    int num_l;
    int max_l;

    int max_body;
    int n_r_max;
    int n_a_max;
    int l_3b_max;
    int l_4b_max;
    int l_5b_max;

    static void SetUpTestSuite() {
        std::cout << "NepDescriptorTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NepDescriptorTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        max_body = 5;
        n_r_max = 10;
        n_a_max = 8;
        l_3b_max = 4;
        l_4b_max = 2;
        l_5b_max = 1;
    }

    void TearDown() override {

    }
};  // class : NepDescriptorTest





TEST_F(PseudoWignerTest, init) {
    ai2pot::ace::PseudoWigner<double> pwigner;
    printf("1. Value of C3B[%d] = %g\n", 0, pwigner.C3B()[0]);
}


TEST_F(NepDescriptorTest, find_hyperparams)
{
    ai2pot::ace::NepDescriptor<double>::find_hyperparams(
        num_descriptors_r,
        num_descriptors_a,
        num_l,
        max_l,
        max_body,
        n_r_max,
        n_a_max,
        l_3b_max,
        l_4b_max,
        l_5b_max);

printf("1. num_r_descriptors = %d\n", num_descriptors_r);
printf("2. num_a_descriptors = %d\n", num_descriptors_a);
printf("3. num_l = %d\n", num_l);
printf("4. max_l = %d\n", max_l);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}