#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include "../include/descriptor.h"


class Descriptor2BTest : public ::testing::Test
{
protected:
    int size;
    double rmax;
    double rmin;
    double lambda_val;

    static void SetUpTestSuite() {
        std::cout << "Descriptor2BTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "Descriptor2BTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        size = 8;
        rmax = 5.0;
        rmin = 2.0;
        lambda_val = 5.0;
    }

    void TearDown() override {
    }
};  // class : Descriptor2BTest

TEST_F(Descriptor2BTest, default_constructor) 
{
    ai2pot::ace::Descriptor2B<double> descriptor_2b;
    ASSERT_EQ(descriptor_2b.chebyshev_size(), 0);
    ASSERT_EQ(descriptor_2b.ptr_rq(), nullptr);
}

TEST_F(Descriptor2BTest, constructor_1) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b(size, rmax, rmin, lambda_val);
    ASSERT_EQ(descriptor_2b.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(Descriptor2BTest, constructor_2) {
    ai2pot::ace::RQ_CChebyshev<double> rq_cchebyshev(size, rmax, rmin, lambda_val);
    ai2pot::ace::Descriptor2B<double> descriptor_2b(&rq_cchebyshev);
    ASSERT_EQ(descriptor_2b.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(Descriptor2BTest, copy_constructor) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Descriptor2B<double> descriptor_2b_2(descriptor_2b_1);
    ASSERT_EQ(descriptor_2b_2.chebyshev_size(), size);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->size(), size);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->rmax(), rmax);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->rmin(), rmin);
    ASSERT_EQ(descriptor_2b_2.ptr_rq()->lambda_val(), lambda_val);
}

TEST_F(Descriptor2BTest, copy_constructor_move) {
    ai2pot::ace::Descriptor2B<double> descriptor_2b_1(size, rmax, rmin, lambda_val);
    ai2pot::ace::Descriptor2B<double> descriptor_2b_2(std::move(descriptor_2b_1));
    ASSERT_EQ(descriptor_2b_1.chebyshev_size(), 0);
    ASSERT_EQ(descriptor_2b_1.ptr_rq(), nullptr);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}