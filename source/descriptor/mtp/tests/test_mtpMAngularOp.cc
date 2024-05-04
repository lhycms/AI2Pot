#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include "../include/mtpMAngularOp.h"


class OuterNu0OpTest : public ::testing::Test
{
protected:
    at::Tensor ircs_tensor;
    c10::TensorOptions options;
    int64_t nneighs;

    static void SetUpTestSuite() {
        std::cout << "OuterNu0OpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "OuterNu0OpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(c10::kCPU);
        ircs_tensor = at::ones({19, 3}, options)
            .requires_grad_(true);
        nneighs = ircs_tensor.sizes()[0];
    }

    void TearDown() override {
    }
};  // class : OuterNu0OpTest


class MtpMAngularOpTest: public ::testing::Test {
protected:
    at::Tensor ircs_tensor;
    int64_t nu_0;
    int64_t nu_1;
    int64_t nu_2;
    int64_t nneighs;

    static void SetUpTestSuite() {
        std::cout << "MtpMAngularOpTest (TestSuite) is setting up...\n";        
    }

    static void TearDownTestSuite() {
        std::cout << "MtpMAngularOpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        c10::TensorOptions options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(c10::kCPU);
        ircs_tensor = at::ones({19, 3}, options);
        ircs_tensor.requires_grad_(true);
        nu_0 = 0;
        nu_1 = 1;
        nu_2 = 2;
        nneighs = ircs_tensor.sizes()[0];
    }

    void TearDown() override {
    }
};  // class : MtpMAngularOpTest




TEST_F(OuterNu0OpTest, forward_and_backward) {
    at::Tensor result = matersdk::mtp::OuterNu0Op(ircs_tensor)[0];
    result.sum().backward();
    //std::cout << result << std::endl;
    //std::cout << ircs_tensor.grad() << std::endl;
    ASSERT_EQ( result.dim(), 1 );
    ASSERT_EQ( result.sizes()[0], nneighs );
    ASSERT_EQ( ircs_tensor.grad().dim(), 2 );
    ASSERT_EQ( ircs_tensor.grad().sizes()[0], nneighs );
    ASSERT_EQ( ircs_tensor.grad().sizes()[1], 3 );
}


TEST_F(MtpMAngularOpTest, forward_and_backward_nu0) {
    at::Tensor mtp_angular_0 = matersdk::mtp::MtpMAngularOp(
        ircs_tensor,
        nu_0);
    mtp_angular_0.sum().backward();
//std::cout << mtp_angular_0 << std::endl;
//std::cout << ircs_tensor.grad() << std::endl;
    ASSERT_EQ( mtp_angular_0.dim(), 1 );
    ASSERT_EQ( mtp_angular_0.sizes()[0], nneighs );
    ASSERT_EQ( ircs_tensor.grad().dim(), 2 );
    ASSERT_EQ( ircs_tensor.grad().sizes()[0], nneighs );
    ASSERT_EQ( ircs_tensor.grad().sizes()[1], 3 );
}


TEST_F(MtpMAngularOpTest, forward_and_backward_nu1) {
    at::Tensor mtp_angular_1 = matersdk::mtp::MtpMAngularOp(
        ircs_tensor,
        nu_1);
    mtp_angular_1.sum().backward();
//std::cout << mtp_angular_1 << std::endl;
//std::cout << ircs_tensor.grad() << std::endl;
    ASSERT_EQ( mtp_angular_1.dim(), 2 );
    ASSERT_EQ( mtp_angular_1.sizes()[0], nneighs );
    ASSERT_EQ( mtp_angular_1.sizes()[1], 3 );
    ASSERT_EQ( ircs_tensor.grad().dim(), 2 );
    ASSERT_EQ( ircs_tensor.grad().sizes()[0], nneighs );
    ASSERT_EQ( ircs_tensor.grad().sizes()[1], 3 );
}

TEST_F(MtpMAngularOpTest, forward_and_backward_nu2) {
    at::Tensor mtp_angular_2 = matersdk::mtp::MtpMAngularOp(
        ircs_tensor,
        nu_2);
    mtp_angular_2.sum().backward();
//std::cout << mtp_angular_2 << std::endl;
//std::cout << ircs_tensor.grad() << std::endl;
    ASSERT_EQ( mtp_angular_2.dim(), 3 );
    ASSERT_EQ( mtp_angular_2.sizes()[0], nneighs );
    ASSERT_EQ( mtp_angular_2.sizes()[1], 3 );
    ASSERT_EQ( mtp_angular_2.sizes()[2], 3 );
    ASSERT_EQ( ircs_tensor.grad().dim(), 2 );
    ASSERT_EQ( ircs_tensor.grad().sizes()[0], nneighs );
    ASSERT_EQ( ircs_tensor.grad().sizes()[1], 3);
}


TEST_F(MtpMAngularOpTest, speed) {
    int times = 5 * 1E3;
    auto time1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int ii=0; ii<times; ii++) {
        at::Tensor mtp_angular = matersdk::mtp::MtpMAngularOp(
            ircs_tensor,
            3);
    }
    auto time2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
    std::cout << "1. Time costing in (ms) = " << duration.count() << std::endl;
}


TEST_F(MtpMAngularOpTest, deriv_accuracy) {
    at::Tensor mtp_angular_tensor = matersdk::mtp::MtpMAngularOp(
        ircs_tensor,
        2);
    at::Tensor result = mtp_angular_tensor.sum();
    result.backward();
    std::cout << "1. Partial derivative wrt. x calculated by Autograd = " << ircs_tensor.grad()[0][0].item<double>() << std::endl;

    
    double* ircs = ircs_tensor.data_ptr<double>();
    ircs[0*3 + 0] += 0.001;
    at::Tensor mtp_angular_tensor1 = matersdk::mtp::MtpMAngularOp(
        ircs_tensor,
        2);
    at::Tensor result1 = mtp_angular_tensor1.sum();
    std::cout << "2. Partial derivative wrt. x calculated by finite difference = " << (result1 - result).item<double>() / 0.001 << std::endl;
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
