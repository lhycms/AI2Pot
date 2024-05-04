#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include "../include/mtpMRadial.h"


class MtpSwitchFunc1Test : public ::testing::Test
{
protected:
    double rcut;
    double rcut_smooth;
    double distance_ij;    // absolute value
    static void SetUpTestSuite() {
        std::cout << "MtpSwitchFunc1Test (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpSwitchFunc1Test (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rcut = 3.2;
        rcut_smooth = 3.0;
    }

    void TearDown() override {
    }
};  // class : MtpSwitchFunc1Test


class MtpSwitchFunc2Test : public ::testing::Test
{
protected:
    double rcut;
    double rcut_smooth;
    double distance_ij;     // absolute value
    
    static void SetUpTestSuite() {
        std::cout << "MtpSwitchFunc2 (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpSwitchFunc2 (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rcut = 3.2;
        rcut_smooth = 3.0;
    }

    void TearDown() override {
    }
};  // class : SwitchFunc2Test


class ChebyshevPolyTest : public ::testing::Test {
protected:
    int size; 
    double rcut;
    double rcut_smooth;
    double distance_ij;
    static void SetUpTestSuite() {
        std::cout << "ChebyshevPolyTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "ChebyshevPolyTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rcut = 3.2;
        rcut_smooth = 3.0;
    }

    void TearDown() override {
    }
};  // class : ChebyshevPolyTest


class MtpQTest : public ::testing::Test {
protected:
    int size;
    double rcut;
    double rcut_smooth;
    double distance_ij;
    static void SetUpTestSuite() {
        std::cout << "MtpQTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSutie() {
        std::cout << "MtpQTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rcut = 3.2;
        rcut_smooth = 3.0;
    }

    void TearDown() override {
    }
};  // class : MtpQ


TEST_F(MtpSwitchFunc1Test, copy_constructor) {
    distance_ij = 3.1;
    matersdk::mtp::MtpSwitchFunc1<double> msf1(rcut, rcut_smooth);
    matersdk::mtp::MtpSwitchFunc1<double> msf2(msf1);
}

TEST_F(MtpSwitchFunc1Test, assignment_operator) {
    distance_ij = 3.1;
    matersdk::mtp::MtpSwitchFunc1<double> msf1(rcut, rcut_smooth);
    matersdk::mtp::MtpSwitchFunc1<double> msf2(rcut-0.1, rcut_smooth);
    msf1 = msf2;
}

TEST_F(MtpSwitchFunc1Test, get_result_and_deriv2r) {
    distance_ij = 0.0;
    matersdk::mtp::MtpSwitchFunc1<double> msf0(rcut, rcut_smooth);
    printf("distance_ij = %5f, switch_func1_result = %6f, switch_func1_deriv2r = %6f\n",
        distance_ij,
        msf0.get_result(distance_ij),
        msf0.get_deriv2r());

    distance_ij = 3.0;
    matersdk::mtp::MtpSwitchFunc1<double> msf1(rcut, rcut_smooth);
    ASSERT_DOUBLE_EQ(msf1.get_result(distance_ij), -1);
    printf("distance_ij = %5f, switch_func1_result = %6f, switch_func1_deriv2r = %6f\n",
        distance_ij,
        msf1.get_result(distance_ij),
        msf1.get_deriv2r());

    distance_ij = 3.1;
    matersdk::mtp::MtpSwitchFunc1<double> msf2(rcut, rcut_smooth);
    ASSERT_GE(msf2.get_result(distance_ij), -1);
    ASSERT_LE(msf2.get_result(distance_ij), 1);
    printf("distance_ij = %5f, switch_func1_result = %6f, switch_func1_deriv2r = %6f\n",
        distance_ij,
        msf2.get_result(distance_ij),
        msf2.get_deriv2r());

    distance_ij = 3.2;
    matersdk::mtp::MtpSwitchFunc1<double> msf3(rcut, rcut_smooth);
    ASSERT_DOUBLE_EQ(msf3.get_result(distance_ij), 1);
    printf("distance_ij = %5f, switch_func1_result = %6f, switch_func1_deriv2r = %6f\n",
        distance_ij,
        msf3.get_result(distance_ij),
        msf3.get_deriv2r());
}



TEST_F(MtpSwitchFunc2Test, copy_constructor) 
{
    distance_ij = 3.1;
    matersdk::mtp::MtpSwitchFunc2<double> msf1(rcut, rcut_smooth);
    matersdk::mtp::MtpSwitchFunc2<double> msf2(msf1);
}

TEST_F(MtpSwitchFunc2Test, assignment_operator)
{
    distance_ij = 3.1;
    matersdk::mtp::MtpSwitchFunc2<double> msf1(rcut, rcut_smooth);
    matersdk::mtp::MtpSwitchFunc2<double> msf2(rcut-0.1, rcut_smooth);
    msf1 = msf2;
}

TEST_F(MtpSwitchFunc2Test, get_result_and_deriv2r) {
    distance_ij = 0.0;
    matersdk::mtp::MtpSwitchFunc2<double> msf0(rcut, rcut_smooth);
    printf("distance_ij = %5f, switch_func2_result = %6f, switch_func1_deriv2r = %6f\n", 
        distance_ij, 
        msf0.get_result(distance_ij),
        msf0.get_deriv2r(distance_ij));
    ASSERT_DOUBLE_EQ(msf0.get_result(distance_ij), 0);

    distance_ij = 3.0;
    matersdk::mtp::MtpSwitchFunc2<double> msf1(rcut, rcut_smooth);
    printf("distance_ij = %5f, switch_func2_result = %6f, switch_func1_deriv2r = %6f\n", 
        distance_ij, 
        msf1.get_result(distance_ij),
        msf1.get_deriv2r(distance_ij));
    ASSERT_DOUBLE_EQ(msf1.get_result(distance_ij), 1.0);

    distance_ij = 3.05;
    matersdk::mtp::MtpSwitchFunc2<double> msf2(rcut, rcut_smooth);
    printf("distance_ij = %5f, switch_func2_result = %6f, switch_func1_deriv2r = %6f\n", 
        distance_ij, 
        msf2.get_result(distance_ij),
        msf2.get_deriv2r(distance_ij));
    ASSERT_GE(msf2.get_result(distance_ij), 0.0);
    ASSERT_LE(msf2.get_result(distance_ij), 1.0);

    distance_ij = 3.2;
    matersdk::mtp::MtpSwitchFunc2<double> msf3(rcut, rcut_smooth);
    printf("distance_ij = %5f, switch_func2_result = %6f, switch_func1_deriv2r = %6f\n", 
        distance_ij, 
        msf3.get_result(distance_ij),
        msf3.get_deriv2r(distance_ij));
    ASSERT_DOUBLE_EQ(msf3.get_result(distance_ij), 0.0);
}


TEST_F(ChebyshevPolyTest, build) 
{
    rcut = 5.0;
    rcut_smooth = 2.0;
    distance_ij = 3.14;  // 3.05   // When r<r_{rcut_smooth} or r >= rcut, Chebyshev will be too much larger
    size = 8;
    matersdk::mtp::ChebyshevPoly<double> chebyshev(size, rcut, rcut_smooth);
    chebyshev.build(distance_ij);
    chebyshev.show();
}

TEST_F(ChebyshevPolyTest, copy_constructor) 
{
    distance_ij = 3.05;
    size = 8;
    matersdk::mtp::ChebyshevPoly<double> chebyshev1(size, rcut, rcut_smooth);
    chebyshev1.build(distance_ij);
    matersdk::mtp::ChebyshevPoly<double> chebyshev2(chebyshev1);
    //chebyshev2.show();
    //chebyshev1.show();

    for (int ii=0; ii<chebyshev2.size(); ii++) {
        ASSERT_DOUBLE_EQ(
            chebyshev1.get_result()[ii],
            chebyshev2.get_result()[ii]);
        ASSERT_DOUBLE_EQ(
            chebyshev1.get_deriv2xi()[ii],
            chebyshev2.get_deriv2xi()[ii]);
        ASSERT_DOUBLE_EQ(
            chebyshev1.get_deriv2r()[ii],
            chebyshev2.get_deriv2r()[ii]);
    }
}

TEST_F(ChebyshevPolyTest, copy_constructor_move)
{
    distance_ij = 3.05;
    size = 8;
    matersdk::mtp::ChebyshevPoly<double> chebyshev1(size, rcut, rcut_smooth);
    chebyshev1.build(distance_ij);
    matersdk::mtp::ChebyshevPoly<double> chebyshev2(std::move(chebyshev1));
    //chebyshev1.show();
    //chebyshev2.show();
    ASSERT_EQ(chebyshev1.get_result(), nullptr);
    ASSERT_EQ(chebyshev1.get_deriv2xi(), nullptr);
    ASSERT_EQ(chebyshev1.get_deriv2r(), nullptr);
}

TEST_F(ChebyshevPolyTest, assignment_operator)
{
    distance_ij = 3.05;
    size = 8;
    matersdk::mtp::ChebyshevPoly<double> chebyshev1(size, rcut, rcut_smooth);
    chebyshev1.build(distance_ij);
    matersdk::mtp::ChebyshevPoly<double> chebyshev2(size, rcut, rcut_smooth);
    chebyshev2 = chebyshev1;

    //chebyshev1.show();
    //chebyshev2.show();
    for (int ii=0; ii<chebyshev2.size(); ii++) {
        ASSERT_DOUBLE_EQ(
            chebyshev1.get_result()[ii],
            chebyshev2.get_result()[ii]);
        ASSERT_DOUBLE_EQ(
            chebyshev1.get_deriv2xi()[ii],
            chebyshev2.get_deriv2xi()[ii]);
        ASSERT_DOUBLE_EQ(
            chebyshev1.get_deriv2r()[ii],
            chebyshev2.get_deriv2r()[ii]);
    }
}

TEST_F(ChebyshevPolyTest, assignment_operator_move)
{
    distance_ij = 3.05;
    size = 8;
    matersdk::mtp::ChebyshevPoly<double> chebyshev1(size, rcut, rcut_smooth);
    chebyshev1.build(distance_ij);
    matersdk::mtp::ChebyshevPoly<double> chebyshev2(size, rcut, rcut_smooth);
    chebyshev2 = std::move(chebyshev1);
    //chebyshev1.show();
    //chebyshev2.show();
    ASSERT_EQ(chebyshev1.get_result(), nullptr);
    ASSERT_EQ(chebyshev1.get_deriv2xi(), nullptr);
    ASSERT_EQ(chebyshev1.get_deriv2r(), nullptr);
}


TEST_F(MtpQTest, build)
{
    size = 8;
    distance_ij = 3.05;
    matersdk::mtp::MtpQ<double> mtp_q(size, rcut, rcut_smooth);
    mtp_q.build(distance_ij);
    mtp_q.show();
}

TEST_F(MtpQTest, copy_constructor)
{
    size = 8;
    distance_ij = 3.05;
    matersdk::mtp::MtpQ<double> mtp_q_1(size, rcut, rcut_smooth);
    mtp_q_1.build(distance_ij);
    matersdk::mtp::MtpQ<double> mtp_q_2(mtp_q_1);
    //mtp_q_1.show();
    //mtp_q_2.show();
    for (int ii=0; ii<mtp_q_1.size(); ii++) {
        ASSERT_EQ(mtp_q_1.get_result()[ii], mtp_q_2.get_result()[ii]);
        ASSERT_EQ(mtp_q_1.get_deriv2r()[ii], mtp_q_2.get_deriv2r()[ii]);
    }
}

TEST_F(MtpQTest, copy_constructor_move)
{
    size = 8;
    distance_ij = 3.05;
    matersdk::mtp::MtpQ<double> mtp_q_1(size, rcut, rcut_smooth);
    mtp_q_1.build(distance_ij);
    matersdk::mtp::MtpQ<double> mtp_q_2(std::move(mtp_q_1));
    //mtp_q_1.show();
    //mtp_q_2.show();

    ASSERT_EQ(mtp_q_1.get_result(), nullptr);
    ASSERT_EQ(mtp_q_1.get_deriv2r(), nullptr);
}

TEST_F(MtpQTest, assignment_operator)
{
    size = 8;
    distance_ij = 3.05;
    matersdk::mtp::MtpQ<double> mtp_q_1(size, rcut, rcut_smooth);
    mtp_q_1.build(distance_ij);
    matersdk::mtp::MtpQ<double> mtp_q_2(size, rcut-0.1, rcut_smooth);
    mtp_q_2 = mtp_q_1;
    //mtp_q_1.show();
    //mtp_q_2.show();
    for (int ii=0; ii<mtp_q_1.size(); ii++) {
        ASSERT_EQ(mtp_q_1.get_result()[ii], mtp_q_2.get_result()[ii]);
        ASSERT_EQ(mtp_q_1.get_deriv2r()[ii], mtp_q_2.get_deriv2r()[ii]);
    }
}

TEST_F(MtpQTest, assignment_operator_move)
{
    size = 8;
    distance_ij = 3.05;
    matersdk::mtp::MtpQ<double> mtp_q_1(size, rcut, rcut_smooth);
    mtp_q_1.build(distance_ij);
    matersdk::mtp::MtpQ<double> mtp_q_2(size, rcut-0.1, rcut_smooth);
    mtp_q_2 = std::move(mtp_q_1);
    //mtp_q_1.show();
    //mtp_q_2.show();
    ASSERT_EQ(mtp_q_1.get_result(), nullptr);
    ASSERT_EQ(mtp_q_1.get_deriv2r(), nullptr);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


