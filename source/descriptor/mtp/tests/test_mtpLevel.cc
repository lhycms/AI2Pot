#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <utility>
#include <vector>
#include "../include/mtpLevel.h"


class MtpMCoeffPairTest : public ::testing::Test 
{
protected:
    int mu_0, nu_0;
    std::pair<int, int> coeff_pair_0;

    static void SetUpTestSuite() {
        std::cout << "MtpMCoeffPairTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpMCoeffPairTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        mu_0 = 1;
        nu_0 = 2;
        coeff_pair_0 = std::pair<int, int>(mu_0, nu_0);
    }

    void TearDown() override {
    }
};  // class : MtpMCoeffPairTest


class MtpMCoeffPairCombsTest : public ::testing::Test
{
protected:
    int max_level_6;
    int max_level_8;
    int max_level_14;
    int max_level_28;

    int aim_level_0;
    int aim_level_1;
    int aim_level_6;
    int aim_level_8;
    int aim_level_14;

    static void SetUpTestSuite() {
        std::cout << "MtpMCoeffPairCombs (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpMCoeffPairCombs (TestSuite) is tearding down...\n";
    }

    void SetUp() override {
        max_level_6 = 6;
        max_level_8 = 8;
        max_level_14 = 14;
        max_level_28 = 28;

        aim_level_0 = 0;
        aim_level_1 = 1;
        aim_level_6 = 6;
        aim_level_8 = 8;
        aim_level_14 = 14;
    }

    void TearDown() override {
    }
};  // class : MtpMCoeffPairCombsTest


std::ostream& operator<<(
    std::ostream& COUT, 
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> coeff_pair_combs)
{
    int count = 0;
    for (auto& coeff_pair_comb : coeff_pair_combs) {
        printf("Comb#%5d:\n\t", count);
        for (auto& coeff_pair : coeff_pair_comb) {
            printf("[%3d, %3d], ", coeff_pair.coeff_pair().first, coeff_pair.coeff_pair().second);
        }
        printf("\n");
        count++;
    }
    printf("MtpM Level = %3d, %5d combinations in total.", coeff_pair_combs.size(), coeff_pair_combs.size());
    return COUT;
}



TEST_F(MtpMCoeffPairTest, constructor_default)
{
    matersdk::mtp::MtpMCoeffPair mcp;
    ASSERT_EQ(mcp.level(), 0);
}

TEST_F(MtpMCoeffPairTest, constructor_1)
{
    matersdk::mtp::MtpMCoeffPair mcp(mu_0, nu_0);
    //std::cout << mcp.level() << std::endl;
    ASSERT_EQ(mcp.level(), 8);
}

TEST_F(MtpMCoeffPairTest, constructor_2)
{
    matersdk::mtp::MtpMCoeffPair mcp(coeff_pair_0);
    //std::cout << mcp.level() << std::endl;
    ASSERT_EQ(mcp.level(), 8);
}

TEST_F(MtpMCoeffPairTest, copy_constructor)
{
    matersdk::mtp::MtpMCoeffPair mcp_1(coeff_pair_0);
    matersdk::mtp::MtpMCoeffPair mcp_2(mcp_1);
    ASSERT_EQ(mcp_1.level(), mcp_2.level());
}

TEST_F(MtpMCoeffPairTest, assignment_operator)
{
    matersdk::mtp::MtpMCoeffPair mcp_1(coeff_pair_0);
    matersdk::mtp::MtpMCoeffPair mcp_2;
    ASSERT_EQ(mcp_1.level(), 8);
    ASSERT_EQ(mcp_2.level(), 0);
    mcp_2 = mcp_1;
    ASSERT_EQ(mcp_1.level(), 8);
    ASSERT_EQ(mcp_2.level(), 8);
}

TEST_F(MtpMCoeffPairCombsTest, constructor_default)
{
    matersdk::mtp::MtpMCoeffPairCombs mcpcs;
    //std::cout << mcpcs.coeff_pair_combs().size() << std::endl;
    ASSERT_EQ(mcpcs.coeff_pair_combs().size(), 0);
}


TEST_F(MtpMCoeffPairCombsTest, constructor_1)
{
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_1(aim_level_6);
    //mcpcs_1.show();
    ASSERT_EQ(mcpcs_1.coeff_pair_combs().size(), 5);

    matersdk::mtp::MtpMCoeffPairCombs mcpcs_2(aim_level_8);
    //mcpcs_2.show();
    ASSERT_EQ(mcpcs_2.coeff_pair_combs().size(), 9);

    // Compare it with MLIP
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_6(6);
ASSERT_GE(mcpcs_6.coeff_pair_combs().size(), 5);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_8(8);
ASSERT_GE(mcpcs_8.coeff_pair_combs().size(), 9);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_10(10);
ASSERT_GE(mcpcs_10.coeff_pair_combs().size(), 16);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_12(12);
ASSERT_GE(mcpcs_12.coeff_pair_combs().size(), 29);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_14(14);
//mcpcs_14.show();
ASSERT_GE(mcpcs_14.coeff_pair_combs().size(), 52);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_16(16);
ASSERT_GE(mcpcs_16.coeff_pair_combs().size(), 92);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_18(18);
ASSERT_GE(mcpcs_18.coeff_pair_combs().size(), 163);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_20(20);
ASSERT_GE(mcpcs_20.coeff_pair_combs().size(), 288);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_22(22);
ASSERT_GE(mcpcs_22.coeff_pair_combs().size(), 500);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_24(24);
ASSERT_GE(mcpcs_24.coeff_pair_combs().size(), 864);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_26(26);
ASSERT_GE(mcpcs_26.coeff_pair_combs().size(), 1464);
    matersdk::mtp::MtpMCoeffPairCombs mcpcs_28(28);
ASSERT_GE(mcpcs_28.coeff_pair_combs().size(), 2445);
}

TEST_F(MtpMCoeffPairCombsTest, copy_constructor)
{
    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_6(max_level_6);
    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_6_(coeff_pair_combs_6);

    for (int ii=0; ii<coeff_pair_combs_6.coeff_pair_combs().size(); ii++) {
        std::vector<matersdk::mtp::MtpMCoeffPair> tmp_coeff_pair_comb_6 = 
            coeff_pair_combs_6.coeff_pair_combs()[ii];
        std::vector<matersdk::mtp::MtpMCoeffPair> tmp_coeff_pair_comb_6_ = 
            coeff_pair_combs_6_.coeff_pair_combs()[ii];
        for (int jj=0; jj<tmp_coeff_pair_comb_6.size(); jj++) {
            ASSERT_EQ(
                tmp_coeff_pair_comb_6[jj].coeff_pair().first,
                tmp_coeff_pair_comb_6_[jj].coeff_pair().first);
            ASSERT_EQ(
                tmp_coeff_pair_comb_6[jj].coeff_pair().second,
                tmp_coeff_pair_comb_6_[jj].coeff_pair().second);
        }
    }
}

TEST_F(MtpMCoeffPairCombsTest, assignment_operator)
{
    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_6(max_level_6);
    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_8(max_level_8);
    coeff_pair_combs_8 = coeff_pair_combs_6;

    for (int ii=0; ii<coeff_pair_combs_6.coeff_pair_combs().size(); ii++)
    {
        std::vector<matersdk::mtp::MtpMCoeffPair> tmp_coeff_pair_comb_6 = 
            coeff_pair_combs_6.coeff_pair_combs()[ii];
        std::vector<matersdk::mtp::MtpMCoeffPair> tmp_coeff_pair_comb_8 = 
            coeff_pair_combs_8.coeff_pair_combs()[ii];
        for (int jj=0; jj<tmp_coeff_pair_comb_6.size(); jj++) {
            ASSERT_EQ(
                tmp_coeff_pair_comb_6[jj].coeff_pair().first,
                tmp_coeff_pair_comb_8[jj].coeff_pair().first);
            ASSERT_EQ(
                tmp_coeff_pair_comb_6[jj].coeff_pair().second,
                tmp_coeff_pair_comb_8[jj].coeff_pair().second);
        }
    }
}

TEST_F(MtpMCoeffPairCombsTest, get_all_schemes4lev)
{
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> combs_0 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_0, 0, 0);
    ASSERT_EQ(combs_0.size(), 1);   // Contains empty scheme.

    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> combs_1 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_1, 0, 0);
    ASSERT_EQ(combs_1.size(), 0);

    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> combs_6 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_6, 0, 0);
//std::cout << combs_6 << std::endl;
    ASSERT_EQ(combs_6.size(), 5);   // Contains empty scheme.

    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> combs_8 =
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_8, 0, 0);
//std::cout << combs_8 << std::endl;
    ASSERT_EQ(combs_8.size(), 9);

    // Note 检查是否包含 [0, 2], [1, 1], [0, 1]
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> combs_14 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_14, 0, 0);
//std::cout << combs_14 << std::endl;
    ASSERT_EQ(combs_14.size(), 67);
}


TEST_F(MtpMCoeffPairCombsTest, get_contracted_combs)
{
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> coeff_pair_combs_0 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_0, 0, 0);
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> contracted_coeff_pair_combs_0 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_contracted_combs(coeff_pair_combs_0);
//std::cout << contracted_coeff_pair_combs_0 << std::endl;
    ASSERT_EQ(contracted_coeff_pair_combs_0.size(), 0);

    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> coeff_pair_combs_1 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_1, 0, 0);
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> contracted_coeff_pair_combs_1 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_contracted_combs(coeff_pair_combs_1);
//std::cout << contracted_coeff_pair_combs_1 << std::endl;
    ASSERT_EQ(contracted_coeff_pair_combs_1.size(), 0);;

    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> coeff_pair_combs_6 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_6, 0, 0);
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> contracted_coeff_pair_combs_6 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_contracted_combs(coeff_pair_combs_6);
//std::cout << contracted_coeff_pair_combs_6 << std::endl;
    ASSERT_EQ(contracted_coeff_pair_combs_6.size(), 3);

    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> coeff_pair_combs_8 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_8, 0, 0);
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> contracted_coeff_pair_combs_8 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_contracted_combs(coeff_pair_combs_8);
//std::cout << contracted_coeff_pair_combs_8 << std::endl;
    ASSERT_EQ(contracted_coeff_pair_combs_8.size(), 4);

    // Note 检查是否包含 [0, 2], [1, 1], [0, 1]
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> coeff_pair_combs_14 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_all_schemes4lev(aim_level_14, 0, 0);
    std::vector<std::vector<matersdk::mtp::MtpMCoeffPair>> contracted_coeff_pair_14 = 
        matersdk::mtp::MtpMCoeffPairCombs::get_contracted_combs(coeff_pair_combs_14);
//std::cout << contracted_coeff_pair_14 << std::endl;
    ASSERT_EQ(contracted_coeff_pair_14.size(), 26);
}


TEST_F(MtpMCoeffPairCombsTest, size)
{
    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_6(max_level_6);
    ASSERT_EQ(
        coeff_pair_combs_6.size(),
        coeff_pair_combs_6.coeff_pair_combs().size());
    
    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_8(max_level_8);
    ASSERT_EQ(
        coeff_pair_combs_8.size(),
        coeff_pair_combs_8.coeff_pair_combs().size());
}

TEST_F(MtpMCoeffPairCombsTest, nmus)
{
    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_6(max_level_6);
    ASSERT_EQ(coeff_pair_combs_6.nmus(), coeff_pair_combs_6.nmus_check());

    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_8(max_level_8);
    ASSERT_EQ(coeff_pair_combs_8.nmus(), coeff_pair_combs_8.nmus_check());

    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_14(max_level_14);
    ASSERT_EQ(coeff_pair_combs_14.nmus(), coeff_pair_combs_14.nmus_check());

    matersdk::mtp::MtpMCoeffPairCombs coeff_pair_combs_28(max_level_28);
    ASSERT_EQ(coeff_pair_combs_28.nmus(), coeff_pair_combs_28.nmus_check());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}