#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <set>
#include "../include/mtpParam.h"


class MtpExceptionTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite() {
        std::cout << "MtpException (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpException (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
    }

    void TearDown() override {
    }
};  // class : MtpExceptionTest

class MtpParamTest : public ::testing::Test
{
protected:
    std::string filename;
    std::vector<std::string> filenames;
    int mom_idx;

    static void SetUpTestSuite() {
        std::cout << "MtpParamTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpParamTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        filename = "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/06.almtp";
        filenames = {
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/depreciated-02.almtp", 
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/depreciated-04.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/06.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/08.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/10.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/12.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/14.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/16.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/18.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/20.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/22.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/24.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/26.almtp",
            "/data/home/liuhanyu/hyliu/code/matersdk/source/descriptor/mtpr/MTP_templates/28.almtp"
        };
    }

    void TearDown() override {
    }
};  // class : MtpParam


TEST_F(MtpExceptionTest, throw_exception) {
    try {
        throw matersdk::mtpr::MtpException("gtest: throw mtp exception.");
    } catch (const matersdk::mtpr::MtpException& e) {
        std::cerr << e.what() << std::endl;
    }
}

TEST_F(MtpExceptionTest, MtpError_test) {
    using namespace matersdk::mtpr;

    try {
        MtpError("gtest: throw mtp exception.");
    } catch (const MtpException& e) {
        std::cerr << e.what() << std::endl;
    }
}

TEST_F(MtpParamTest, constructor)
{
    //for (const auto& f : filenames) {
        //matersdk::mtpr::MtpParam mtp_param(f);
//mtp_param.show();
    //}

    matersdk::mtpr::MtpParam mtp_param(filenames[6]);
mtp_param.show();
}

/*
TEST_F(MtpParamTest, load) {
    matersdk::mtpr::MtpParam mtp_param;
    mtp_param._load(filenames[0]);
ASSERT_EQ(mtp_param.alpha_index_times(), nullptr);
//mtp_param.show();
}

TEST_F(MtpParamTest, _get_mus4all_mom)
{
    matersdk::mtpr::MtpParam mtp_param;
    mtp_param._load(filenames[6]);   // mtp_level(4) = 10, mtp_level(6) = 14
    int mom_idx = 172;

    std::set<int> mus_lst = mtp_param._get_mus4all_mom(mom_idx);
    for (auto& v : mus_lst)
        printf("%d, ", v);
    printf("\n");
}


TEST_F(MtpParamTest, _get_mus4all_mom_dp)
{
    matersdk::mtpr::MtpParam mtp_param;
    mtp_param._load(filenames[6]);  // mtp_level(4) = 10, mtp_level(6) = 14
    int num_moms = 174;  

    std::vector<std::set<int>> mus4moms_lst = mtp_param._get_mus4all_mom_dp(num_moms);
    std::set<int> mus_lst = mus4moms_lst[172];
    for (auto& v : mus_lst)
        printf("%d, ", v);
    printf("\n");   
}


TEST_F(MtpParamTest, nmus) {
    matersdk::mtpr::MtpParam mtp_param;
    mtp_param._load(filenames[8]);
//printf("Number of mus given MTP whose level is 18 = %d\n", mtp_param.nmus());
    ASSERT_EQ(mtp_param.nmus(), 5);
}

TEST_F(MtpParamTest, copy_constructor)
{
    matersdk::mtpr::MtpParam mp1(filenames[0]);
    matersdk::mtpr::MtpParam mp2(mp1);

    ASSERT_EQ(mp1.alpha_moments_count(), mp2.alpha_moments_count());
    ASSERT_EQ(mp1.alpha_index_basic_count(), mp2.alpha_index_basic_count());
    ASSERT_EQ(mp1.alpha_index_times_count(), mp2.alpha_index_times_count());
    ASSERT_EQ(mp1.alpha_scalar_moments(), mp2.alpha_scalar_moments());
    for (int ii=0; ii<mp1.alpha_index_basic_count(); ii++)
        for (int jj=0; jj<4; jj++) 
            ASSERT_EQ(
                mp1.alpha_index_basic()[ii][jj],
                mp2.alpha_index_basic()[ii][jj]);
    for (int ii=0; ii<mp1.alpha_index_times_count(); ii++)
        for (int jj=0; jj<4; jj++)
            ASSERT_EQ(
                mp1.alpha_index_times()[ii][jj],
                mp2.alpha_index_times()[ii][jj]);
    for (int ii=0; ii<mp1.alpha_scalar_moments(); ii++)
        ASSERT_EQ(
            mp1.alpha_moment_mapping()[ii],
            mp2.alpha_moment_mapping()[ii]);
}

TEST_F(MtpParamTest, copy_constructor_move)
{
    matersdk::mtpr::MtpParam mp1(filenames[2]);
    matersdk::mtpr::MtpParam mp2(std::move(mp1));
//mp2.show();
    ASSERT_EQ(mp1.alpha_index_basic(), nullptr);
    ASSERT_EQ(mp1.alpha_index_times(), nullptr);
    ASSERT_EQ(mp1.alpha_moment_mapping(), nullptr);
}

TEST_F(MtpParamTest, assignment_operator)
{
    matersdk::mtpr::MtpParam mp1(filenames[2]);
    matersdk::mtpr::MtpParam mp2(filenames[0]);
    mp2 = mp1;

    ASSERT_EQ(mp1.alpha_moments_count(), mp2.alpha_moments_count());
    ASSERT_EQ(mp1.alpha_index_basic_count(), mp2.alpha_index_basic_count());
    ASSERT_EQ(mp1.alpha_index_times_count(), mp2.alpha_index_times_count());
    ASSERT_EQ(mp1.alpha_scalar_moments(), mp2.alpha_scalar_moments());
    for (int ii=0; ii<mp1.alpha_index_basic_count(); ii++)
        for (int jj=0; jj<4; jj++) 
            ASSERT_EQ(
                mp1.alpha_index_basic()[ii][jj],
                mp2.alpha_index_basic()[ii][jj]);
    for (int ii=0; ii<mp1.alpha_index_times_count(); ii++)
        for (int jj=0; jj<4; jj++)
            ASSERT_EQ(
                mp1.alpha_index_times()[ii][jj],
                mp2.alpha_index_times()[ii][jj]);
    for (int ii=0; ii<mp1.alpha_scalar_moments(); ii++)
        ASSERT_EQ(
            mp1.alpha_moment_mapping()[ii],
            mp2.alpha_moment_mapping()[ii]);
}

TEST_F(MtpParamTest, assignment_operator_move)
{
    matersdk::mtpr::MtpParam mp1(filenames[2]);
    matersdk::mtpr::MtpParam mp2(filenames[0]);
    //mp2 = std::move(mp1);

//mp2.show();
    //ASSERT_EQ(mp1.alpha_index_basic(), nullptr);
    //ASSERT_EQ(mp1.alpha_index_times(), nullptr);
    //ASSERT_EQ(mp1.alpha_moment_mapping(), nullptr);
}


TEST_F(MtpParamTest, AlphaIndexBasic_copy_constrcutor)
{
    matersdk::mtpr::MtpParam mp;
    mp._load(filenames[4]);  // mtp_level = 10
    
    matersdk::mtpr::AlphaIndexTimes at1(
        (int)mp.alpha_index_times_count(), 
        (int(*)[4])mp.alpha_index_times());
//at1.show();
    matersdk::mtpr::AlphaIndexTimes at2(at1);

    ASSERT_EQ(at1.alpha_index_times_count(), at2.alpha_index_times_count());
    for (int ii=0; ii<at1.alpha_index_times_count(); ii++)
        for (int jj=0; jj<4; jj++)
            ASSERT_EQ(at1.alpha_index_times()[ii][jj], at2.alpha_index_times()[ii][jj]);
}

TEST_F(MtpParamTest, AlphaIndexBasic_copy_constructor_move)
{
    matersdk::mtpr::MtpParam mp;
    mp._load(filenames[4]); // mtp_level = 10

    matersdk::mtpr::AlphaIndexTimes at1(
        (int)mp.alpha_index_times_count(),
        (int(*)[4])mp.alpha_index_times());
    matersdk::mtpr::AlphaIndexTimes at2(std::move(at1));

    ASSERT_EQ(at1.alpha_index_times_count(), 0);
    ASSERT_EQ(at1.alpha_index_times(), nullptr);
}

TEST_F(MtpParamTest, AlphaIndexBasic_assignment_operator)
{
    matersdk::mtpr::MtpParam mp;
    mp._load(filenames[4]); // mtp_level = 10
    matersdk::mtpr::AlphaIndexTimes at1(
        (int)mp.alpha_index_times_count(),
        (int(*)[4])mp.alpha_index_times());
    
    mp._load(filenames[5]); // mtp_level = 12
    matersdk::mtpr::AlphaIndexTimes at2(
        (int)mp.alpha_index_times_count(),
        (int(*)[4])mp.alpha_index_times());
    
    at2 = at1;
    ASSERT_EQ(at1.alpha_index_times_count(), at2.alpha_index_times_count());
    for (int ii=0; ii<at1.alpha_index_times_count(); ii++)
        for (int jj=0; jj<4; jj++)
            ASSERT_EQ(at1.alpha_index_times()[ii][jj], at2.alpha_index_times()[ii][jj]);
}

TEST_F(MtpParamTest, AlphaIndexTimes_assignment_operator_move)
{
    matersdk::mtpr::MtpParam mp;
    mp._load(filenames[4]); // mtp_level = 10
    matersdk::mtpr::AlphaIndexTimes at1(
        (int)mp.alpha_index_times_count(),
        (int(*)[4])mp.alpha_index_times());

    mp._load(filenames[5]); // mtp_level = 12
    matersdk::mtpr::AlphaIndexTimes at2(
        (int)mp.alpha_index_times_count(),
        (int(*)[4])mp.alpha_index_times());
    
    at2 = std::move(at1);
    ASSERT_EQ(at1.alpha_index_times_count(), 0);
    ASSERT_EQ(at1.alpha_index_times(), nullptr);
}
*/


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}