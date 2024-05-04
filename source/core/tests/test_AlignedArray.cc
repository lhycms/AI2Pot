#include <gtest/gtest.h>
#include <iostream>

// cmake ..; make -j 8; ./bin/core/test_AlignedArray
#include "../include/AlignedArray.h"


class AlignedArrayTest : public ::testing::Test {
protected:
    matersdk::AlignedArray<float> *ptr_aa_1;
    matersdk::AlignedArray<float> *ptr_aa_2;

    static void SetUpTestSuite() {
        std::cout << "Set up AlignedArrayTest (TestSuite)...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "Tear down AlignedArrayTest (TestSuite)...\n";
    }

    void SetUp() override {
        ptr_aa_1 = new matersdk::AlignedArray<float>();
        ptr_aa_2 = new matersdk::AlignedArray<float>(12);
    }

    void TearDown() override {
        // Automatically call destructor.        
    }

};


TEST_F(AlignedArrayTest, Size) {
    EXPECT_EQ(ptr_aa_1->size(), 0);
    EXPECT_EQ(ptr_aa_2->size(), 12);
}

TEST_F(AlignedArrayTest, Resize) {
    ptr_aa_1->resize(11);
    ptr_aa_2->resize(121);
    EXPECT_EQ(ptr_aa_1->size(), 11);
    EXPECT_EQ(ptr_aa_2->size(), 121);
}


TEST_F(AlignedArrayTest, GetElementByIndex) {
    std::cout << "(*ptr_aa_2)[0] = " << (*ptr_aa_2)[0] << std::endl;
    std::cout << "(*ptr_aa_2)[5] = " << (*ptr_aa_2)[5] << std::endl;
    

    matersdk::AlignedArray<float> tmp_aa(12);
    matersdk::AlignedArray<float> *ptr_tmp_aa;
    ptr_tmp_aa = &tmp_aa;
    EXPECT_EQ(tmp_aa[0], (*ptr_tmp_aa)[0]);
    EXPECT_EQ(tmp_aa[5], (*ptr_tmp_aa)[5]);
    EXPECT_EQ(tmp_aa[10], (*ptr_tmp_aa)[10]);

    (*ptr_tmp_aa)[0] = 101;
    EXPECT_EQ((*ptr_tmp_aa)[0], 101);
}




int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}