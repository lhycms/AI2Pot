#include <gtest/gtest.h>
#include <iostream>

// cmake -DBUILD_TEST=1 ..; make -j 8; ./bin/core/test_vecx 
#include "../include/vecx.h"


class Vec3Test : public ::testing::Test {
protected:
    matersdk::Vec3 *ptr_vec3_1;
    matersdk::Vec3 *ptr_vec3_2;

    static void SetUpTestSuite() {
        std::cout << "Set up Vec3Test (TestSuite)...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "Tear down Vec3Test (TestSuite)...\n";
    }

    void SetUp() override {
        ptr_vec3_1 = new matersdk::Vec3(1.0, 2.0, 3.0);
        ptr_vec3_2 = new matersdk::Vec3(1.0, 2.0, 3.0);
    }

    void TearDown() override {
        delete ptr_vec3_1;
        delete ptr_vec3_2;
    }
};


/**
 * @brief Construct a new test f object 
 * for `matersdk::Vec3::operator[]`
 * 
 */
TEST_F(Vec3Test, Index) {
    EXPECT_EQ((*ptr_vec3_1)[0], 1.0);
    EXPECT_EQ((*ptr_vec3_1)[1], 2.0);
    EXPECT_EQ((*ptr_vec3_1)[2], 3.0);


    //const matersdk::Vec3 vec3_const(1.0, 2.0, 3.0);
    //double &result = vec3_const[0];
    //result += 1;
    //EXPECT_EQ(vec3_const[0], 1.0);

    matersdk::Vec3 vec3_nonconst(1.0, 2.0, 3.0);
    double &result = vec3_nonconst[0];
    result += 1;
    EXPECT_EQ(vec3_nonconst[0], 2.0);
}

/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3::operator==` and `matersdk::Vec3::operator!=`
 * 
 */
TEST_F(Vec3Test, EqNe) {
    EXPECT_EQ(*(ptr_vec3_1), *(ptr_vec3_2));

    matersdk::Vec3 vec3_3(2.0, 3.0, 4.0);
    EXPECT_NE(*(ptr_vec3_1), vec3_3);
}


/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3::operator+` (Unary plus)
 * 
 */
TEST_F(Vec3Test, UnaryPlus) {
    EXPECT_EQ(+(*ptr_vec3_1), (*ptr_vec3_1));
}


/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3::operator+` (Binary plus)
 * 
 */
TEST_F(Vec3Test, BinaryPlus) {
    matersdk::Vec3 result = (*ptr_vec3_1) + (*ptr_vec3_2);
    EXPECT_EQ(result, matersdk::Vec3(2.0, 4.0, 6.0));
}

/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3::operator+=`
 * 
 */
TEST_F(Vec3Test, SelfPlus) {
    (*ptr_vec3_1) += (*ptr_vec3_1);
    EXPECT_EQ((*ptr_vec3_1), matersdk::Vec3(2.0, 4.0, 6.0));
}

/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3::operator-` (unary minus)
 * 
 */
TEST_F(Vec3Test, UnaryMinus) {
    EXPECT_EQ(-(*ptr_vec3_1), matersdk::Vec3(-1.0, -2.0, -3.0));
}


/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3::operator-` (binary minus)
 * 
 */
TEST_F(Vec3Test, BinaryMinus) {
    matersdk::Vec3 result = (*ptr_vec3_1) - (*ptr_vec3_2);
    EXPECT_EQ(result, matersdk::Vec3(0.0, 0.0, 0.0));
}


/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3::operator-=`
 */
TEST_F(Vec3Test, SelfMinus) {
    (*ptr_vec3_1) -= (*ptr_vec3_2);
    EXPECT_EQ((*ptr_vec3_1), matersdk::Vec3(0.0, 0.0, 0.0));
}

/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3` scalar product
 * 
 */
TEST_F(Vec3Test, MultiScalar) {
    matersdk::Vec3 result = (*ptr_vec3_1) * 3;
    EXPECT_EQ(result, matersdk::Vec3(3.0, 6.0, 9.0));
}

/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3` self scalar product
 * 
 */
TEST_F(Vec3Test, SelfMultiScalar) {
    (*ptr_vec3_1) *= 3;
    EXPECT_EQ((*ptr_vec3_1), matersdk::Vec3(3.0, 6.0, 9.0));
}


/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3` scalar division
 * 
 */
TEST_F(Vec3Test, DivScalar) {
    matersdk::Vec3 result = (*ptr_vec3_1) / 0.5;
    EXPECT_EQ(result, matersdk::Vec3(2.0, 4.0, 6.0));
}

/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3` self scalar division
 * 
 */
TEST_F(Vec3Test, SelfDivScalar) {
    (*ptr_vec3_1) /= 0.5;
    EXPECT_EQ((*ptr_vec3_1), matersdk::Vec3(2.0, 4.0, 6.0));
}


/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3` dot
 * 
 */
TEST_F(Vec3Test, Dot) {
    double result = ptr_vec3_1->dot(*ptr_vec3_1);
    EXPECT_EQ(result, 14.0);
}


/**
 * @brief Construct a new test f object
 * for `matersdk::Vec3` cross
 * 
 */
TEST_F(Vec3Test, Cross) {
    matersdk::Vec3 vec3_3(2.0, 3.0, 4.0);
    matersdk::Vec3 result = ptr_vec3_1->cross(vec3_3);
    EXPECT_EQ(result, matersdk::Vec3(-1, 2, -1));
}


/**
 * @brief Construct a new test f object
 * for `matersdk::operator*(double, const Vec3&)`
 * 
 */
TEST_F(Vec3Test, ScalarMulti) {
    matersdk::Vec3 result = 2.0 * (*ptr_vec3_1);
    EXPECT_EQ(result, matersdk::Vec3(2.0, 4.0, 6.0));
}

TEST_F(Vec3Test, PrintVec3) {
    std::cout << (*ptr_vec3_1) << std::endl;
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}