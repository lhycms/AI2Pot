#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include "../include/vec3Operation.h"


class Vec3OperationPointerTest : public ::testing::Test {
protected:
    double* vec1;
    double* vec2;

    static void SetUpTestSuite() {
        std::cout << "Vec3OperationTest is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "Vec3OperationTest is tearing down...\n";
    }


    void SetUp() override {
        vec1 = (double*)malloc(sizeof(double) * 3);
        vec2 = (double*)malloc(sizeof(double) * 3);
        vec1[0] = 1;
        vec1[1] = 2;
        vec1[2] = 3;
        vec2[0] = 2;
        vec2[1] = 3;
        vec2[2] = 4;
    }

    
    void TearDown() override {
        free(vec1);
        free(vec2);
    }
};


TEST_F(Vec3OperationPointerTest, dot) {
    double inner_product = matersdk::vec3Operation::dot<double>(vec1, vec2);
    EXPECT_EQ(inner_product, 20);
}


TEST_F(Vec3OperationPointerTest, RecursionOuterProduct_1) {
    int nju = 1;
    double* vec = (double*)malloc(sizeof(double) * 3);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;

    double** result = matersdk::vec3Operation::RecursionOuterProduct<double>(nju, vec);

    for (int ii=0; ii<std::pow(3, nju-1); ii++) {
        for (int jj=0; jj<3; jj++) {
            printf("%10f, ", result[ii][jj]);
        }
        printf("\n");
    }
}


TEST_F(Vec3OperationPointerTest, RecursionOuterProduct_2) {
    int nju = 2;
    double* vec = (double*)malloc(sizeof(double) * 3);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;

    double** result = matersdk::vec3Operation::RecursionOuterProduct<double>(nju, vec);

    for (int ii=0; ii<std::pow(3, nju-1); ii++) {
        for (int jj=0; jj<3; jj++) {
            printf("%10f, ", result[ii][jj]);
        }
        printf("\n");
    }
}


TEST_F(Vec3OperationPointerTest, RecursionOuterProduct_3) {
    int nju = 4;
    double* vec = (double*)malloc(sizeof(double) * 3);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;

    double** result = matersdk::vec3Operation::RecursionOuterProduct<double>(nju, vec);

    for (int ii=0; ii<std::pow(3, nju-1); ii++) {
        for (int jj=0; jj<3; jj++) {
            printf("%10f, ", result[ii][jj]);
        }
        printf("\n");
    }
}


TEST_F(Vec3OperationPointerTest, cross) {
    double *vertical_vec = matersdk::vec3Operation::cross<double>(vec1, vec2);

    EXPECT_EQ(matersdk::vec3Operation::dot<double>(vertical_vec, vec1), 0);
    EXPECT_EQ(matersdk::vec3Operation::dot<double>(vertical_vec, vec2), 0);

    free(vertical_vec);
}


TEST_F(Vec3OperationPointerTest, norm) {
    double vec_length_1 = matersdk::vec3Operation::norm<double>(vec1);
    printf("vec_length_1 = %f\n", vec_length_1);
}



TEST_F(Vec3OperationPointerTest, normalize) {
    double* unit_vec_1 = matersdk::vec3Operation::normalize(vec1);
    double* unit_vec_2 = matersdk::vec3Operation::normalize(vec2);

    double vec_length_1 = matersdk::vec3Operation::norm(unit_vec_1);
    double vec_length_2 = matersdk::vec3Operation::norm(unit_vec_2);
    printf("%f\n", vec_length_1);
    printf("%f\n", vec_length_2);
    //EXPECT_EQ(vec_length_1, 1.0);
    //EXPECT_EQ(vec_length_2, 1.0);

    free(unit_vec_1);
    free(unit_vec_2);
}





class Vec3OperationArrayTest : public ::testing::Test {
protected:
    double vec1[3];
    double vec2[3];

    static void SetUpTestSuite() {
        std::cout << "Vec3OperationArrayTest is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "Vec3OperationArrayTest is tearing down...\n";
    }


    void SetUp() override {
        vec1[0] = 1;
        vec1[1] = 2;
        vec1[2] = 3;
        vec2[0] = 2;
        vec2[1] = 3;
        vec2[2] = 4;
    }


    void TearDown() override {

    }
};  // class: vec3Operation


TEST_F(Vec3OperationArrayTest, dot) {
    double inner_product = matersdk::vec3Operation::dot<double>(vec1, vec2);
    EXPECT_EQ(inner_product, 20);
}


TEST_F(Vec3OperationArrayTest, cross) {
    /*
        error: array must be initialized with a brace-enclosed initializer
    */

    // double vertical_vec[3] = matersdk::vec3Operation::cross<double>(vec1, vec2);
    // EXPECT_EQ(matersdk::vec3Operation::dot<double>(vertical_vec, vec1), 0);
    // EXPECT_EQ(matersdk::vec3Operation::dot<double>(vertical_vec, vec2), 0);
}



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}