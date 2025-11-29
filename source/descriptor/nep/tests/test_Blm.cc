/*
    Copyright 2025 Hanyu Liu
    This file is part of AI2Pot.
    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../include/nep_utilities.h"

class BlmTest : public ::testing::Test
{
protected:
    double blm_val1;
    double blm_val2;
    double delta;
    double *rc1;
    double *rc2;
    double *blm_der2xyz;

    static void SetUpTestSuite() {
        std::cout << "BlmTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "BlmTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        blm_der2xyz = (double*)malloc(sizeof(double) * 3);

        blm_val1 = 0.0;
        blm_val2 = 0.0;
        delta = 1e-5;

        rc1 = (double*)malloc(sizeof(double) * 3);
        rc1[0] = 1.37429;
        rc1[1] = 3.98653;
        rc1[2] = 4.09821;
        rc2 = (double*)malloc(sizeof(double) * 3);
    }

    void TearDown() override {
        free(blm_der2xyz);
        free(rc1);
        free(rc2);
    }
};  // class : BlmTest


TEST_F(BlmTest, test_b10_der2x) {
    int l = 1;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B10_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B10_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b10_der2y) {
    int l = 1;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B10_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B10_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b10_der2z) {
    int l = 1;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B10_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B10_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b11_der2x) {
    int l = 1;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B11_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B11_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b11_der2y) {
    int l = 1;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B11_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B11_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b11_der2z) {
    int l = 1;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B11_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B11_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b12_der2x) {
    int l = 1;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B12_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B12_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b12_der2y) {
    int l = 1;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B12_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B12_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b12_der2z) {
    int l = 1;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B12_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B12_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b20_der2x) {
    int l = 2;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B20_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B20_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b20_der2y) {
    int l = 2;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B20_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B20_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b20_der2z) {
    int l = 2;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B20_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B20_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b21_der2x) {
    int l = 2;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B21_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B21_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b21_der2y) {
    int l = 2;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B21_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B21_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b21_der2z) {
    int l = 2;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B21_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B21_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b22_der2x) {
    int l = 2;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B22_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B22_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b22_der2y) {
    int l = 2;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B22_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B22_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b22_der2z) {
    int l = 2;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B22_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B22_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b23_der2x) {
    int l = 2;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B23_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B23_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b23_der2y) {
    int l = 2;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B23_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B23_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b23_der2z) {
    int l = 2;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B23_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B23_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b24_der2x) {
    int l = 2;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B24_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B24_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b24_der2y) {
    int l = 2;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B24_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B24_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b24_der2z) {
    int l = 2;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B24_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B24_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b30_der2x) {
    int l = 3;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B30_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B30_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b30_der2y) {
    int l = 3;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B30_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B30_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b30_der2z) {
    int l = 3;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B30_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B30_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b31_der2x) {
    int l = 3;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B31_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B31_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b31_der2y) {
    int l = 3;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B31_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B31_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b31_der2z) {
    int l = 3;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B31_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B31_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b32_der2x) {
    int l = 3;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B32_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B32_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b32_der2y) {
    int l = 3;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B32_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B32_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b32_der2z) {
    int l = 3;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B32_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B32_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b33_der2x) {
    int l = 3;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B33_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B33_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b33_der2y) {
    int l = 3;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B33_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B33_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b33_der2z) {
    int l = 3;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B33_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B33_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b34_der2x) {
    int l = 3;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B34_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B34_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b34_der2y) {
    int l = 3;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B34_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B34_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b34_der2z) {
    int l = 3;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B34_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B34_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b35_der2x) {
    int l = 3;
    int mp = 5;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B35_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B35_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b35_der2y) {
    int l = 3;
    int mp = 5;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B35_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B35_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b35_der2z) {
    int l = 3;
    int mp = 5;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B35_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B35_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b36_der2x) {
    int l = 3;
    int mp = 6;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B36_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B36_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b36_der2y) {
    int l = 3;
    int mp = 6;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B36_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B36_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b36_der2z) {
    int l = 3;
    int mp = 6;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B36_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B36_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b40_der2x) {
    int l = 4;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B40_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B40_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b40_der2y) {
    int l = 4;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B40_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B40_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b40_der2z) {
    int l = 4;
    int mp = 0;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B40_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B40_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b41_der2x) {
    int l = 4;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B41_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B41_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b41_der2y) {
    int l = 4;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B41_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B41_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b41_der2z) {
    int l = 4;
    int mp = 1;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B41_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B41_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b42_der2x) {
    int l = 4;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B42_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B42_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b42_der2y) {
    int l = 4;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B42_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B42_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b42_der2z) {
    int l = 4;
    int mp = 2;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B42_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B42_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b43_der2x) {
    int l = 4;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B43_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B43_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b43_der2y) {
    int l = 4;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B43_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B43_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b43_der2z) {
    int l = 4;
    int mp = 3;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B43_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B43_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b44_der2x) {
    int l = 4;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B44_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B44_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b44_der2y) {
    int l = 4;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B44_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B44_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b44_der2z) {
    int l = 4;
    int mp = 4;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B44_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B44_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b45_der2x) {
    int l = 4;
    int mp = 5;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B45_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B45_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b45_der2y) {
    int l = 4;
    int mp = 5;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B45_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B45_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b45_der2z) {
    int l = 4;
    int mp = 5;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B45_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B45_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b46_der2x) {
    int l = 4;
    int mp = 6;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B46_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B46_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b46_der2y) {
    int l = 4;
    int mp = 6;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B46_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B46_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b46_der2z) {
    int l = 4;
    int mp = 6;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B46_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B46_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b47_der2x) {
    int l = 4;
    int mp = 7;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B47_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B47_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b47_der2y) {
    int l = 4;
    int mp = 7;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B47_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B47_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b47_der2z) {
    int l = 4;
    int mp = 7;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B47_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B47_der2z calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b48_der2x) {
    int l = 4;
    int mp = 8;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0] + delta;
    rc2[1] = rc1[1];
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B48_der2x calculated by custom code = %.10lf\n", blm_der2xyz[0]);
    printf("\t2. B48_der2x calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b48_der2y) {
    int l = 4;
    int mp = 8;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1] + delta;
    rc2[2] = rc1[2];
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B48_der2y calculated by custom code = %.10lf\n", blm_der2xyz[1]);
    printf("\t2. B48_der2y calculated by fdm = %.10lf\n", blm_der);
}

TEST_F(BlmTest, test_b48_der2z) {
    int l = 4;
    int mp = 8;
    
    double distance_ij_1 = std::sqrt(std::pow(rc1[0], 2) + std::pow(rc1[1], 2) + std::pow(rc1[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val1, l, mp, rc1, distance_ij_1);
    ai2pot::nep::Blm<double>::find_blm_der2xyz(blm_der2xyz, l, mp, rc1, distance_ij_1);

    rc2[0] = rc1[0];
    rc2[1] = rc1[1];
    rc2[2] = rc1[2] + delta;
    double distance_ij_2 = std::sqrt(std::pow(rc2[0], 2) + std::pow(rc2[1], 2) + std::pow(rc2[2], 2));
    ai2pot::nep::Blm<double>::find_blm_val(&blm_val2, l, mp, rc2, distance_ij_2);

    double blm_der = (blm_val2 - blm_val1) / delta;
    printf("\t1. B48_der2z calculated by custom code = %.10lf\n", blm_der2xyz[2]);
    printf("\t2. B48_der2z calculated by fdm = %.10lf\n", blm_der);
}






int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
