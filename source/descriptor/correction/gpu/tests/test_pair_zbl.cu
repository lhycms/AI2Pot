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
#include <stdio.h>
#include <iostream>

#include "../include/zbl.cuh"


class PairZBLTest : public ::testing::Test
{
protected:
    int Zi;
    int Zj;
    double rmax;
    double rmin;
    double *ck;
    double *dk;
    double coord_1[3];
    double coord_2[3];
    double neigh_vec[3];
    double force[3];
    double virial[9];

    static void SetUpTestSuite() {
        std::cout << "PairZBLTest (TestSuite) is setting up...\n";
    }

    static void TearDownSuite() {
        std::cout << "PairZBLTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        Zi = 20;
        Zj = 24;
        rmax = 2.0;
        rmin = 1.0;
        ck = (double*)malloc(sizeof(double) * 4);
        dk = (double*)malloc(sizeof(double) * 4);
        ck[0] = 0.18175;
        ck[1] = 0.50986;
        ck[2] = 0.28022;
        ck[3] = 0.02817;

        dk[0] = 3.1998;
        dk[1] = 0.94229;
        dk[2] = 0.4029;
        dk[3] = 0.20162;

        coord_1[0] = 1.0;
        coord_1[1] = 1.0;
        coord_1[2] = 1.0;
        coord_2[0] = 2.01;
        coord_2[1] = 2.02;
        coord_2[2] = 2.03;
        neigh_vec[0] = coord_2[0] - coord_1[0];
        neigh_vec[1] = coord_2[1] - coord_1[1];
        neigh_vec[2] = coord_2[2] - coord_1[2];

        force[0] = 0;
        force[1] = 0;
        force[2] = 0;

        for (int ii=0; ii<9; ii++)
            virial[ii] = 0.0;
    }

    void TearDown() override {
        free(ck);
        free(dk);
    }
};  // class : PairTest


TEST_F(PairZBLTest, find_switch_func) {
    double result = ai2pot::correction::PairZBL<double>::find_switch_func(rmax, rmin, 1.0);
    ASSERT_EQ(result, 1.0);
    result = ai2pot::correction::PairZBL<double>::find_switch_func(rmax, rmin, 2.0);
    ASSERT_EQ(result, 0.0);
}


TEST_F(PairZBLTest, find_switch_func_gradient) {
    double delta = 1e-5;

    double result = ai2pot::correction::PairZBL<double>::find_switch_func(rmax, rmin, 1.5);
    double gradient = ai2pot::correction::PairZBL<double>::find_switch_func_der2rij(rmax, rmin, 1.5);
    double result_ = ai2pot::correction::PairZBL<double>::find_switch_func(rmax, rmin, 1.5+delta);

printf("\t1. Gradient calculated by custom code = %.10f\n", gradient);
printf("\t2. Gradient calculated by finite difference method = %.10f\n", (result_ - result) / delta);
}


TEST_F(PairZBLTest, find_phi_func_gradient) {
    double delta = 1e-5;

    double result = ai2pot::correction::PairZBL<double>::find_phi_func(Zi, Zj, 1.1, ck, dk);
    double gradient = ai2pot::correction::PairZBL<double>::find_phi_func_der2rij(Zi, Zj, 1.1, ck, dk);
    double result_ = ai2pot::correction::PairZBL<double>::find_phi_func(Zi, Zj, 1.1+delta, ck, dk);

printf("\t1. Gradient calculated by custom code = %.10f\n", gradient);
printf("\t2. Gradient calculated by finite difference method = %.10f\n", (result_ - result) / delta);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}