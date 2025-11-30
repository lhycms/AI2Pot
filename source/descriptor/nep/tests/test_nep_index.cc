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


class NepIndexTest : public ::testing::Test
{
protected:
    int n_radial_basis;
    int n_angular_basis;
    int l_max;

    static void SetUpTestSuite() {
        std::cout << "NepIndexTest (TestSuite) is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "NepIndexTest (TestSuite) is tearing down...\n";
    }


    void SetUp() override {
        n_radial_basis = 8;
        n_angular_basis = 2;
        l_max = 4;
    }

    void TearDown() override {

    }
};  // class : NepIndexTest


TEST_F(NepIndexTest, test_get_num_descriptors) {
    int num_descriptors = ai2pot::nep::NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
    printf("Number of descriptors = %d\n", num_descriptors);
}


TEST_F(NepIndexTest, test_get_num_Sinlm) {
    int num_Sinlm = ai2pot::nep::NepIndex::get_num_Sinlm(n_angular_basis, l_max);
    printf("Number of Sinlm = %d\n", num_Sinlm);
}

TEST_F(NepIndexTest, test_get_Clm_index) {
    for (int l=1; l<=l_max; l++) {
        for (int mp=0; mp<2*l+1; mp++) {
            int idx_Clm = ai2pot::nep::NepIndex::get_Clm_index(l, mp);
            printf("l = %d, mp = %d, idx_Clm = %d\n", l, mp, idx_Clm);
        }
    }
}


TEST_F(NepIndexTest, test_get_Sinlm_index) {
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            for (int mp=0; mp<2*l+1; mp++) {
                int idx_Sinlm = ai2pot::nep::NepIndex::get_Sinlm_index(l_max, mu, l, mp);
                printf("mu = %d, l = %d, mp = %d, idx_Sinlm = %d\n", mu, l, mp, idx_Sinlm);
            }
        }
    }
}


TEST_F(NepIndexTest, test_get_qinl_index) {
    for (int mu=0; mu<n_angular_basis; mu++) {
        for (int l=1; l<=l_max; l++) {
            int idx_qinl = ai2pot::nep::NepIndex::get_qinl_index(l_max, mu, l);
            printf("mu = %d, l = %d, idx_qinl = %d\n", mu, l, idx_qinl);
        }
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}