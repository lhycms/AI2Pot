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
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "../include/targetStatistics.h"


typedef double real;


class TargetStatisticsTest : public ::testing::Test
{
protected:
    int *natoms_in_batch_ptr;
    real *ei_mean_ptr;
    real *ei_M2_ptr;
    real *force_M2_ptr;
    int batch_size;
    int natoms_pad;
    int *binum;
    real *betot_dft;
    real (*bforce_dft)[3];


    static void SetUpTestSuite() {
        std::cout << "TargetStatisticsTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "TargetStatisticsTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        natoms_in_batch_ptr = (int*)malloc(sizeof(int));
        (*natoms_in_batch_ptr) = 0.0;
        ei_mean_ptr = (real*)malloc(sizeof(real));
        (*ei_mean_ptr) = 0.0;
        ei_M2_ptr = (real*)malloc(sizeof(real));
        (*ei_M2_ptr) = 0.0;
        force_M2_ptr = (real*)malloc(sizeof(real));
        (*force_M2_ptr) = 0.0;

        batch_size = 12;
        natoms_pad = 30;
        binum = (int*)malloc(sizeof(int) * batch_size);
        betot_dft = (real*)malloc(sizeof(real) * batch_size);
        bforce_dft = (real (*)[3])malloc(sizeof(real) * batch_size * natoms_pad * 3);

        srand(time(NULL));
        for (int bb=0; bb<batch_size; bb++) {
            binum[bb] = rand() % (natoms_pad+1);
            betot_dft[bb] = (real)rand() / RAND_MAX;
        }
        
        for (int bb=0; bb<batch_size; bb++)
            for (int ii=0; ii<binum[bb]; ii++)
                for (int aa=0; aa<3; aa++)
                    bforce_dft[bb*natoms_pad+ii][aa] = (real)rand() / RAND_MAX;
    }

    void TearDown() override {
        free(natoms_in_batch_ptr);
        free(ei_mean_ptr);
        free(ei_M2_ptr);
        free(force_M2_ptr);

        free(binum);
        free(betot_dft);
        free(bforce_dft);
    }
};  // class : TargetStatisticsTest


TEST_F(TargetStatisticsTest, find_ef_statistics) {
    ai2pot::fitutils::TargetStatistics<real>::find_ef_statistics(natoms_in_batch_ptr,
                                                                 ei_mean_ptr,
                                                                 ei_M2_ptr,
                                                                 force_M2_ptr,
                                                                 batch_size,
                                                                 natoms_pad,
                                                                 binum,
                                                                 betot_dft,
                                                                 bforce_dft);
    printf("1. natoms_in_batch = %d\n", (*natoms_in_batch_ptr));
    printf("2. ei_mean = %g\n", (*ei_mean_ptr));
    printf("3. ei_M2 = %g\n", (*ei_M2_ptr));
    printf("4. force_M2 = %g\n", (*force_M2_ptr));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
