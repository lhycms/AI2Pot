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
#include <math.h>
#include <time.h>

#include "../include/descriptorsStatistics.h"

typedef double real;

class AllTypeDescriptorsStatisticsTest : public ::testing::Test
{
protected:
    int *natoms_in_batch_ptr;
    real *descriptors_mean;
    real *descriptors_M2;
    real *descriptors_max;
    real *descriptors_min;

    int batch_size;
    int natoms_pad;
    int descriptor_dim;
    int *binum;
    real *bdescriptors;

    static void SetUpTestSuite() {
        std::cout << "AllTypeDescriptorsStatisticsTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "AllTypeDescriptorsStatisticsTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        descriptor_dim = 10;
        natoms_in_batch_ptr = (int*)malloc(sizeof(real));
        descriptors_mean = (real*)malloc(sizeof(real) * descriptor_dim);
        descriptors_M2 = (real*)malloc(sizeof(real) * descriptor_dim);
        descriptors_max = (real*)malloc(sizeof(real) * descriptor_dim);
        descriptors_min = (real*)malloc(sizeof(real) * descriptor_dim);
        
        batch_size = 33;
        natoms_pad = 50;
        binum = (int*)malloc(sizeof(int) * batch_size);
        srand(time(NULL));
        for (int bb=0; bb<batch_size; bb++)
            binum[bb] = rand() % (natoms_pad+1);
        
        bdescriptors = (real*)malloc(sizeof(real) * batch_size * natoms_pad * descriptor_dim);
        for (int bb=0; bb<batch_size; bb++)
            for (int ii=0; ii<binum[bb]; ii++)
                for (int kk=0; kk<descriptor_dim; kk++)
                    bdescriptors[bb*natoms_pad*descriptor_dim + ii*descriptor_dim + kk] = (real)rand() / RAND_MAX;
    }

    void TearDown() override {
        free(natoms_in_batch_ptr);
        free(descriptors_mean);
        free(descriptors_M2);
        free(descriptors_max);
        free(descriptors_min);
        free(binum);
        free(bdescriptors);
    }
};  // class : AllTypeDescriptorsStatisticsTest


TEST_F(AllTypeDescriptorsStatisticsTest, find_descriptors_statistics) {
    ai2pot::fitutils::AllTypeDescriptorsStatistics<real>::find_descriptors_statistics(
        natoms_in_batch_ptr,
        descriptors_mean,
        descriptors_M2,
        batch_size,
        natoms_pad,
        descriptor_dim,
        binum,
        bdescriptors);
    
    printf("\t1. natoms_in_batch = %d\n", *natoms_in_batch_ptr);
    printf("\t2. descriptors_mean: ");
    for (int kk=0; kk<descriptor_dim; kk++)
        printf("%g, ", descriptors_mean[kk]);
    printf("\n");
    printf("\t3. descriptors_M2: ");
    for (int kk=0; kk<descriptor_dim; kk++)
        printf("%g, ", descriptors_M2[kk]);
    printf("\n");
}


TEST_F(AllTypeDescriptorsStatisticsTest, find_descriptors_maxmin) {
    ai2pot::fitutils::AllTypeDescriptorsMaxmin<real>::find_descriptors_maxmin(
        descriptors_max,
        descriptors_min,
        batch_size,
        natoms_pad,
        descriptor_dim,
        binum,
        bdescriptors);
    
    printf("\t1. descriptors_max: ");
    for (int kk=0; kk<descriptor_dim; kk++)
        printf("%g, ", descriptors_max[kk]);
    printf("\n");
    printf("\t2. descriptors_min: ");
    for (int kk=0; kk<descriptor_dim; kk++)
        printf("%g, ", descriptors_min[kk]);
    printf("\n");
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
