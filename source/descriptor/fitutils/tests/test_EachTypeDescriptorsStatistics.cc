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


class EachTypeDescriptorsStatisticsTest : public ::testing::Test
{
protected:
    int *natoms_in_batch;
    real *descriptors_mean;
    real *descriptors_M2;
    int batch_size;
    int natoms_pad;
    int descriptor_dim;
    int *binum;
    int *bilist;
    int *btypes;
    int ntypes;
    real *bdescriptors;


    static void SetUpTestSuite() {
        std::cout << "EachTypeDescriptorsStatisticsTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "EachTypeDescriptorsStatisticsTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        batch_size = 32;
        natoms_pad = 30;
        descriptor_dim = 9;
        ntypes = 1;

        natoms_in_batch = (int*)malloc(sizeof(int) * ntypes);
        descriptors_mean = (real*)malloc(sizeof(real) * ntypes * descriptor_dim);
        descriptors_M2 = (real*)malloc(sizeof(real) * ntypes * descriptor_dim);
        
        binum = (int*)malloc(sizeof(int) * batch_size);
        srand(time(NULL));
        for (int bb=0; bb<batch_size; bb++)
            binum[bb] = rand() % (natoms_pad+1);

        bilist = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        btypes = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        for (int bb=0; bb<batch_size; bb++) {
            for (int ii=0; ii<binum[bb]; ii++) {
                bilist[bb*natoms_pad + ii] = ii;
                btypes[bb*natoms_pad + ii] = 0;
            }
        }

        bdescriptors = (real*)malloc(sizeof(real) * batch_size * natoms_pad * descriptor_dim);
        for (int bb=0; bb<batch_size; bb++)
            for (int ii=0; ii<binum[bb]; ii++)
                for (int kk=0; kk<descriptor_dim; kk++)
                    bdescriptors[bb*natoms_pad*descriptor_dim + ii*descriptor_dim + kk] = (real)rand() / RAND_MAX;
    }

    void TearDown() override {
        free(natoms_in_batch);
        free(descriptors_mean);
        free(descriptors_M2);

        free(binum);
        free(bilist);
        free(btypes);
        free(bdescriptors);
    }
};  // class : EachTypeDescriptorsStatisticsTest



TEST_F(EachTypeDescriptorsStatisticsTest, find_descriptors_statistics) {
    ai2pot::fitutils::EachTypeDescriptorsStatistics<real>::find_descriptors_statistics(
        natoms_in_batch,
        descriptors_mean,
        descriptors_M2,
        batch_size,
        natoms_pad,
        descriptor_dim,
        binum,
        bilist,
        btypes,
        ntypes,
        bdescriptors);
    
printf("1. natoms_in_batch:\n\t");
for (int tt=0; tt<ntypes; tt++)
    printf("%d, ", natoms_in_batch[tt]);
printf("\n");

printf("2. descriptors_mean:\n\t");
for (int tt=0; tt<ntypes; tt++) {
    for (int kk=0; kk<descriptor_dim; kk++)
        printf("%g, ", descriptors_mean[tt*descriptor_dim + kk]);
}
printf("\n");

printf("3. descriptors_M2:\n\t");
for (int tt=0; tt<ntypes; tt++) {
    for (int kk=0; kk<descriptor_dim; kk++)
        printf("%g, ", descriptors_M2[tt*descriptor_dim + kk]);
}
printf("\n");
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}