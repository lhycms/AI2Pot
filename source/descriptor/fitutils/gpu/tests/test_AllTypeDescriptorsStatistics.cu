#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include "../include/descriptorsStatistics.cuh"

typedef double real;


class AllTypeDescriptorsStatisticsTest : public ::testing::Test
{
protected:
    int *natoms_in_batch_ptr;
    real *descriptors_mean;
    real *descriptors_M2;
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
        batch_size = 33;
        natoms_pad = 50;
        descriptor_dim = 10;

        natoms_in_batch_ptr = (int*)malloc(sizeof(int));
        descriptors_mean = (real*)malloc(sizeof(real) * descriptor_dim);
        descriptors_M2 = (real*)malloc(sizeof(real) * descriptor_dim);

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
        free(binum);
        free(bdescriptors);
    }
};  // class : AllTypeDescriptorsStatisticsTest


TEST_F(AllTypeDescriptorsStatisticsTest, find_all_type_descriptors_statistics_launcher) {
    ai2pot::fitutils::find_all_type_descriptors_statistics_launcher(
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


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}