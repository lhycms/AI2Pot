#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/schmidt_orth.h"


class SchmidtOrthTest : public ::testing::Test {
protected:
    int ntypes;
    int nmus;
    int chebyshev_size;
    int num_coeffs;
    double *coeffs;
    double *orth_coeffs;

    static void SetUpTestSuite() {
        std::cout << "SchmidtOrthTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "SchmidtOrthTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        ntypes = 2;
        nmus = 3;
        chebyshev_size = 8;
        num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
        coeffs = (double*)malloc(sizeof(double) * num_coeffs);
        orth_coeffs = (double*)malloc(sizeof(double) * num_coeffs);

        for (int t1=0; t1<ntypes; t1++) {
            for (int t2=0; t2<ntypes; t2++) {
                for (int mu=0; mu<nmus; mu++) {
                   for (int xi=0; xi<chebyshev_size; xi++) {
                        int idx = (t1*ntypes+t2)*nmus*chebyshev_size
                                  + mu*chebyshev_size + xi;
                        coeffs[idx] = t1*0.01 + t2*0.02 + mu*0.04 + xi*0.05
                                      + 0.001*((double)rand()/RAND_MAX); // 小扰动
                   }
                }
            }
        }
    }

    void TearDown() override {
        free(orth_coeffs);
        free(coeffs);
    }
};  // class : SchmidtOrthTest


void check_orthogonality(
    double *orth_coeffs,
    int ntypes,
    int nmus,
    int chebyshev_size,
    int mu1,
    int mu2)
{
    double scal = 0.0;
    for (int type_central=0; type_central<ntypes; type_central++) {
        for (int type_outer=0; type_outer<ntypes; type_outer++) {
            for (int xi=0; xi<chebyshev_size; xi++) {
                int idx1 = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                           + mu1*chebyshev_size + xi;
                int idx2 = (type_central*ntypes+type_outer)*nmus*chebyshev_size
                           + mu2*chebyshev_size + xi;
                scal += orth_coeffs[idx1] * orth_coeffs[idx2];
            }
        }
    }
    printf("1. scal of %d and %d: = %.10lf\n", mu1, mu2, scal);
}





TEST_F(SchmidtOrthTest, find_orthogonal_basis_set) {    
    ai2pot::mtpr::CoeffsSchmidtOrth<double>::find_orthogonal_basis_set(
        orth_coeffs,
        ntypes,
        nmus,
        chebyshev_size,
        coeffs);
    
    for (int k1=0; k1<nmus; k1++) {
        for (int k2=0; k2<nmus; k2++) {
            check_orthogonality(orth_coeffs,
                               ntypes,
                               nmus,
                               chebyshev_size,
                               k1, 
                               k2);
        }
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
