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
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../include/schmidt_orth.cuh"



class CoeffsSchmidtOrthTest : public ::testing::Test {
protected:
    int ntypes;
    int nmus;
    int chebyshev_size;
    int num_coeffs;
    double *coeffs;
    double *orth_coeffs;

    static void SetUpTestSuite() {
        std::cout << "CoeffsSchmidtOrthTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "CoeffsSchmidtOrthTets (TestSuite) is tearing down...\n";
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
};  // class : CoeffsSchmidtOrthTest



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



TEST_F(CoeffsSchmidtOrthTest, find_coeffs_orthogonal_basis_set) {
    ai2pot::mtpr::find_coeffs_orthogonal_basis_set_launcher(
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