#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>

#include "../include/basis.h"


class RQ_ChebyshevTest : public ::testing::Test
{
protected:
    int size;
    double rmax;
    double distance_ij;
    double delta;

    static void SetUpTestSuite() {
        std::cout << "RQ_ChebyshevTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "RQ_ChebyshevTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        size = 8;
        rmax = 5.0;
        distance_ij = 3.14;
        delta = 1e-3;
    }

    void TearDown() override {}
};  // class : RQ_ChebyshevTest


TEST_F(RQ_ChebyshevTest, der_accuracy) {
    ai2pot::nep::RQ_Chebyshev<double> *p_RadialBasis_1;
    ai2pot::nep::RQ_Chebyshev<double> *p_RadialBasis_2;
    p_RadialBasis_1 = new ai2pot::nep::RQ_Chebyshev<double>(size, rmax);
    p_RadialBasis_1->build(distance_ij);
    p_RadialBasis_2 = new ai2pot::nep::RQ_Chebyshev<double>(size, rmax);
    p_RadialBasis_2->build(distance_ij+delta);
    
printf("0. Value of RQ_Chebyshev:\n");
for (int ii=0; ii<p_RadialBasis_1->size(); ii++)
    printf("%10lf,", p_RadialBasis_1->vals()[ii]);
printf("\n");

printf("1. Custom code method: deriv of Radial Q wrt. r:\n\t");
for (int ii=0; ii<p_RadialBasis_1->size(); ii++)
    printf("%10lf,", p_RadialBasis_1->ders2r()[ii]);
printf("\n");

printf("2. Finite difference method: deriv of Radial Q wrt. r:\n\t");
for (int ii=0; ii<p_RadialBasis_2->size(); ii++) {
    double tmp_der = (p_RadialBasis_2->vals()[ii] - p_RadialBasis_1->vals()[ii]) / delta;
    printf("%10lf, ", tmp_der);
}
printf("\n");

    delete p_RadialBasis_1;
    delete p_RadialBasis_2;
}   



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}