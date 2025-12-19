#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <string.h>

#include "../include/nep_utilities.cuh"


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


class TanhActivateFuncTest : public ::testing::Test
{
protected:
    double val1;
    double val2;
    double der1;
    double der2;
    double der2der;
    double hidden_val;
    double delta;

    static void SetUpTestSuite() {
        std::cout << "TanhActivateFuncTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "TanhActivateFuncTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        val1 = 0.0;
        val2 = 0.0;
        der1 = 0.0;
        der2 = 0.0;
        der2der = 0.0;
        hidden_val = 0.1;
        delta = 1e-5;
    }

    void TearDown() override {

    }
};  // class : TanhActivateFuncTest





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


TEST_F(TanhActivateFuncTest, der_accuracy) {
    ai2pot::nep::TanhActivationFunc<double>::find_val(val1, hidden_val);
    ai2pot::nep::TanhActivationFunc<double>::find_val(val2, hidden_val+delta);
    ai2pot::nep::TanhActivationFunc<double>::find_der(der1, hidden_val);

printf("1. Tanh derivative calculated by custom code = %.15lf\n", der1);
printf("2. Tanh detivative calculated by fdm = %.15lf\n", (val2 - val1) / delta);
}


TEST_F(TanhActivateFuncTest, der2der_accuracy) {
    ai2pot::nep::TanhActivationFunc<double>::find_der(der1, hidden_val);
    ai2pot::nep::TanhActivationFunc<double>::find_der(der2, hidden_val+delta);
    ai2pot::nep::TanhActivationFunc<double>::find_der2der(der2der, hidden_val);

printf("1. Tanh der2der calculated by custom code = %.15lf\n", der2der);
printf("2. Tanh der2der calculated by fdm = %.15lf\n", (der2 - der1) / delta);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
