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

#include "../include/mtpUtilities.cuh"


class NNMtpUtilsTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite() {
        std::cout << "NNMtpUtilsTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NNMtpUtilsTest (TestSuite) is tearing down...\n";
    }
};  // class : NNMtpUtilsTest


TEST_F(NNMtpUtilsTest, mtp_hyperparameters) {
    printf("mtp_hyperparameters:\n");
    printf("1. MAX_NUM_MUS = %d\n", MAX_NUM_MUS);
    printf("2. MAX_CHEBYSHEV_SIZE = %d\n", MAX_CHEBYSHEV_SIZE);
    printf("3. MAX_NUM_MTP_BASIS = %d\n", MAX_NUM_MTP_BASIS);
    printf("4. MAX_ALPHA_INDEX_BASIC = %d\n", MAX_ALPHA_INDEX_BASIC);
    printf("5. MAX_ALPHA_MOMENTS_COUNT = %d\n", MAX_ALPHA_MOMENTS_COUNT);
}

TEST_F(NNMtpUtilsTest, tanh_der_accuracy) {
    double val1 = 0.0;
    double val2 = 0.0;
    double der = 0.0;
    double hidden_val = 0.1;
    double delta = 1e-5;

    TanhActivationFunc<double>::find_val(val1, hidden_val);
    TanhActivationFunc<double>::find_val(val2, hidden_val + delta);
    TanhActivationFunc<double>::find_der(der, hidden_val);

printf("1. Tanh derivative calculated by custom code = %.15lf\n", der);
printf("2. Tanh detivative calculated by fdm = %.15lf\n", (val2 - val1) / delta);
}


TEST_F(NNMtpUtilsTest, tanh_der2der_accuracy) {
    double der1 = 0.0;
    double der2 = 0.0;
    double der2der = 0.0;
    double hidden_val = 0.1;
    double delta = 1e-5;

    TanhActivationFunc<double>::find_der(der1, hidden_val);
    TanhActivationFunc<double>::find_der(der2, hidden_val + delta);
    TanhActivationFunc<double>::find_der2der(der2der, hidden_val);

printf("1. Tanh der2der calculated by custom code = %.15lf\n", der2der);
printf("2. Tanh der2der calculated by fdm = %.15lf\n", (der2 - der1) / delta);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
