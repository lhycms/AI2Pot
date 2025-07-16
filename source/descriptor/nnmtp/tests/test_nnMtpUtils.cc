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
#include <stdio.h>
#include <stdlib.h>

#include "../include/nnMtpUtils.h"



typedef double real;

class NNMtpUtilsTest : public ::testing::Test
{
protected:
    real hidden_val;
    real activated_hidden_val;
    real activated_hidden_val_;
    real activated_hidden_der;
    real activated_hidden_der_;
    real activated_hidden_der2der;
    
    static void SetUpTestSuite() {
        printf("NNMtpUtilsTest (TestSuite) is setting up...\n");
    }

    static void TearDownTestSuite() {
        printf("NNMtpUtilsTest (TestSuite) is tearing down...\n");
    }

    void SetUp() override {
        activated_hidden_val = 0;
        activated_hidden_val_ = 0;
        activated_hidden_der = 0;
        activated_hidden_der_ = 0;
        activated_hidden_der2der = 0;
        hidden_val = 0.1;
    }

    void TearDown() override {

    }
};  // class : NNMtpUtilsTest


TEST_F(NNMtpUtilsTest, der_accuracy) {
    real delta = 1e-5;
    ai2pot::nnmtp::TanhActivationFunc<real>::find_val(activated_hidden_val, hidden_val);
    ai2pot::nnmtp::TanhActivationFunc<real>::find_der(activated_hidden_der, hidden_val);
    ai2pot::nnmtp::TanhActivationFunc<real>::find_val(activated_hidden_val_, hidden_val+delta);

    printf("Tanh der accuracy:\n");
    printf("\t1. Derivative calculated by custom code = %.10lf\n", (activated_hidden_val_ - activated_hidden_val)/delta);
    printf("\t2. Derivative calculated by fdm = %.10lf\n", activated_hidden_der);
}


TEST_F(NNMtpUtilsTest, der2der_accuracy) {
    real delta = 1e-5;
    ai2pot::nnmtp::TanhActivationFunc<real>::find_der(activated_hidden_der, hidden_val);
    ai2pot::nnmtp::TanhActivationFunc<real>::find_der2der(activated_hidden_der2der, hidden_val);
    ai2pot::nnmtp::TanhActivationFunc<real>::find_der(activated_hidden_der_, hidden_val+delta);

    printf("Tanh der2der accuracy:\n");
    printf("\t1. Derivative^2 calculated by custom code = %.10lf\n", (activated_hidden_der_ - activated_hidden_der)/delta);
    printf("\t2. Derivative^2 calculated by fdm = %.10lf\n", activated_hidden_der2der);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}