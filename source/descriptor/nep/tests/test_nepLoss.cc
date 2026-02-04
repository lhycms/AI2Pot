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
#include <iostream>
#include <stdlib.h>

#include "../include/nepLoss.h"


class NepLossTest : public ::testing::Test
{
protected:
    double loss;
    int inum;
    int *ilist;
    double e_weight;
    double f_weight;
    double v_weight;
    double etot_ml;
    double etot_dft;
    double (*force_ml)[3];
    double (*force_dft)[3];
    double *virial_ml;
    double *virial_dft;

    double force_scaler;


    static void SetUpTestSuite() {
        std::cout << "NepLossTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NepLossTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        loss = 0.0;
        inum = 12;
        ilist = (int*)malloc(sizeof(int) * inum);
        for (int ii=0; ii<inum; ii++)
            ilist[ii] = ii;
        e_weight = 1.0;
        f_weight = 1.0;
        v_weight = 1.0;
        etot_ml = 100;
        etot_dft = 102;

        force_ml = (double (*)[3])malloc(sizeof(double) * inum * 3);
        force_dft = (double (*)[3])malloc(sizeof(double) * inum * 3);
        for (int ii=0; ii<inum; ii++) {
            for (int aa=0; aa<3; aa++) {
                force_ml[ii][aa] = 1.00;
                force_dft[ii][aa] = 1.02 + ii*0.01;
            }
        }

        virial_ml = (double*)malloc(sizeof(double) * 9);
        virial_dft = (double*)malloc(sizeof(double) * 9);
        for (int aa=0; aa<3; aa++) {
            for (int bb=0; bb<3; bb++) {
                virial_ml[aa*3 + bb] = 1.00;
                virial_dft[aa*3 + bb] = 1.00 + (aa+bb)*0.01;
            }
        }

        force_scaler = 8;
    }

    void TearDown() override {
        free(ilist);
        free(force_ml);
        free(force_dft);
        free(virial_ml);
        free(virial_dft);
    }
};  // class : NepLossTest


TEST_F(NepLossTest, find_ef_loss) {
    ai2pot::nep::NepLoss<double>::find_ef_loss(loss,
                                            inum,
                                            ilist,
                                            e_weight,
                                            f_weight,
                                            etot_ml,
                                            etot_dft,
                                            force_ml,
                                            force_dft);
printf("1. EF Loss = %.15f\n", loss);
}



TEST_F(NepLossTest, find_loss) {
    ai2pot::nep::NepLoss<double>::find_loss(loss,
                                            inum,
                                            ilist,
                                            e_weight,
                                            f_weight,
                                            v_weight,
                                            etot_ml,
                                            etot_dft,
                                            force_ml,
                                            force_dft,
                                            virial_ml,
                                            virial_dft);
printf("1. Loss = %.15f\n", loss);
}


TEST_F(NepLossTest, rescale_f) {
    int center_idx = 1;
    int direction_idx = 2;
printf("1, Rescale force:\n");
printf("\t1.1. Force before rescale = %.10f\n", force_ml[center_idx][direction_idx]);

    ai2pot::nep::NepLoss<double>::rescale_f(
        force_ml,
        inum,
        force_scaler);

printf("\t1.2. Force after rescale = %.10f\n", force_ml[center_idx][direction_idx]);
}


TEST_F(NepLossTest, rescale_fv) {
    int center_idx = 1;
    int direction_idx1 = 2;
    int direction_idx2 = 0;
printf("1, Rescale force:\n");
printf("\t1.1. Force before rescale = %.10f\n", force_ml[center_idx][direction_idx1]);
printf("\t1.1. Virial before rescale = %.10f\n", virial_ml[direction_idx1*3 + direction_idx2]);

    ai2pot::nep::NepLoss<double>::rescale_fv(
        force_ml,
        virial_ml,
        inum,
        force_scaler);

printf("\t1.2. Force before rescale = %.10f\n", force_ml[center_idx][direction_idx1]);
printf("\t1.2. Virial before rescale = %.10f\n", virial_ml[direction_idx1*3 + direction_idx2]);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}