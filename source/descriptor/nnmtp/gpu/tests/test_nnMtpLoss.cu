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

#include "../include/mtpUtilities.cuh"
#include "../include/nnMtpLoss.cuh"


class NNMtpLossTest : public ::testing::Test
{
protected:
    int batch_size;
    int natoms_pad;

    double *bloss;
    int *binum;
    int *bilist;
    double e_weight;
    double f_weight;
    double v_weight;
    double *betot_ml;
    double *betot_dft;
    double (*bforce_ml)[3];
    double (*bforce_dft)[3];
    double *bvirial_ml;
    double *bvirial_dft;

    static void SetUpTestSuite() {
        std::cout << "NNMtpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NNMtpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        batch_size = 1;
        natoms_pad = 12;
        
        bloss = (double*)malloc(sizeof(double) * batch_size);
        for (int ii=0; ii<batch_size; ii++)
            bloss[ii] = 0.0;
        binum = (int*)malloc(sizeof(double) * batch_size);
        binum[0] = 12;
        bilist = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        for (int ii=0; ii<natoms_pad; ii++)
            bilist[0*natoms_pad + ii] = ii;
        e_weight = 1.0;
        f_weight = 1.0;
        v_weight = 1.0;
        betot_ml = (double*)malloc(sizeof(double) * batch_size);
        betot_ml[0] = 100;
        betot_dft = (double*)malloc(sizeof(double) * batch_size);
        betot_dft[0] = 102;

        bforce_ml = (double (*)[3])malloc(sizeof(double) * batch_size * natoms_pad * 3);
        bforce_dft = (double (*)[3])malloc(sizeof(double) * batch_size * natoms_pad * 3);
        for (int ii=0; ii<natoms_pad; ii++) {
            for (int aa=0; aa<3; aa++) {
                bforce_ml[0*natoms_pad + ii][aa] = 1.00;
                bforce_dft[0*natoms_pad + ii][aa] = 1.02 + ii*0.01;
            }
        }

        bvirial_ml = (double*)malloc(sizeof(double) * batch_size * 9);
        bvirial_dft = (double*)malloc(sizeof(double) * batch_size * 9);
        for (int aa=0; aa<3; aa++) {
            for (int bb=0; bb<3; bb++) {
                bvirial_ml[0*9 + aa*3 + bb] = 1.00;
                bvirial_dft[0*9 + aa*3 + bb] = 1.00 + (aa+bb)*0.01;
            }
        }
    }

    void TearDown() override {
        free(bloss);
        free(binum);
        free(bilist);
        free(betot_ml);
        free(betot_dft);
        free(bforce_ml);
        free(bforce_dft);
        free(bvirial_ml);
        free(bvirial_dft);
    }
};  // class : NNMtpTest


TEST_F(NNMtpLossTest, find_loss) {
    ai2pot::nnmtp::find_loss_launcher<double>(
        bloss,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        e_weight,
        f_weight,
        v_weight,
        betot_ml,
        betot_dft,
        bforce_ml,
        bforce_dft,
        bvirial_ml,
        bvirial_dft);

printf("1. Loss = %.15f\n", bloss[0]);
}


TEST_F(NNMtpLossTest, find_ef_loss) {
    ai2pot::nnmtp::find_ef_loss_launcher<double>(
        bloss,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        e_weight,
        f_weight,
        betot_ml,
        betot_dft,
        bforce_ml,
        bforce_dft);
printf("1. Loss = %.15f\n", bloss[0]);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}