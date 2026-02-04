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

#include "../include/nepLoss_cpu_launcher.h"


class NepLossTest : public ::testing::Test
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

    double force_scaler;


    static void SetUpTestSuite() {
        std::cout << "NepLossTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NepLossTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        batch_size = 1;
        natoms_pad = 12;

        bloss = (double*)malloc(sizeof(double) * batch_size);
        memset(bloss, 0, sizeof(double) * batch_size);
        binum = (int*)malloc(sizeof(int) * batch_size);
        binum[0] = 12;
        bilist = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        for (int ii=0; ii<natoms_pad; ii++)
            bilist[ii] = ii;
        e_weight = 1.0;
        f_weight = 1.0;
        v_weight = 1.0;
        betot_ml = (double*)malloc(sizeof(double) * batch_size);
        betot_dft = (double*)malloc(sizeof(double) * batch_size);
        betot_ml[0] = 100;
        betot_dft[0] = 102;

        bforce_ml = (double (*)[3])malloc(sizeof(double) * batch_size * natoms_pad * 3);
        bforce_dft = (double (*)[3])malloc(sizeof(double) * batch_size * natoms_pad * 3);
        for (int ii=0; ii<natoms_pad; ii++) {
            for (int aa=0; aa<3; aa++) {
                bforce_ml[ii][aa] = 1.00;
                bforce_dft[ii][aa] = 1.02 + ii*0.01;
            }
        }

        bvirial_ml = (double*)malloc(sizeof(double) * 9);
        bvirial_dft = (double*)malloc(sizeof(double) * 9);
        for (int aa=0; aa<3; aa++) {
            for (int bb=0; bb<3; bb++) {
                bvirial_ml[aa*3 + bb] = 1.00;
                bvirial_dft[aa*3 + bb] = 1.00 + (aa+bb)*0.01;
            }
        }

        force_scaler = 8.0;
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
};  // class : NepLossTest


TEST_F(NepLossTest, find_ef_loss) {
    ai2pot::nep::find_ef_loss_cpu_launcher(
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

printf("1. EF Loss = %.15f\n", bloss[0]);
}


TEST_F(NepLossTest, rescale_f_cpu_launcher) {
    ai2pot::nep::rescale_f_cpu_launcher(
        bforce_ml,
        bforce_dft,
        batch_size,
        natoms_pad,
        binum,
        force_scaler);

printf("1.1. force_ml after rescale:\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t[%.10f, %.10f, %.10f]\n", bforce_ml[ii][0], bforce_ml[ii][1], bforce_ml[ii][2]);
printf("1.2. force_dft after rescale:\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t[%.10f, %.10f, %.10f]\n", bforce_dft[ii][0], bforce_dft[ii][1], bforce_dft[ii][2]);
    
}


TEST_F(NepLossTest, rescale_fv_cpu_launcher) {
    ai2pot::nep::rescale_fv_cpu_launcher(
        bforce_ml,
        bforce_dft,
        bvirial_ml,
        bvirial_dft,
        batch_size,
        natoms_pad,
        binum,
        force_scaler);

printf("1.1. force_ml after rescale:\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t[%.10f, %.10f, %.10f]\n", bforce_ml[ii][0], bforce_ml[ii][1], bforce_ml[ii][2]);
printf("1.2. force_ml after rescale:\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t[%.10f, %.10f, %.10f]\n", bforce_dft[ii][0], bforce_dft[ii][1], bforce_dft[ii][2]);
printf("2.1. virial_ml after rescale:\n");
for (int a=0; a<3; a++)
    printf("\t[%.10f, %.10f, %.10f]\n", bvirial_ml[a*3 + 0], bvirial_ml[a*3 + 1], bvirial_ml[a*3 + 2]);
printf("2.2. virial_dft after rescale:\n");
for (int a=0; a<3; a++)
    printf("\t[%.10f, %.10f, %.10f]\n", bvirial_dft[a*3 + 0], bvirial_dft[a*3 + 1], bvirial_dft[a*3 + 2]);
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}