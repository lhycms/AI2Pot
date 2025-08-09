#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#include "../include/mtp_utilities.cuh"
#include "../include/linearMtpLoss.cuh"


class LinearMtpLossTest : public ::testing::Test
{
protected:
    double *bloss;
    int batch_size;
    int natoms_pad;
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
        std::cout << "LinearMtpLossTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "LinearMtpLossTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        bloss = (double*)malloc(sizeof(double) * batch_size);
        batch_size = 1;
        natoms_pad = 12;
        binum = (int*)malloc(sizeof(int) * batch_size);
        for (int bb=0; bb<batch_size; bb++)
            binum[bb] = 12;
        bilist = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        for (int bb=0; bb<batch_size; bb++)
            for (int ii=0; ii<natoms_pad; ii++)
                bilist[ii] = ii;
        e_weight = 1.0;
        f_weight = 1.0;
        v_weight = 1.0;
        betot_ml = (double*)malloc(sizeof(double) * batch_size);
        betot_ml[0] = 100;
        betot_dft = (double*)malloc(sizeof(double) * batch_size);
        betot_dft[0] = 102;

        bforce_ml = (double (*)[3])malloc(sizeof(double) * batch_size * natoms_pad * 3);
        bforce_dft = (double (*)[3])malloc(sizeof(double) * batch_size * natoms_pad * 3);
        for (int bb=0; bb<batch_size; bb++) {
            for (int ii=0; ii<natoms_pad; ii++) {
                for (int aa=0; aa<3; aa++) {
                    bforce_ml[bb*natoms_pad + ii][aa] = 1.00;
                    bforce_dft[bb*natoms_pad + ii][aa] = 1.02 + ii*0.01;
                }
            }
        }

        bvirial_ml = (double*)malloc(sizeof(double) * batch_size * 9);
        bvirial_dft = (double*)malloc(sizeof(double) * batch_size * 9);
        for (int b=0; b<batch_size; b++) {
            for (int aa=0; aa<3; aa++) {
                for (int bb=0; bb<3; bb++) {
                    bvirial_ml[b*9 + aa*3 + bb] = 1.00;
                    bvirial_dft[b*9 + aa*3 + bb] = 1.00 + (aa+bb)*0.01;
                }
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
};  // class : LinearMtpLossTest



TEST_F(LinearMtpLossTest, find_loss) {
    ai2pot::mtpr::find_loss_launcher<double>(
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



TEST_F(LinearMtpLossTest, find_ef_loss) {
    ai2pot::mtpr::find_ef_loss_launcher<double>(
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