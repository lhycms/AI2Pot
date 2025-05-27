#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>

#include "../include/mtp_utilities.cuh"
#include "../include/linearMtp.cuh"
#include "../../include/mtpParam.h"
#include "../../../../nblist/include/structure.h"
#include "../../../../nblist/include/neighborList.h"


class LinearMtpTest : public ::testing::Test
{
protected:
    std::vector<std::string> filenames;
    ai2pot::mtpr::MtpParam mtp_param;
    int ntypes;
    double zbl_rmax;
    double zbl_rmin;
    double *zbl_cks;
    double *zbl_dks;



    static void SetUpTestSuite() {
        std::cout << "LinearMtpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "LinearMtpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        filenames = {
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/depreciated-02.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/depreciated-04.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/06.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/08.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/10.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/12.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/14.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/16.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/18.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/20.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/22.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/24.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/26.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/28.almtp"
        };
        mtp_param._load(filenames[4]);

        ntypes = 2;
        zbl_rmin = 1.0;
        zbl_rmax = 2.0;
        zbl_cks = (double*)malloc(sizeof(double) * ntypes * ntypes * 4);
        zbl_dks = (double*)malloc(sizeof(double) * ntypes * ntypes * 4);
        for (int ii=0; ii<ntypes; ii++) {
            for (int jj=0; jj<ntypes; jj++) {
                zbl_cks[(ii*ntypes + jj)*4 + 0] = 0.18175;
                zbl_cks[(ii*ntypes + jj)*4 + 1] = 0.50986;
                zbl_cks[(ii*ntypes + jj)*4 + 2] = 0.28022;
                zbl_cks[(ii*ntypes + jj)*4 + 3] = 0.02817;
                zbl_dks[(ii*ntypes + jj)*4 + 0] = 3.1998;
                zbl_dks[(ii*ntypes + jj)*4 + 1] = 0.94229;
                zbl_dks[(ii*ntypes + jj)*4 + 2] = 0.4029;
                zbl_dks[(ii*ntypes + jj)*4 + 3] = 0.20162;
            }
        }
    }

    void TearDown() override {

    }
};  // class : LinearMtpTest


TEST_F(LinearMtpTest, find_efv_launcher)
{
    
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}