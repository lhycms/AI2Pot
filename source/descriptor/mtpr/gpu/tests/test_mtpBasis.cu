#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <string>

#include "../include/mtp_utilities.cuh"
#include "../include/mtpBasis.cuh"
#include "../../include/mtpParam.h"
#include "../../../../nblist/include/structure.h"
#include "../../../../nblist/include/neighborList.h"


class BasisUtilitesTest : public ::testing::Test 
{
protected:
    double rmax;
    double rmin;
    double distance_ij;

    double h_switch_func_val;
    double h_switch_func_der2r;

    int chebyshev_size;
    double *h_rb_chebyshev_vals;
    double *h_rb_chebyshev_ders2r;

    double *h_rq_chebyshev_vals;
    double *h_rq_chebyshev_ders2r;

    static void SetUpTestSuite() {
        std::cout << "BasisUtilitiesTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "BasisUtilitiesTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rmax = 5.0;
        rmin = 2.0;
        distance_ij = 3.14;

        chebyshev_size = 8;
        h_rb_chebyshev_vals = (double*)malloc(chebyshev_size * sizeof(double));
        h_rb_chebyshev_ders2r = (double*)malloc(chebyshev_size * sizeof(double));

        h_rq_chebyshev_vals = (double*)malloc(chebyshev_size * sizeof(double));
        h_rq_chebyshev_ders2r = (double*)malloc(chebyshev_size * sizeof(double));
    }

    void TearDown() override {
        free(h_rb_chebyshev_vals);
        free(h_rb_chebyshev_ders2r);

        free(h_rq_chebyshev_vals);
        free(h_rq_chebyshev_ders2r);
    }
};  // class : BasisUtilitiesTest


class MtpBasisTest : public ::testing::Test
{
protected:
    double *h_mtp_basis_val;
    double *d_mtp_basis_val;
    double (*h_mtp_basis_der)[3];
    double (*d_mtp_basis_der)[3];
    double *h_mtp_basis_der2coeffs;
    double *d_mtp_basis_der2coeffs;
    double *h_mtp_basis_val_;
    double *d_mtp_basis_val_;
    double (*h_mtp_basis_der_)[3];
    double (*d_mtp_basis_der_)[3];
    double *h_mtp_basis_der2coeffs_;
    double *d_mtp_basis_der2coeffs_;
    int chebyshev_size;
    double *h_coeffs;
    double *d_coeffs;
    int nmus;
    double rmax;
    double rmin;
    int ntypes;

    std::vector<std::string> filenames;
    ai2pot::mtpr::MtpParam mtp_param;
    
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];

    int umax_num_neigh_atoms;
    int inum;
    int *h_ilist;
    int *d_ilist;
    int *h_numneigh;
    int *d_numneigh;
    int *h_firstneigh;
    int *d_firstneigh;
    double *h_rcs;
    double *d_rcs;
    int *h_types;
    int *d_types;
    int nghost;

    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> neighbor_list;

    static void SetUpTestSuite() {
        std::cout << "MtpBasisTest (TestSuite) is setting up...\n";

    }

    static void TearDownTestSuite() {
        std::cout << "MtpBasisTest (TestSuite) is tearing down...\n";
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
        mtp_param._load(filenames[4]); // mtp_level = 10
mtp_param.show();

    }

    void TearDown() override {

    }
};  // class : MtpBasisTest


TEST_F(BasisUtilitesTest, switch_func_test) {
    distance_ij = 3.14;
    ai2pot::mtpr::find_switch_func<double>(h_switch_func_val,
                                           h_switch_func_der2r,
                                           rmax,
                                           rmin, 
                                           distance_ij);
printf("BasisUtilitiesTest.switch_func_test:\n");
printf("\t1. h_switch_func_val = %g\n", h_switch_func_val);
printf("\t2. h_switch_func_der2r = %g\n", h_switch_func_der2r);
}


TEST_F(BasisUtilitesTest, rb_chebyshev_test) {
    distance_ij = 3.14;
    ai2pot::mtpr::find_rb_chebyshev<double>(h_rb_chebyshev_vals,
                                            h_rb_chebyshev_ders2r,
                                            chebyshev_size,
                                            rmax,
                                            rmin,
                                            distance_ij);
printf("BasisUtilitiesTest.rb_chebyshev_test:\n");
printf("\t1. rb_chebyshev_vals = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rb_chebyshev_vals[ii]);
printf("\n");
printf("\t1. rb_chebyshev_der2r = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rb_chebyshev_ders2r[ii]);
printf("\n");
}


TEST_F(BasisUtilitesTest, rq_chebyshev_test) {
    distance_ij = 3.14;
    ai2pot::mtpr::find_rq_chebyshev<double>(h_rq_chebyshev_vals,
                                            h_rq_chebyshev_ders2r,
                                            chebyshev_size,
                                            rmax,
                                            rmin,
                                            distance_ij);
printf("BasisUtilitiesTest.rq_chebyshev_test:\n");
printf("\t1. rq_chebyshev_vals = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rq_chebyshev_vals[ii]);
printf("\n");
printf("\t1. rq_chebyshev_der2r = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rq_chebyshev_ders2r[ii]);
printf("\n");
}


TEST_F(MtpBasisTest, find_mtp_basis_val_der) {
    
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
