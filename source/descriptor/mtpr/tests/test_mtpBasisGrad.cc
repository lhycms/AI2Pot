#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include "../include/mtpBasis.h"
#include "../include/mtpBasisGrad.h"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"


class MtpBasisGradTest : public ::testing::Test
{
protected:
    bool calculate_mtp_basis;

    double *mtp_basis_val;
    double (*mbg_val)[3];
    double *mbg_der2coeffs;
    double *mtp_basis_val_;
    double (*mbg_val_)[3];
    double *mbg_der2coeffs_;
    int chebyshev_size;
    double *coeffs;
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
    int *ilist;
    int *numneigh;
    int *firstneigh;
    double *rcs;
    int *types;
    int nghost;

    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> neighbor_list;

    static void SetUpTestSuite() {
        std::cout << "MtpBasisGradTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpBasisGradTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        calculate_mtp_basis = true;

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
        mtp_param._load(filenames[7]);

        inum = 12;
        ntypes = 2;
        chebyshev_size = 8;
        nmus = mtp_param.nmus();
        rmax = 5.0;
        rmin = 2.0;
        umax_num_neigh_atoms = 20;

        mtp_basis_val = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments());
        mbg_val = (double (*)[3])malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments() * umax_num_neigh_atoms * 3);
        mbg_der2coeffs = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments() * umax_num_neigh_atoms * 3 * ntypes * ntypes * nmus * chebyshev_size);
        mtp_basis_val_ = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments());
        mbg_val_ = (double (*)[3])malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments() * umax_num_neigh_atoms * 3);
        mbg_der2coeffs_ = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments() * umax_num_neigh_atoms * 3 * ntypes * ntypes * nmus * chebyshev_size);
        coeffs = (double*)malloc(sizeof(double) * ntypes * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        for (int ii=0; ii<ntypes*ntypes*nmus*chebyshev_size; ii++)
            coeffs[ii] = 1.0;

        // Establish neighbor list
        num_atoms = 12;
        basis_vectors[0][0] = 3.1903157348;
        basis_vectors[0][1] = 5.5257885468;
        basis_vectors[0][2] = 0.0000000000;
        basis_vectors[1][0] = -6.3806307800;
        basis_vectors[1][1] = 0.0000000000;
        basis_vectors[1][2] = 0.0000000000;
        basis_vectors[2][0] = 0.0000000000;
        basis_vectors[2][1] = 0.0000000000;
        basis_vectors[2][2] = 23.1297687334;

        // 42: 0;  16: 1
        atomic_numbers[0] = 0;
        atomic_numbers[1] = 1;
        atomic_numbers[2] = 1;
        atomic_numbers[3] = 0;
        atomic_numbers[4] = 1;
        atomic_numbers[5] = 1;
        atomic_numbers[6] = 0;
        atomic_numbers[7] = 1;
        atomic_numbers[8] = 1;
        atomic_numbers[9] = 0; 
        atomic_numbers[10] = 1;
        atomic_numbers[11] = 1;

        frac_coords[0][0] = 0.333333333333;
        frac_coords[0][1] = 0.166666666667;
        frac_coords[0][2] = 0.500000000000;
        frac_coords[1][0] = 0.166666666667;
        frac_coords[1][1] = 0.333333333333;
        frac_coords[1][2] = 0.432343276548;
        frac_coords[2][0] = 0.166666666667;
        frac_coords[2][1] = 0.333333333333;
        frac_coords[2][2] = 0.567656723452;
        frac_coords[3][0] = 0.333333333333;
        frac_coords[3][1] = 0.666666666667;
        frac_coords[3][2] = 0.500000000000;
        frac_coords[4][0] = 0.166666666667;
        frac_coords[4][1] = 0.833333333333;
        frac_coords[4][2] = 0.432343276548;
        frac_coords[5][0] = 0.166666666667;
        frac_coords[5][1] = 0.833333333333;
        frac_coords[5][2] = 0.567656723452;
        frac_coords[6][0] = 0.833333333333;
        frac_coords[6][1] = 0.166666666667;
        frac_coords[6][2] = 0.500000000000;
        frac_coords[7][0] = 0.666666666667;
        frac_coords[7][1] = 0.333333333333;
        frac_coords[7][2] = 0.432343276548;
        frac_coords[8][0] = 0.666666666667;
        frac_coords[8][1] = 0.333333333333;
        frac_coords[8][2] = 0.567656723452;
        frac_coords[9][0] = 0.833333333333;
        frac_coords[9][1] = 0.666666666667;
        frac_coords[9][2] = 0.500000000000;
        frac_coords[10][0] = 0.666666666667;
        frac_coords[10][1] = 0.833333333333;
        frac_coords[10][2] = 0.432343276548;
        frac_coords[11][0] = 0.666666666667;
        frac_coords[11][1] = 0.833333333333;
        frac_coords[11][2] = 0.567656723452;

        rcut = 5.0;
        bin_size_xyz[0] = 2.5;
        bin_size_xyz[1] = 2.5;
        bin_size_xyz[2] = 2.5;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;

        structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = ai2pot::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, true);

        umax_num_neigh_atoms = 19;
        inum = 12;
        ilist = (int*)malloc(sizeof(int) * inum);
        numneigh = (int*)malloc(sizeof(int) * inum);
        firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        rcs = (double*)malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
        types = (int*)malloc(sizeof(int) * inum);
        neighbor_list.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);
    }

    void TearDown() override {
        free(mtp_basis_val);
        free(mtp_basis_val_);
        free(mbg_val);
        free(mbg_val_);
        free(mbg_der2coeffs);
        free(mbg_der2coeffs_);
        free(coeffs);

        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);
    }
};  // class : MtpBasisGradTest


TEST_F(MtpBasisGradTest, mbg_der2coeff_accuracy)
{
    int center_idx_modify = 0;
    int neigh_idx_modify = 12;
    int direction_idx_modify = 2;

    int itype_idx_modify = types[ilist[center_idx_modify]];
    int jtype_idx_modify = types[firstneigh[center_idx_modify*umax_num_neigh_atoms + neigh_idx_modify]];
    int mu_idx_modify = 0;
    int cheby_idx_modify = 0;
    int coeff_idx_modify = (itype_idx_modify*ntypes + jtype_idx_modify)*nmus*chebyshev_size + mu_idx_modify*chebyshev_size + cheby_idx_modify;
    int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;
    
    double delta = 1e-5;

    ai2pot::mtpr::MtpBasisGrad<double>::find_val_der(
        mtp_basis_val,
        mbg_val,
        mbg_der2coeffs,
        calculate_mtp_basis,
        chebyshev_size,
        coeffs,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.max_num_mus4mom(),
        mtp_param.num_mus4moms(),
        mtp_param.mus4moms_ptr(),
        mtp_param.nmus(),
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        umax_num_neigh_atoms,
        rmax,
        rmin);
    coeffs[coeff_idx_modify] += delta;
    ai2pot::mtpr::MtpBasisGrad<double>::find_val_der(
        mtp_basis_val_,
        mbg_val_,
        mbg_der2coeffs_,
        calculate_mtp_basis,
        chebyshev_size,
        coeffs,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.max_num_mus4mom(),
        mtp_param.num_mus4moms(),
        mtp_param.mus4moms_ptr(),
        mtp_param.nmus(),
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        umax_num_neigh_atoms,
        rmax,
        rmin);    


printf("0. Value of mtp basis val:\n\t[");
for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
    printf("%.*lf, ", 15, mtp_basis_val[center_idx_modify*mtp_param.alpha_scalar_moments() + ii]);
printf("\n\n");


printf("1. Value of mbg (mtp basis gradient w.r.t. rcs):\n\t");
for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
    printf("%10lf, ", 
        mbg_val[center_idx_modify*mtp_param.alpha_scalar_moments()*umax_num_neigh_atoms + ii*umax_num_neigh_atoms + neigh_idx_modify][direction_idx_modify]);
printf("]\n\n");

printf("2. Derivative w.r.t coeffs of mbg (mtp basis gradient w.r.t rcs):\n");
printf("2.1. Deriv of mbg w.r.t coeffs calculated with custom code:\n\t[");
for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++) {
    int idx = center_idx_modify*mtp_param.alpha_scalar_moments()*umax_num_neigh_atoms*3*num_coeffs
              + ii*umax_num_neigh_atoms*3*num_coeffs
              + neigh_idx_modify*3*num_coeffs
              + direction_idx_modify*num_coeffs
              + (itype_idx_modify*ntypes + jtype_idx_modify)*nmus*chebyshev_size + mu_idx_modify*chebyshev_size + cheby_idx_modify;
    printf("%10lf, ", mbg_der2coeffs[idx]);
}
printf("]\n");

printf("2.2. Deriv of mbg w.r.t coeffs calculated with custom code:\n\t[");
for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++) {
    int idx = center_idx_modify*mtp_param.alpha_scalar_moments()*umax_num_neigh_atoms*3*num_coeffs
              + ii*umax_num_neigh_atoms*3*num_coeffs
              + neigh_idx_modify*3*num_coeffs
              + direction_idx_modify*num_coeffs
              + (itype_idx_modify*ntypes + jtype_idx_modify)*nmus*chebyshev_size + mu_idx_modify*chebyshev_size + cheby_idx_modify;
    printf("%10lf, ", mbg_der2coeffs_[idx]);
}
printf("]\n");

printf("2.3. Deriv of mbg w.r.t rcs calculated with finite difference method:\n\t[");
for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++) {
    int idx = center_idx_modify*mtp_param.alpha_scalar_moments()*umax_num_neigh_atoms
              + ii*umax_num_neigh_atoms
              + neigh_idx_modify;
    double der_fdm = (mbg_val_[idx][direction_idx_modify] - mbg_val[idx][direction_idx_modify]) / delta;
    printf("%10lf, ", der_fdm);
}
printf("]\n");

}



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
