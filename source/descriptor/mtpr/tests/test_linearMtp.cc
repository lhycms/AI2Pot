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
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <cassert>
#include <random>

#include "../include/linearMtp.h"
#include "../include/mtpBasis.h"
#include "../include/mtpParam.h"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"


class LinearMtpTest : public ::testing::Test
{
protected:    
    double *mom_vals;
    double (*mom_ders)[3];

    double loss;
    double loss_;
    double e_weight;
    double f_weight;
    double v_weight;

    double etot;
    double (*force)[3];
    double *virial;
    double etot_;
    double (*force_)[3];
    double *virial_;
    double etot_dft;
    double (*force_dft)[3];
    double *virial_dft;

    int chebyshev_size;
    double *coeffs;
    int nmus;
    double rmax;
    double rmin;
    int ntypes;

    double zbl_rmax;
    double zbl_rmin;
    double *zbl_cks;
    double *zbl_dks;
    
    std::vector<std::string> filenames;
    ai2pot::mtpr::MtpParam mtp_param;

    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];
    int type_map[2];

    int umax_num_neigh_atoms;
    int inum;
    int* ilist;
    int* numneigh;
    int* firstneigh;
    double* rcs;
    int* types;
    int nghost;

    double *linear_coeffs;
    double *type_bias;

    double *loss_der2coeffs;
    double *loss_der2linear_coeffs;
    double *loss_der2type_bias;

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
        mtp_param._load(filenames[4]);
//mtp_param.show();

        e_weight = 1.0;
        f_weight = 1.0;
        v_weight = 0.1;

        inum = 12;
        ntypes = 2;
        chebyshev_size = 8;
        nmus = mtp_param.nmus();
        rmax = 5.0;
        rmin = 2.0;
        umax_num_neigh_atoms = 20;
        coeffs = (double*)malloc(sizeof(double) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        
        std::random_device rd;  // 用于生成随机种子
        std::mt19937 gen(rd()); // 随机数生成器，使用 Mersenne Twister 算法
        std::normal_distribution<> dis(0.0, 0.3); // 高斯分布，均值 0，标准差 0.3
        //for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
        //    coeffs[ii] = dis(gen);
        for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
            coeffs[ii] = 0.1 + 0.01 * ii;

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
        type_map[0] = 42;
        type_map[1] = 16;
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

        mom_vals = (double*)malloc(sizeof(double) * mtp_param.alpha_moments_count());
        mom_ders = (double (*)[3])malloc(sizeof(double) * mtp_param.alpha_index_basic_count() * umax_num_neigh_atoms * 3);

        etot = 0;
        force = (double (*)[3])malloc(sizeof(double) * (nghost+inum) * 3);
        virial = (double*)malloc(sizeof(double) * 9);
        etot_ = 0;
        force_ = (double (*)[3])malloc(sizeof(double) * (nghost+inum) * 3);
        virial_ = (double*)malloc(sizeof(double) * 9);
        etot_dft = 0;
        force_dft = (double (*)[3])malloc(sizeof(double) * (nghost+inum) * 3);
        memset(force_dft, 0.0, sizeof(double) * (inum+nghost) * 3);
        virial_dft = (double*)malloc(sizeof(double) * 9);
        memset(virial_dft, 0.0, sizeof(double) * 9);
        for (int ii=0; ii<(inum+nghost); ii++)
            for (int aa=0; aa<3; aa++)
                force[ii][aa] = 0.0;
        for (int ab=0; ab<9; ab++)
            virial_dft[ab] = 0.0;

        linear_coeffs = (double*)malloc(sizeof(double) * mtp_param.alpha_scalar_moments());
        type_bias = (double*)malloc(sizeof(double) * ntypes);

        //for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
        //    linear_coeffs[ii] = dis(gen);
        for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
            linear_coeffs[ii] = 0.1 + 0.01 * ii;
        type_bias[0] = -0.1;
        type_bias[1] = -0.2;

        loss_der2coeffs = (double*)malloc(sizeof(double) * ntypes * ntypes * nmus * chebyshev_size);
        loss_der2linear_coeffs = (double*)malloc(sizeof(double) * mtp_param.alpha_scalar_moments());
        loss_der2type_bias = (double*)malloc(sizeof(double) * ntypes);


        zbl_rmax = 2.0;
        zbl_rmin = 1.0;
        zbl_cks = (double*)malloc(sizeof(double) * ntypes * ntypes * 4);
        zbl_dks = (double*)malloc(sizeof(double) * ntypes * ntypes * 4);
        for (int ii=0; ii<ntypes; ii++) {
            for (int jj=0; jj<ntypes; jj++) {
                int zbl_idx = ii*ntypes + jj;
                zbl_cks[zbl_idx * 4 + 0] = 0.18175;
                zbl_cks[zbl_idx * 4 + 1] = 0.50986;
                zbl_cks[zbl_idx * 4 + 2] = 0.28022;
                zbl_cks[zbl_idx * 4 + 3] = 0.02817;
                zbl_dks[zbl_idx * 4 + 0] = 3.1998;
                zbl_dks[zbl_idx * 4 + 1] = 0.94229;
                zbl_dks[zbl_idx * 4 + 2] = 0.4029;
                zbl_dks[zbl_idx * 4 + 3] = 0.20162;
            }
        }
    }

    void TearDown() override {
        free(mom_vals);
        free(mom_ders);

        free(force);
        free(force_);
        free(force_dft);
        free(virial);
        free(virial_);
        free(virial_dft);

        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);
        
        free(coeffs);
        free(linear_coeffs);
        free(type_bias);

        free(loss_der2coeffs);
        free(loss_der2linear_coeffs);
        free(loss_der2type_bias);

        free(zbl_cks);
        free(zbl_dks);
    }
};


/*
TEST_F(LinearMtpTest, find_efv) {
    int center_idx_modify = 0;
    int direction1_idx_modify = 0;
    int direction2_idx_modify = 0;
    double delta = 1E-5;


    ai2pot::mtpr::LinearMtp<double>::find_efv(
        etot,
        force,
        virial,
        chebyshev_size,
        coeffs,
        linear_coeffs,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        nmus,
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);
printf("1. etot = %g\n", etot);
printf("2. force[%d][%d] = %g\n", center_idx_modify, direction1_idx_modify, force[center_idx_modify][direction1_idx_modify]);
printf("3. virial[%d][%d] = %g\n", direction1_idx_modify, direction2_idx_modify, virial[direction1_idx_modify*3 + direction2_idx_modify]);
}
*/



TEST_F(LinearMtpTest, force_accuracy) {
    int center_idx_modify = 0;
    int direction1_idx_modify = 0;
    int direction2_idx_modify = 0;
    double delta = 1E-8;

    ai2pot::mtpr::LinearMtp<double>::find_efv(
        etot,
        force,
        virial,
        chebyshev_size,
        coeffs,
        linear_coeffs,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        nmus,
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks,
        zbl_dks);

    // *** delta
    double cart_coords[inum][3] = {0};
    for (int ii=0; ii<inum; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction1_idx_modify] += delta;
    structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);
    neighbor_list = ai2pot::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, true);
    neighbor_list.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);
    // *** delta

    ai2pot::mtpr::LinearMtp<double>::find_efv(
        etot_,
        force_,
        virial_,
        chebyshev_size,
        coeffs,
        linear_coeffs,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        nmus,
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks,
        zbl_dks);

printf("1.1. energy = %.15lf\n", etot);
printf("1.1. force[%d][%d] calculated by custom code = %.15lf\n", center_idx_modify, direction1_idx_modify, force[center_idx_modify][direction1_idx_modify]);
printf("1.2. energy = %.15lf\n", etot_);
printf("1.2. force[%d][%d] calculated by finite difference method = %.15lf\n", center_idx_modify, direction1_idx_modify, -(etot_ - etot) / delta);

printf("2.1. energy = %.15lf\n", etot);
printf("2.2. force=\n");
for (int ii=0; ii<inum; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, force[ii][0], force[ii][1], force[ii][2]);
}


/*
TEST_F(LinearMtpTest, find_loss) {
    int center_idx_modify = 0;
    int neigh_idx_modify = 17;
    int direction1_idx_modify = 2;
    int direction2_idx_modify = 0;
    double delta = 1E-7;

    ai2pot::mtpr::LinearMtp<double>::find_loss(
        loss,
        e_weight,
        f_weight,
        v_weight,
        etot_dft,
        force_dft,
        virial_dft,
        chebyshev_size,
        coeffs,
        linear_coeffs,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        nmus,
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);
printf("1. loss = %g\n", loss);
}


TEST_F(LinearMtpTest, find_loss_backward) {
    ai2pot::mtpr::LinearMtp<double>::find_loss_backward(
        loss_der2coeffs,
        loss_der2linear_coeffs,
        loss_der2type_bias,
        e_weight,
        f_weight,
        v_weight,
        etot_dft,
        force_dft,
        virial_dft,
        chebyshev_size,
        coeffs,
        linear_coeffs,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
