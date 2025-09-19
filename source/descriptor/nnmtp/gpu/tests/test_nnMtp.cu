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

#include "../../include/mtpParam.h"
#include "../include/nnMtp.cuh"
#include "../include/nnMtpLoss.cuh"
#include "../../../../nblist/include/structure.h"
#include "../../../../nblist/include/neighborList.h"

typedef double real;


class NNMtpTest : public ::testing::Test
{
protected:
    std::vector<std::string> filenames;
    ai2pot::nnmtp::MtpParam mtp_param;
    int chebyshev_size;
    int num_neurons;
    int ntypes;
    int batch_size;
    int natoms_pad;

    real *coeffs;
    real *w0;
    real *w1;
    real *type_bias;
    real rmax;
    real rmin;

    int num_atoms;
    real basis_vectors[3][3];
    int atomic_numbers[12];
    real frac_coords[12][3];
    real rcut;
    real bin_size_xyz[3];
    bool pbc_xyz[3];
    int type_map[2];

    int umax_num_neigh_atoms;
    int *binum;
    int *bilist;
    int *bnumneigh;
    int *bfirstneigh;
    real *brcs;
    int *btypes;
    int nghost;

    ai2pot::Structure<real> structure;
    ai2pot::NeighborList<real> neighbor_list;


    real *betot;
    real *betot_;
    real (*bforce)[3];
    real (*bforce_)[3];
    real *bvirial;
    real *bvirial_;
    real *bdescriptors;


    real *bloss_der2coeffs;
    real *bloss_der2w0;
    real *bloss_der2w1;
    real *bloss_der2type_bias;

    real *betot_dft;
    real (*bforce_dft)[3];
    real *bvirial_dft;


    static void SetUpTestSuite() {
        std::cout << "NNMtpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NNMtpTest (TestSutie) is tearing down...\n";
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
        chebyshev_size = 8;
        num_neurons = 30;
        batch_size = 1;
        natoms_pad = 12;
        nghost = 0;

        ntypes = 2;

        coeffs = (real*)malloc(sizeof(real) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        w0 = (real*)malloc(sizeof(real) * ntypes * num_neurons * mtp_param.alpha_scalar_moments());
        w1 = (real*)malloc(sizeof(real) * ntypes * num_neurons);
        type_bias = (real*)malloc(sizeof(real) * ntypes);
        for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
            coeffs[ii] = 0.01 + 0.001 * ii;
        for (int ii=0; ii<ntypes * num_neurons * mtp_param.alpha_scalar_moments(); ii++)
            w0[ii] = 0.1 + 0.001 * ii;
        for (int ii=0; ii<ntypes * num_neurons; ii++)
            w1[ii] = 0.1 + 0.001 * ii;
        type_bias[0] = -0.1;
        type_bias[1] = -0.2;
        rmax = 5.0;
        rmin = 0.0;

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
        
        type_map[0] = 42;
        type_map[1] = 16;

        structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);
        umax_num_neigh_atoms = 20;
        binum = (int*)malloc(sizeof(int) * batch_size);
        binum[0] = 12;
        bilist = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        bnumneigh = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        bfirstneigh = (int*)malloc(sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms);
        brcs = (real*)malloc(sizeof(real) * batch_size * natoms_pad * umax_num_neigh_atoms * 3);
        btypes = (int*)malloc(sizeof(int) * batch_size * (natoms_pad+nghost));
        for (int bb=0; bb<batch_size; bb++)
            neighbor_list.find_info4mlff(
                binum[bb],
                &bilist[bb*natoms_pad],
                &bnumneigh[bb*natoms_pad],
                &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms],
                &brcs[bb*natoms_pad*umax_num_neigh_atoms*3],
                &btypes[bb*(natoms_pad+nghost)],
                nghost,
                umax_num_neigh_atoms);
        

        betot = (real*)malloc(sizeof(real)*batch_size);
        memset(betot, 0.0, sizeof(real)*batch_size);
        bforce = (real (*)[3])malloc(sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        memset(bforce, 0.0, sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        bvirial = (real*)malloc(sizeof(real)*batch_size*9);
        memset(bvirial, 0.0, sizeof(real)*batch_size*9);

        betot_ = (real*)malloc(sizeof(real)*batch_size);
        memset(betot_, 0.0, sizeof(real)*batch_size);
        bforce_ = (real (*)[3])malloc(sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        memset(bforce_, 0.0, sizeof(real)*batch_size*(natoms_pad+nghost)*3);
        bvirial_ = (real*)malloc(sizeof(real)*batch_size*9);
        memset(bvirial_, 0.0, sizeof(real)*batch_size*9);

        bloss_der2coeffs = (real*)malloc(sizeof(real)*batch_size*ntypes*ntypes*mtp_param.nmus()*chebyshev_size);
        memset(bloss_der2coeffs, 0, sizeof(real)*batch_size*ntypes*ntypes*mtp_param.nmus()*chebyshev_size);
        bloss_der2w0 = (real*)malloc(sizeof(real)*batch_size*ntypes*num_neurons*mtp_param.alpha_scalar_moments());
        memset(bloss_der2w0, 0, sizeof(real)*batch_size*ntypes*num_neurons*mtp_param.alpha_scalar_moments());
        bloss_der2w1 = (real*)malloc(sizeof(real)*batch_size*ntypes*num_neurons);
        memset(bloss_der2w1, 0, sizeof(real)*batch_size*ntypes*num_neurons);
        bloss_der2type_bias = (real*)malloc(sizeof(real)*batch_size*ntypes);
        memset(bloss_der2type_bias, 0, sizeof(real)*batch_size*ntypes);

        betot_dft = (real*)malloc(sizeof(real)*batch_size);
        memset(betot_dft, 0, sizeof(real)*batch_size);
        bforce_dft = (real (*)[3])malloc(sizeof(real)*batch_size*natoms_pad*3);
        memset(bforce_dft, 0, sizeof(real)*batch_size*natoms_pad*3);
        bvirial_dft = (real*)malloc(sizeof(real)*batch_size*9);
        memset(bvirial_dft, 0, sizeof(real)*batch_size*9);

        bdescriptors = (real*)malloc(sizeof(real) * batch_size * natoms_pad * mtp_param.alpha_scalar_moments());
        memset(bdescriptors, 0, sizeof(real) * batch_size * natoms_pad * mtp_param.alpha_scalar_moments());
    }

    void TearDown() override {
        free(binum);
        free(bilist);
        free(bnumneigh);
        free(bfirstneigh);
        free(brcs);
        free(btypes);

        free(coeffs);
        free(w0);
        free(w1);
        free(type_bias);

        free(betot);
        free(betot_);
        free(bforce);
        free(bvirial);
        free(bforce_);
        free(bvirial_);

        free(bloss_der2coeffs);
        free(bloss_der2w0);
        free(bloss_der2w1);
        free(bloss_der2type_bias);
        free(betot_dft);
        free(bforce_dft);
        free(bvirial_dft);
        free(bdescriptors);
    }
};  // class : NNMtpTest


TEST_F(NNMtpTest, find_efv_accuracy) {
    real delta = 1e-6;
    int center_idx_modify = 1;
    int direction_idx_modify = 2;

    ai2pot::nnmtp::find_efv_launcher<real>(
        betot,
        bforce,
        bvirial,
        chebyshev_size,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);

    // *** delta
    real cart_coords[natoms_pad][3] = {0.};
    for (int ii=0; ii<natoms_pad; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction_idx_modify] += delta;
    structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);
    neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);

    for (int bb=0; bb<batch_size; bb++)
        neighbor_list.find_info4mlff(
            binum[bb],
            &bilist[bb*natoms_pad],
            &bnumneigh[bb*natoms_pad],
            &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms],
            &brcs[bb*natoms_pad*umax_num_neigh_atoms*3],
            &btypes[bb*(natoms_pad+nghost)],
            nghost,
            umax_num_neigh_atoms);
    
    ai2pot::nnmtp::find_efv_launcher<real>(
        betot_,
        bforce_,
        bvirial_,
        chebyshev_size,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);


printf("1.1. energy = %.15lf\n", betot[0]);
printf("1.1. bforce[%d][%d][%d] calculated by custom code = %.15lf\n", 0, center_idx_modify, direction_idx_modify, bforce[0*(natoms_pad+nghost) + center_idx_modify][direction_idx_modify]);
printf("1.2. energy = %.15lf\n", betot_[0]);
printf("1.2. bforce[%d][%d][%d] calculated by finite difference method = %.15lf\n", 0, center_idx_modify, direction_idx_modify, -(betot_[0] - betot[0]) / delta);
printf("\n\nfind_efv_launcher:\n");
printf("\t2.1. Energy = %.15f\n", betot[0]);
printf("\t2.2. Force =\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, bforce[0*(natoms_pad+nghost) + ii][0], bforce[0*(natoms_pad+nghost) + ii][1], bforce[0*(natoms_pad+nghost) + ii][2]);
}



TEST_F(NNMtpTest, find_ef_accuracy) {
    real delta = 1e-6;
    int center_idx_modify = 1;
    int direction_idx_modify = 2;

    ai2pot::nnmtp::find_ef_launcher<real>(
        betot,
        bforce,
        chebyshev_size,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);

    // *** delta
    real cart_coords[natoms_pad][3] = {0.};
    for (int ii=0; ii<natoms_pad; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction_idx_modify] += delta;
    structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);
    neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);
    for (int bb=0; bb<batch_size; bb++)
        neighbor_list.find_info4mlff(
            binum[bb],
            &bilist[bb*natoms_pad],
            &bnumneigh[bb*natoms_pad],
            &bfirstneigh[bb*natoms_pad*umax_num_neigh_atoms],
            &brcs[bb*natoms_pad*umax_num_neigh_atoms*3],
            &btypes[bb*(natoms_pad+nghost)],
            nghost,
            umax_num_neigh_atoms);
    
    ai2pot::nnmtp::find_ef_launcher<real>(
        betot_,
        bforce_,
        chebyshev_size,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);


printf("1.1. energy = %.15lf\n", betot[0]);
printf("1.1. bforce[%d][%d][%d] calculated by custom code = %.15lf\n", 0, center_idx_modify, direction_idx_modify, bforce[0*(natoms_pad+nghost) + center_idx_modify][direction_idx_modify]);
printf("1.2. energy = %.15lf\n", betot_[0]);
printf("1.2. bforce[%d][%d][%d] calculated by finite difference method = %.15lf\n", 0, center_idx_modify, direction_idx_modify, -(betot_[0] - betot[0]) / delta);
printf("\n\nfind_efv_launcher:\n");
printf("\t2.1. Energy = %.15f\n", betot[0]);
printf("\t2.2. Force =\n");
for (int ii=0; ii<natoms_pad; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, bforce[0*(natoms_pad+nghost) + ii][0], bforce[0*(natoms_pad+nghost) + ii][1], bforce[0*(natoms_pad+nghost) + ii][2]);
}


TEST_F(NNMtpTest, find_loss_backward_launcer)
{
    real e_weight = 1.0;
    real f_weight = 1.0;
    real v_weight = 0.0;
    
    ai2pot::nnmtp::find_efv_launcher<real>(
        betot,
        bforce,
        bvirial,
        chebyshev_size,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);

    ai2pot::nnmtp::find_loss_backward_launcher<real>(
        bloss_der2coeffs,
        bloss_der2w0,
        bloss_der2w1,
        bloss_der2type_bias,
        e_weight,
        f_weight,
        v_weight,
        betot,
        betot_dft,
        bforce,
        bforce_dft,
        bvirial,
        bvirial_dft,
        chebyshev_size,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);

printf("1. loss_der2coeffs:\n");
for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
    printf("%.15f, ", bloss_der2coeffs[ii]);
printf("\n\n");

printf("2. loss_der2w0:\n");
for (int ii=0; ii<ntypes*num_neurons*mtp_param.alpha_scalar_moments(); ii++)
    printf("%.15f, ", bloss_der2w0[ii]);
printf("\n\n");

printf("3. loss_der2w1:\n");
for (int ii=0; ii<ntypes*num_neurons; ii++)
    printf("%.15f, ", bloss_der2w1[ii]);
printf("\n\n");

printf("4. loss_der2type_bias:\n");
for (int ii=0; ii<ntypes; ii++)
    printf("%.15f, ", bloss_der2type_bias[ii]);
printf("\n\n");
}



TEST_F(NNMtpTest, find_ef_loss_backward_launcer)
{
    real e_weight = 1.0;
    real f_weight = 1.0;
    
    ai2pot::nnmtp::find_ef_launcher<real>(
        betot,
        bforce,
        chebyshev_size,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);

    ai2pot::nnmtp::find_ef_loss_backward_launcher<real>(
        bloss_der2coeffs,
        bloss_der2w0,
        bloss_der2w1,
        bloss_der2type_bias,
        e_weight,
        f_weight,
        betot,
        betot_dft,
        bforce,
        bforce_dft,
        chebyshev_size,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);

printf("1. loss_der2coeffs:\n");
for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
    printf("%.15f, ", bloss_der2coeffs[ii]);
printf("\n\n");

printf("2. loss_der2w0:\n");
for (int ii=0; ii<ntypes*num_neurons*mtp_param.alpha_scalar_moments(); ii++)
    printf("%.15f, ", bloss_der2w0[ii]);
printf("\n\n");

printf("3. loss_der2w1:\n");
for (int ii=0; ii<ntypes*num_neurons; ii++)
    printf("%.15f, ", bloss_der2w1[ii]);
printf("\n\n");

printf("4. loss_der2type_bias:\n");
for (int ii=0; ii<ntypes; ii++)
    printf("%.15f, ", bloss_der2type_bias[ii]);
printf("\n\n");
}


TEST_F(NNMtpTest, find_descriptors_launcher) {
    int center_idx = 0;
    ai2pot::nnmtp::find_descriptors_launcher(
        bdescriptors,
        chebyshev_size,
        coeffs,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        (real (*)[3])brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);
    
    printf("1. Descriptors of atom#%d:\n", center_idx);
    for (int k=0; k<mtp_param.alpha_scalar_moments(); k++)
        printf("%.5lf, ", bdescriptors[0*natoms_pad*mtp_param.alpha_scalar_moments() + center_idx*mtp_param.alpha_scalar_moments() + k]);
    printf("\n");
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
