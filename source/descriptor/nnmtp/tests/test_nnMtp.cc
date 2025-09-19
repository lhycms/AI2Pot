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
#include <string>
#include <vector>

#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../include/mtpParam.h"
#include "../include/nnMtp.h"



class NNMtpTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    ai2pot::Structure<double> structure;

    double rcut;
    bool pbc_xyz[3];
    ai2pot::NeighborList<double> nblist;
    int umax_num_neigh_atoms;
    int ntypes;
    int inum;
    int *ilist;
    int *numneigh;
    int *firstneigh;
    double *rcs;
    int *types;
    int type_map[2];
    int nghost;

    std::vector<std::string> filenames;
    ai2pot::nnmtp::MtpParam mtp_param;

    double etot;
    double etot_;
    double (*forces)[3];
    double (*forces_)[3];
    double *virial;
    double *virial_;
    double rmax;
    double rmin;
    int chebyshev_size;
    int num_neurons;
    double *coeffs;
    double *w0;
    double *w1;
    double *type_bias;

    double zbl_rmax;
    double zbl_rmin;
    double *zbl_cks;
    double *zbl_dks;


    // Loss
    double e_weight;
    double f_weight;
    double v_weight;
    double etot_dft;
    double (*forces_dft)[3];
    double *virial_dft;

    // Loss derivatives
    double *loss_der2coeffs;
    double *loss_der2w0;
    double *loss_der2w1;
    double *loss_der2type_bias;

    // Utils
    double *descriptors;
    double *e_sites;

    static void SetUpTestSuite() {
        printf("NNMtpTest (TestSuite) is setting up...\n");
    }

    static void TearDownTestSuite() {
        printf("NNMtpTest (TestSuite) is tearing down...\n");
    }

    void SetUp() override {
        // Structure and NeighborList
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
        structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);

        rcut = 5.0;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;
        nblist = ai2pot::NeighborList<double>(structure, rcut, pbc_xyz, false);
        inum = 12;
        ntypes = 2;
        umax_num_neigh_atoms = 20;
        nghost = 0;
        ilist = (int*)malloc(sizeof(int) * inum);
        numneigh = (int*)malloc(sizeof(int) * inum);
        firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        rcs = (double*)malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
        types = (int*)malloc(sizeof(int) * inum);
        type_map[0] = 42;
        type_map[1] = 16;
        nblist.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);

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


        etot = 0.0;
        etot_ = 0.0;
        forces = (double (*)[3])malloc(sizeof(double) * inum * 3);
        memset(forces, 0, sizeof(double) * inum * 3);
        forces_ = (double (*)[3])malloc(sizeof(double) * inum * 3);
        memset(forces_, 0, sizeof(double) * inum * 3);
        virial = (double*)malloc(sizeof(double) * 9);
        memset(virial, 0, sizeof(double) * 9);
        virial_ = (double*)malloc(sizeof(double) * 9);
        memset(virial_, 0, sizeof(double) * 9);

        chebyshev_size = 8;
        num_neurons = 30;
        rmax = 5.0;
        rmin = 0.0;
        coeffs = (double*)malloc(sizeof(double) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        for (int ii=0; ii<ntypes * ntypes * mtp_param.nmus() * chebyshev_size; ii++)
            coeffs[ii] = 0.01 + 0.001 * ii;
        w0 = (double*)malloc(sizeof(double) * ntypes * num_neurons * mtp_param.alpha_scalar_moments());
        for (int ii=0; ii<ntypes * num_neurons * mtp_param.alpha_scalar_moments(); ii++)
            w0[ii] = 0.1 + 0.001 * ii;
        w1 = (double*)malloc(sizeof(double) * ntypes * num_neurons);
        for (int ii=0; ii<ntypes * num_neurons; ii++)
            w1[ii] = 0.1 + 0.001 * ii;
        type_bias = (double*)malloc(sizeof(double) * ntypes);
        type_bias[0] = -0.1;
        type_bias[1] = -0.2;

        zbl_rmax = 0.0;
        zbl_rmin = 0.0;
        zbl_cks = nullptr;
        zbl_dks = nullptr;


        // Loss
        e_weight = 1.0;
        f_weight = 1.0;
        v_weight = 1.0;
        etot_dft = 0.0;
        forces_dft = (double (*)[3])malloc(sizeof(double) * inum * 3);
        memset(forces_dft, 0.0, sizeof(double) * inum * 3);
        virial_dft = (double *)malloc(sizeof(double) * 9);
        memset(virial_dft, 0.0, sizeof(double) * 9);

        // Loss derivative
        loss_der2coeffs = (double*)malloc(sizeof(double) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        memset(loss_der2coeffs, 0, sizeof(double) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        loss_der2w0 = (double*)malloc(sizeof(double) * ntypes * num_neurons * mtp_param.alpha_scalar_moments());
        memset(loss_der2w0, 0, sizeof(double) * ntypes * num_neurons * mtp_param.alpha_scalar_moments());
        loss_der2w1 = (double*)malloc(sizeof(double) * ntypes * num_neurons);
        memset(loss_der2w1, 0, sizeof(double) * ntypes * num_neurons);
        loss_der2type_bias = (double*)malloc(sizeof(double) * ntypes);
        memset(loss_der2type_bias, 0, sizeof(double) * ntypes);

        // Utils
        descriptors = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments());
        memset(descriptors, 0.0, sizeof(double) * inum * mtp_param.alpha_scalar_moments());
        e_sites = (double*)malloc(sizeof(double) * inum);
        memset(e_sites, 0.0, sizeof(double) * inum);
    }


    void TearDown() override {
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);
        free(forces);
        free(forces_);
        free(virial);
        free(virial_);
        free(coeffs);
        free(w0);
        free(w1);
        free(type_bias);

        // Loss
        free(forces_dft);
        free(virial_dft);

        // Loss derivatives
        free(loss_der2coeffs);
        free(loss_der2w0);
        free(loss_der2w1);
        free(loss_der2type_bias);

        // Utils
        free(descriptors);
        free(e_sites);
    }
};  // class : NNMtpTEst


TEST_F(NNMtpTest, find_ef_accuracy) {
    int center_idx_modify = 1;
    int direction1_idx_modify = 2;
    double delta = 1e-8;

    ai2pot::nnmtp::NNMtp<double>::find_ef(
        etot,
        forces,
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
        rmin);

    // *** delta
    double cart_coords[12][3] = {0};
    for (int ii=0; ii<inum; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction1_idx_modify] += delta;
    structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);
    nblist = ai2pot::NeighborList<double>(structure, rcut, pbc_xyz, true);
    nblist.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);
    // *** delta

    ai2pot::nnmtp::NNMtp<double>::find_ef(
        etot_,
        forces_,
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
        rmin);

printf("1.1. energy = %.15lf\n", etot);
printf("1.1. force[%d][%d] calculated by custom code = %.15lf\n", center_idx_modify, direction1_idx_modify, forces[center_idx_modify][direction1_idx_modify]);
printf("1.2. energy = %.15lf\n", etot_);
printf("1.2. force[%d][%d] calculated by finite difference method = %.15lf\n", center_idx_modify, direction1_idx_modify, -(etot_ - etot) / delta);
printf("2.1. energy = %.15lf\n", etot);
printf("2.2. force=\n");
for (int ii=0; ii<inum; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, forces[ii][0], forces[ii][1], forces[ii][2]);
}


TEST_F(NNMtpTest, find_efv_accuracy) {
    int center_idx_modify = 1;
    int direction1_idx_modify = 2;
    double delta = 1e-8;

    ai2pot::nnmtp::NNMtp<double>::find_efv(
        etot,
        forces,
        virial,
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
        rmin);

    // *** delta
    double cart_coords[12][3] = {0};
    for (int ii=0; ii<inum; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction1_idx_modify] += delta;
    structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);
    nblist = ai2pot::NeighborList<double>(structure, rcut, pbc_xyz, true);
    nblist.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);
    // *** delta

    ai2pot::nnmtp::NNMtp<double>::find_efv(
        etot_,
        forces_,
        virial_,
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
        rmin);

printf("1.1. energy = %.15lf\n", etot);
printf("1.1. force[%d][%d] calculated by custom code = %.15lf\n", center_idx_modify, direction1_idx_modify, forces[center_idx_modify][direction1_idx_modify]);
printf("1.2. energy = %.15lf\n", etot_);
printf("1.2. force[%d][%d] calculated by finite difference method = %.15lf\n", center_idx_modify, direction1_idx_modify, -(etot_ - etot) / delta);
printf("2.1. energy = %.15lf\n", etot);
printf("2.2. force=\n");
for (int ii=0; ii<inum; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, forces[ii][0], forces[ii][1], forces[ii][2]);
}



TEST_F(NNMtpTest, find_ef_loss)
{
    double loss = 0.0;
    ai2pot::nnmtp::NNMtp<double>::find_ef(
        etot,
        forces,
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
        rmin);

    ai2pot::nnmtp::NNMtpLoss<double>::find_ef_loss(
        loss,
        inum,
        ilist,
        e_weight,
        f_weight,
        etot,
        etot_dft,
        forces,
        forces_dft);
printf("\t1. ef_loss = %.15lf\n", loss);
}


TEST_F(NNMtpTest, find_loss)
{
    double loss = 0.0;
    double v_weight = 0.0;
    ai2pot::nnmtp::NNMtp<double>::find_efv(
        etot,
        forces,
        virial,
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
        rmin);

    ai2pot::nnmtp::NNMtpLoss<double>::find_loss(
        loss,
        inum,
        ilist,
        e_weight,
        f_weight,
        v_weight,
        etot,
        etot_dft,
        forces,
        forces_dft,
        virial,
        virial_dft);
printf("\t1. efv_loss = %.15lf\n", loss);
}


TEST_F(NNMtpTest, find_ef_loss_backward) {
    ai2pot::nnmtp::NNMtp<double>::find_ef(
        etot,
        forces,
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
        rmin);

    ai2pot::nnmtp::NNMtpLoss<double>::find_ef_loss_backward(
        loss_der2coeffs,
        loss_der2w0,
        loss_der2w1,
        loss_der2type_bias,
        e_weight,
        f_weight,
        etot,
        etot_dft,
        forces,
        forces_dft,
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
        rmin);

printf("1. loss_der2coeffs:\n");
for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
    printf("%.15f, ", loss_der2coeffs[ii]);
printf("\n\n");

printf("2. loss_der2w0:\n");
for (int ii=0; ii<ntypes*num_neurons*mtp_param.alpha_scalar_moments(); ii++)
    printf("%.15f, ", loss_der2w0[ii]);
printf("\n\n");

printf("3. loss_der2w1:\n");
for (int ii=0; ii<ntypes*num_neurons; ii++)
    printf("%.15f, ", loss_der2w1[ii]);
printf("\n\n");

printf("4. loss_der2type_bias:\n");
for (int ii=0; ii<ntypes; ii++)
    printf("%.15f, ", loss_der2type_bias[ii]);
printf("\n\n");
}



TEST_F(NNMtpTest, find_loss_backward) {
    v_weight = 0.0;
    
    ai2pot::nnmtp::NNMtp<double>::find_efv(
        etot,
        forces,
        virial,
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
        rmin);

    ai2pot::nnmtp::NNMtpLoss<double>::find_loss_backward(
        loss_der2coeffs,
        loss_der2w0,
        loss_der2w1,
        loss_der2type_bias,
        e_weight,
        f_weight,
        v_weight,
        etot,
        etot_dft,
        forces,
        forces_dft,
        virial,
        virial_dft,
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
        rmin);

printf("1. loss_der2coeffs:\n");
for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
    printf("%.15f, ", loss_der2coeffs[ii]);
printf("\n\n");

printf("2. loss_der2w0:\n");
for (int ii=0; ii<ntypes*num_neurons*mtp_param.alpha_scalar_moments(); ii++)
    printf("%.15f, ", loss_der2w0[ii]);
printf("\n\n");

printf("3. loss_der2w1:\n");
for (int ii=0; ii<ntypes*num_neurons; ii++)
    printf("%.15f, ", loss_der2w1[ii]);
printf("\n\n");

printf("4. loss_der2type_bias:\n");
for (int ii=0; ii<ntypes; ii++)
    printf("%.15f, ", loss_der2type_bias[ii]);
printf("\n\n");
}



TEST_F(NNMtpTest, find_descriptors) {
    int center_idx = 0;
    
    ai2pot::nnmtp::NNMtp<double>::find_descriptors(
        descriptors,
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
        rmin);
    
    printf("1. Descriptors of atom#%d:\n", center_idx);
    for (int k=0; k<mtp_param.alpha_scalar_moments(); k++)
        printf("%.5lf, ", descriptors[center_idx*mtp_param.alpha_scalar_moments() + k]);
    printf("\n");
}


/*
TEST_F(NNMtpTest, find_e_sites) {
    ai2pot::nnmtp::NNMtp<double>::find_e_sites(
        e_sites,
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
        rmin;

double result = 0;
for (int ii=0; ii<inum; ii++)
    result += e_sites[ii];
printf("1. Sum of e_sites = %.15lf\n", result);
}
*/

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}