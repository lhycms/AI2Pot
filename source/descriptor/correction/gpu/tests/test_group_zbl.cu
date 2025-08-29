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
#include "../include/zbl.cuh"


class GroupZBLTest : public ::testing::Test {
protected:
    int batch_size;
    int natoms_pad;
    int nghost;
    int ntypes;
    int* btypes;
    int* binum;
    int umax_num_neigh_atoms;
    int* bilist;
    int* bnumneigh;
    int* bfirstneigh;
    double (*brcs)[3];
    int* type_map;

    double coord_0[3];
    double coord_1[3];

    double *betot_ptr;
    double *betot_ptr_;
    double *bforces;
    double *bforces_;
    double *bvirial;
    double *bvirial_;

    double rmax;
    double rmin;
    int *Zis;
    int *Zjs;
    double *cks;
    double *dks;

    static void SetUpTestSuite() {
        std::cout << "GroupZBLTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "GroupZBLTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        batch_size = 1;
        natoms_pad = 2;
        nghost = 0;

        ntypes = 2;
        umax_num_neigh_atoms = 5;
        type_map = (int*)malloc(sizeof(int) * ntypes);
        type_map[0] = 20;
        type_map[1] = 24;

        binum = (int*)malloc(sizeof(int) * batch_size);
        binum[0] = 2;

        btypes = (int*)malloc(sizeof(int) * batch_size * (natoms_pad+nghost));
        btypes[0*(natoms_pad + nghost) + 0] = 0;
        btypes[0*(natoms_pad + nghost) + 1] = 1;

        bilist = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        bilist[0*natoms_pad + 0] = 0;
        bilist[0*natoms_pad + 1] = 1;
        bnumneigh = (int*)malloc(sizeof(int) * batch_size * natoms_pad);
        bnumneigh[0*natoms_pad + 0] = 1;
        bnumneigh[0*natoms_pad + 1] = 1;
        bfirstneigh = (int*)malloc(sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms);
        bfirstneigh[0*natoms_pad*umax_num_neigh_atoms + 0*umax_num_neigh_atoms + 0] = 1;
        bfirstneigh[0*natoms_pad*umax_num_neigh_atoms + 1*umax_num_neigh_atoms + 0] = 0;

        coord_0[0] = 1.0;
        coord_0[1] = 1.0;
        coord_0[2] = 1.0;
        coord_1[0] = 2.01;
        coord_1[1] = 2.02;
        coord_1[2] = 2.03;
        brcs = (double (*)[3])malloc(sizeof(double) * batch_size * natoms_pad * umax_num_neigh_atoms * 3);
        memset(brcs, 0.0, sizeof(double) * batch_size * natoms_pad * umax_num_neigh_atoms * 3);

        betot_ptr = (double*)malloc(sizeof(double) * batch_size);
        betot_ptr_ = (double*)malloc(sizeof(double) * batch_size);
        bforces = (double*)malloc(sizeof(double) * batch_size * natoms_pad * 3);
        bforces_ = (double*)malloc(sizeof(double) * batch_size * natoms_pad * 3);
        bvirial = (double*)malloc(sizeof(double) * batch_size * 9);
        bvirial_ = (double*)malloc(sizeof(double) * batch_size * 9);

        memset(betot_ptr, 0.0, sizeof(double) * batch_size);
        memset(betot_ptr_, 0.0, sizeof(double) * batch_size);
        memset(bforces, 0.0, sizeof(double) * batch_size * natoms_pad * 3);
        memset(bforces_, 0.0, sizeof(double) * batch_size * natoms_pad * 3);
        memset(bvirial, 0.0, sizeof(double) * batch_size * 9);
        memset(bvirial_, 0.0, sizeof(double) * batch_size * 9);

        rmax = 2.0;
        rmin = 1.0;
        cks = (double*)malloc(sizeof(double) * ntypes * ntypes * 4);
        dks = (double*)malloc(sizeof(double) * ntypes * ntypes * 4);
        for (int ii=0; ii<ntypes; ii++) {
            for (int jj=0; jj<ntypes; jj++) {
                int idx = ii*ntypes + jj;
                cks[idx*4 + 0] = 0.18175;
                cks[idx*4 + 1] = 0.50986;
                cks[idx*4 + 2] = 0.28022;
                cks[idx*4 + 3] = 0.02817;
                dks[idx*4 + 0] = 3.1998;
                dks[idx*4 + 1] = 0.94229;
                dks[idx*4 + 2] = 0.4029;
                dks[idx*4 + 3] = 0.20162;
            }
        }
    }

    void TearDown() override {
        free(type_map);
        free(btypes);
        free(binum);
        free(bilist);
        free(bnumneigh);
        free(bfirstneigh);
        free(brcs);
        free(cks);
        free(dks);
        free(betot_ptr);
        free(betot_ptr_);
        free(bforces);
        free(bforces_);
        free(bvirial);
        free(bvirial_);
    }
};  // class : GroupZBLTest


TEST_F(GroupZBLTest, efv_force_accuracy) {
    brcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    brcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    brcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    brcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    brcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    brcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    ai2pot::correction::correct_zbl_efv_launcher(
        betot_ptr,
        bforces,
        bvirial,
        rmax,
        rmin,
        cks,
        dks,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost);

    double delta = 1e-6;
    coord_0[1] += delta;
    brcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    brcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    brcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    brcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    brcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    brcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    ai2pot::correction::correct_zbl_efv_launcher(
        betot_ptr_,
        bforces_,
        bvirial_,
        rmax,
        rmin,
        cks,
        dks,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost);
printf("energy = %.10lf\n", betot_ptr[0]);
printf("1. Force[0][1] calculated by custom code = %.10lf\n", bforces[0*3+1]);
printf("2. Force[0][1] calculated by definition = %.10lf\n", -(betot_ptr_[0] - betot_ptr[0]) / delta);
}



TEST_F(GroupZBLTest, efv_virial_accuracy)
{
    brcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    brcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    brcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    brcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    brcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    brcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];

    ai2pot::correction::correct_zbl_efv_launcher(
        betot_ptr,
        bforces,
        bvirial,
        rmax,
        rmin,
        cks,
        dks,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost);

    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            bvirial_[aa*3 + bb] += coord_0[aa] * bforces[0*3 + bb];
        }
    }
    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            bvirial_[aa*3 + bb] += coord_1[aa] * bforces[1*3 + bb];
        }
    }

printf("1. Virial calculated by custom code =\n");
for (int ii=0; ii<9; ii++)
    printf("%.10lf, ", bvirial[ii]);
printf("\n");
printf("2. Virial calculated by definition =\n");
for (int ii=0; ii<9; ii++)
    printf("%.10lf, ", bvirial_[ii]);
printf("\n");
}



TEST_F(GroupZBLTest, ef_force_accuracy) {
    brcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    brcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    brcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    brcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    brcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    brcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    ai2pot::correction::correct_zbl_ef_launcher<double>(
        betot_ptr,
        bforces,
        rmax,
        rmin,
        cks,
        dks,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost);

    double delta = 1e-6;
    coord_0[1] += delta;
    brcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    brcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    brcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    brcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    brcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    brcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    ai2pot::correction::correct_zbl_ef_launcher<double>(
        betot_ptr_,
        bforces_,
        rmax,
        rmin,
        cks,
        dks,
        batch_size,
        natoms_pad,
        binum,
        bilist,
        bnumneigh,
        bfirstneigh,
        brcs,
        btypes,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost);
printf("energy = %.10lf\n", betot_ptr[0]);
printf("1. Force[0][1] calculated by custom code = %.10lf\n", bforces[0*3+1]);
printf("2. Force[0][1] calculated by definition = %.10lf\n", -(betot_ptr_[0] - betot_ptr[0]) / delta);
}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
