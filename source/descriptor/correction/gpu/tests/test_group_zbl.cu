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
    int ntypes;
    int* types;
    int inum;
    int umax_num_neigh_atoms;
    int* ilist;
    int* numneigh;
    int* firstneigh;
    double (*rcs)[3];
    int* type_map;

    double coord_0[3];
    double coord_1[3];

    double etot;
    double etot_;
    double *forces;
    double *forces_;
    double virial[9];
    double virial_[9];

    double rmax;
    double rmin;
    int *Zis;
    int *zjs;
    double *cks;
    double *dks;

    static void SetUpTestSuite() {
        std::cout << "GroupZBLTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "GroupZBLTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        ntypes = 2;
        umax_num_neigh_atoms = 5;
        type_map = (int*)malloc(sizeof(int) * ntypes);
        type_map[0] = 20;
        type_map[1] = 24;

        inum = 2;
        types = (int*)malloc(sizeof(int) * inum);
        types[0] = 0;
        types[1] = 1;

        ilist = (int*)malloc(sizeof(int) * inum);
        ilist[0] = 0;
        ilist[1] = 1;
        numneigh = (int*)malloc(sizeof(int) * inum);
        numneigh[0] = 1;
        numneigh[1] = 1;
        firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        firstneigh[0] = 1;
        firstneigh[1*umax_num_neigh_atoms + 0] = 0;

        coord_0[0] = 1.0;
        coord_0[1] = 1.0;
        coord_0[2] = 1.0;
        coord_1[0] = 2.01;
        coord_1[1] = 2.02;
        coord_1[2] = 2.03;
        rcs = (double (*)[3])malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
        memset(rcs, 0.0, sizeof(double) * inum * umax_num_neigh_atoms * 3);

        etot = 0;
        etot_ = 0;
        forces = (double*)malloc(sizeof(double) * inum * 3);
        forces_ = (double*)malloc(sizeof(double) * inum * 3);

        memset(forces, 0.0, sizeof(double) * inum * 3);
        memset(forces_, 0.0, sizeof(double) * inum * 3);
        memset(virial, 0.0, sizeof(double) * 9);
        memset(virial_, 0.0, sizeof(double) * 9);

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
        free(types);
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(cks);
        free(dks);
        free(forces);
        free(forces_);
    }
};  // class : GroupZBLTest


TEST_F(GroupZBLTest, efv_force_accuracy) {
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    ai2pot::correction::correct_zbl_efv_launcher(etot,
                                                 forces,
                                                 virial,
                                                 rmax,
                                                 rmin,
                                                 cks,
                                                 dks,
                                                 inum,
                                                 ilist,
                                                 numneigh,
                                                 firstneigh,
                                                 rcs,
                                                 types,
                                                 ntypes,
                                                 type_map,
                                                 umax_num_neigh_atoms);

    double delta = 1e-6;
    coord_0[1] += delta;
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    ai2pot::correction::correct_zbl_efv_launcher(etot_,
                                                 forces_,
                                                 virial_,
                                                 rmax,
                                                 rmin,
                                                 cks,
                                                 dks,
                                                 inum,
                                                 ilist,
                                                 numneigh,
                                                 firstneigh,
                                                 rcs,
                                                 types,
                                                 ntypes,
                                                 type_map,
                                                 umax_num_neigh_atoms);
printf("energy = %.10lf\n", etot);
printf("1. Force[0][1] calculated by custom code = %.10lf\n", forces[0*3+1]);
printf("2. Force[0][1] calculated by definition = %.10lf\n", -(etot_ - etot) / delta);
}


TEST_F(GroupZBLTest, efv_virial_accuracy)
{
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];

    ai2pot::correction::correct_zbl_efv_launcher<double>(etot,
                                                         forces,
                                                         virial,
                                                         rmax,
                                                         rmin,
                                                         cks,
                                                         dks,
                                                         inum,
                                                         ilist,
                                                         numneigh,
                                                         firstneigh,
                                                         rcs,
                                                         types,
                                                         ntypes,
                                                         type_map,
                                                         umax_num_neigh_atoms);

    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            virial_[aa*3 + bb] += coord_0[aa] * forces[0*3 + bb];
        }
    }
    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            virial_[aa*3 + bb] += coord_1[aa] * forces[1*3 + bb];
        }
    }

printf("1. Virial calculated by custom code =\n");
for (int ii=0; ii<9; ii++)
    printf("%.10lf, ", virial[ii]);
printf("\n");
printf("2. Virial calculated by definition =\n");
for (int ii=0; ii<9; ii++)
    printf("%.10lf, ", virial_[ii]);
printf("\n");
}


TEST_F(GroupZBLTest, ef_force_accuracy) {
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    ai2pot::correction::correct_zbl_ef_launcher<double>(etot,
                                                        forces,
                                                        rmax,
                                                        rmin,
                                                        cks,
                                                        dks,
                                                        inum,
                                                        ilist,
                                                        numneigh,
                                                        firstneigh,
                                                        rcs,
                                                        types,
                                                        ntypes,
                                                        type_map,
                                                        umax_num_neigh_atoms);

    double delta = 1e-6;
    coord_0[1] += delta;
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    ai2pot::correction::correct_zbl_ef_launcher<double>(etot_,
                                                        forces_,
                                                        rmax,
                                                        rmin,
                                                        cks,
                                                        dks,
                                                        inum,
                                                        ilist,
                                                        numneigh,
                                                        firstneigh,
                                                        rcs,
                                                        types,
                                                        ntypes,
                                                        type_map,
                                                        umax_num_neigh_atoms);
printf("energy = %.10lf\n", etot);
printf("1. Force[0][1] calculated by custom code = %.10lf\n", forces[0*3+1]);
printf("2. Force[0][1] calculated by definition = %.10lf\n", -(etot_ - etot) / delta);
}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}