#include <gtest/gtest.h>
#include <iostream>

#include "../include/zbl.h"


class GroupZBLTest : public ::testing::Test
{
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
    double* forces;
    double* forces_;
    double virial[9];
    double virial_[9];

    double rmax;
    double rmin;
    int *Zis;
    int *zjs;
    double *cks;
    double *dks;
    ai2pot::correction::GroupZBL<double> group_zbl;

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
        group_zbl = ai2pot::correction::GroupZBL<double>(ntypes, 
                                                         type_map, 
                                                         type_map, 
                                                         rmax,
                                                         rmin,
                                                         cks,
                                                         dks);
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
};  // class : GroupTest


TEST_F(GroupZBLTest, correct_e_sites) {
    double *e_sites = (double*)malloc(sizeof(double) * inum);
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    group_zbl.correct_e_sites(e_sites,
                            inum,
                            ilist,
                            numneigh,
                            firstneigh,
                            rcs,
                            types,
                            ntypes,
                            type_map,
                            umax_num_neigh_atoms,
                            0);
    
    double etot = 0.0;
    for (int ii=0; ii<inum; ii++)
        etot += e_sites[ii];
    printf("0. Energy = %.10lf\n", etot);

    free(e_sites);
}


TEST_F(GroupZBLTest, force_accuracy) {
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    group_zbl.correct_efv(etot,
                          forces,
                          virial,
                          inum,
                          ilist,
                          numneigh,
                          firstneigh,
                          rcs,
                          types,
                          ntypes,
                          type_map,
                          umax_num_neigh_atoms,
                          0);

    double delta = 1e-6;
    coord_0[1] += delta;
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    group_zbl.correct_efv(etot_,
                          forces_,
                          virial_,
                          inum,
                          ilist,
                          numneigh,
                          firstneigh,
                          rcs,
                          types,
                          ntypes,
                          type_map,
                          umax_num_neigh_atoms,
                          0);
    
    printf("0. Energy = %.10lf\n", etot);
    printf("1. Force[0][1] calculated by custom code = %.10lf\n", forces[0*3+1]);
    printf("2. Force[0][1] calculated by definition = %.10lf\n", -(etot_ - etot) / delta);
}


TEST_F(GroupZBLTest, virial_accuracy) {
    rcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    rcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    rcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    rcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    rcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    rcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];
    group_zbl.correct_efv(etot,
                          forces,
                          virial,
                          inum,
                          ilist,
                          numneigh,
                          firstneigh,
                          rcs,
                          types,
                          ntypes,
                          type_map,
                          umax_num_neigh_atoms,
                          0);
    

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


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
