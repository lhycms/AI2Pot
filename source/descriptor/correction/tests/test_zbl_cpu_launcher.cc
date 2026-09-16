#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>


#include "../include/zbl_cpu_launcher.h"

class ZBLCPULauncherTest : public ::testing::Test
{
protected:
    int batch_size;
    int natoms_pad;
    int ntypes;
    int umax_num_neigh_atoms;
    int *btypes;
    int *binum;
    int *bilist;
    int *bnumneigh;
    int *bfirstneigh;
    double (*brcs)[3];
    int *type_map;
    int nghost;

    double coord_0[3];
    double coord_1[3];

    double *betot;
    double *betot_;
    double (*bforce)[3];
    double (*bforce_)[3];
    double *bvirial;
    double *bvirial_;

    double rmax;
    double rmin;
    int *Zis;
    int *Zjs;
    double *cks;
    double *dks;


    static void SetUpTestSuite() {
        std::cout << "ZBLCPULauncher (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "ZBLCPULauncher (TestSuite) us tearing down...\n";
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

        // 
        binum = (int*)malloc(sizeof(int) * batch_size);
        binum[0] = 2;
        bilist = (int*)malloc(sizeof(int) * batch_size * binum[0]);
        bnumneigh = (int*)malloc(sizeof(int) * batch_size * binum[0]);
        bfirstneigh = (int*)malloc(sizeof(int) * batch_size * binum[0] * umax_num_neigh_atoms);
        brcs = (double (*)[3])malloc(sizeof(double) * batch_size * binum[0] * umax_num_neigh_atoms * 3);
        btypes = (int*)malloc(sizeof(int) * batch_size * binum[0]);
        
        btypes[0] = 0;
        btypes[1] = 1;
        bilist[0] = 0;
        bilist[1] = 1;
        bnumneigh[0] = 1;
        bnumneigh[1] = 1;
        bfirstneigh[0] = 1;
        bfirstneigh[1*umax_num_neigh_atoms + 0] = 0;

        coord_0[0] = 1.0;
        coord_0[1] = 1.0;
        coord_0[2] = 1.0;
        coord_1[0] = 2.01;
        coord_1[1] = 2.02;
        coord_1[2] = 2.03;
        memset(brcs, 0.0, sizeof(double) * batch_size * binum[0] * umax_num_neigh_atoms * 3);

        betot = (double*)malloc(sizeof(double) * batch_size);
        betot_ = (double*)malloc(sizeof(double) * batch_size);
        betot[0] = 0;
        betot_[0] = 0;
        bforce = (double (*)[3])malloc(sizeof(double) * binum[0] * 3);
        bforce_ = (double (*)[3])malloc(sizeof(double) * binum[0] * 3);
        bvirial = (double *)malloc(sizeof(double) * batch_size * 9);
        bvirial_ = (double *)malloc(sizeof(double) * batch_size * 9);
        memset(bforce, 0.0, sizeof(double) * binum[0] * 3);
        memset(bforce_, 0.0, sizeof(double) * binum[0] * 3);
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
        
        free(binum);
        free(bilist);
        free(bnumneigh);
        free(bfirstneigh);
        free(brcs);
        free(btypes);

        free(betot);
        free(betot_);
        free(bforce);
        free(bforce_);
        free(bvirial);
        free(bvirial_);

        free(cks);
        free(dks);
    }
};  // class : ZBLCPULauncher


TEST_F(ZBLCPULauncherTest, force_accuracy) {
    brcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    brcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    brcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    brcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    brcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    brcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];

    ai2pot::correction::find_zbl_efv_cpu_launcher(
        betot,
        bforce,
        bvirial,
        type_map,
        type_map,
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

    ai2pot::correction::find_zbl_efv_cpu_launcher(
        betot_,
        bforce_,
        bvirial_,
        type_map,
        type_map,
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


    printf("0. Energy = %.10lf\n", betot[0]);
    printf("1. Force[0][1] calculated by custom code = %.10lf\n", bforce[0][1]);
    printf("2. Force[0][1] calculated by definition = %.10lf\n", -(betot_[0] - betot[0]) / delta);
}


TEST_F(ZBLCPULauncherTest, virial_accuracy)
{
    brcs[0*umax_num_neigh_atoms + 0][0] = coord_1[0] - coord_0[0];
    brcs[0*umax_num_neigh_atoms + 0][1] = coord_1[1] - coord_0[1];
    brcs[0*umax_num_neigh_atoms + 0][2] = coord_1[2] - coord_0[2];
    brcs[1*umax_num_neigh_atoms + 0][0] = coord_0[0] - coord_1[0];
    brcs[1*umax_num_neigh_atoms + 0][1] = coord_0[1] - coord_1[1];
    brcs[1*umax_num_neigh_atoms + 0][2] = coord_0[2] - coord_1[2];

    ai2pot::correction::find_zbl_efv_cpu_launcher(
        betot,
        bforce,
        bvirial,
        type_map,
        type_map,
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
            bvirial_[aa*3 + bb] += coord_0[aa] * bforce[0][bb];
        }
    }
    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            bvirial_[aa*3 + bb] += coord_1[aa] * bforce[1][bb];
        }
    }

printf("1. Virial calculated by custom code =\n");
for (int aa=0; aa<3; aa++) {
    for (int bb=0; bb<3; bb++) {
        printf("%.10lf, ", bvirial[aa*3 + bb]);
    }
    printf("\n");
}
printf("2. Virial calculated by definition =\n");
for (int aa=0; aa<3; aa++) {
    for (int bb=0; bb<3; bb++) {
        printf("%.10lf, ", bvirial_[aa*3 + bb]);
    }
    printf("\n");
}
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
