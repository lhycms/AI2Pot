#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "../include/altbc.h"


class AltbcTest : public ::testing::Test
{
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];

    int inum;
    int* ilist;
    int* numneigh;
    int* firstneigh;
    double* rcs;
    int* types;
    int nghost;
    int umax_num_neigh_atoms;

    double angle_threshold;

    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> nblist;

    static void SetUpTestSuite() {
        printf("AltbcTest (TestSuite) is setting up...\n");
    }

    static void TearDownTestSuite() {
        printf("AltbcTest (TestSuite) is tearing down...\n");
    }

    void SetUp() override {
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

        atomic_numbers[0] = 42;
        atomic_numbers[1] = 16;
        atomic_numbers[2] = 16;
        atomic_numbers[3] = 42;
        atomic_numbers[4] = 16;
        atomic_numbers[5] = 16;
        atomic_numbers[6] = 42;
        atomic_numbers[7] = 16;
        atomic_numbers[8] = 16;
        atomic_numbers[9] = 42; 
        atomic_numbers[10] = 16;
        atomic_numbers[11] = 16;

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

        rcut = 3.2;
        bin_size_xyz[0] = 2.5;
        bin_size_xyz[1] = 2.5;
        bin_size_xyz[2] = 2.5;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;

        umax_num_neigh_atoms = 19;
        angle_threshold = 120.0;

        structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        nblist = ai2pot::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, false);
        inum = structure.get_num_atoms();
        nghost = 0;
        ilist = (int*)malloc(inum * sizeof(int));
        numneigh = (int*)malloc(inum * sizeof(int));
        firstneigh = (int*)malloc(inum * umax_num_neigh_atoms * sizeof(int));
        rcs = (double*)malloc(inum * umax_num_neigh_atoms * 3 * sizeof(double));
        types = (int*)malloc((inum+nghost) * sizeof(int));
    }

    void TearDown() override {
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);
    }
};   // class: AltbcTest


TEST_F(AltbcTest, find_long_short_bonds) {
    nblist.find_info4mlff(inum,
                         ilist,
                         numneigh,
                         firstneigh,
                         rcs,
                         types,
                         nghost,
                         umax_num_neigh_atoms);
    std::vector<double> long_bond_lengths;
    std::vector<double> short_bond_lengths;
    ai2pot::gst::Altbc<double>::find_long_short_bonds(short_bond_lengths,
                                                      long_bond_lengths,
                                                      inum,
                                                      ilist,
                                                      numneigh,
                                                      firstneigh,
                                                      rcs,
                                                      types,
                                                      nghost,
                                                      umax_num_neigh_atoms,
                                                      angle_threshold);
for (int ii=0; ii<long_bond_lengths.size(); ii++)
    printf("%g, %g\n", long_bond_lengths[ii], short_bond_lengths[ii]);
printf("\n");
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}