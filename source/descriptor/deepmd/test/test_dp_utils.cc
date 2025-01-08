#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>

#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../include/dp_utils.h"

class DpUtilsTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double rcut_smooth;
    bool pbc_xyz[3];
    int nghost;
    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> neighbor_list;

    int inum;
    int* ilist;
    int* numneigh;
    int* firstneigh;
    double* relative_coords;
    int* types;
    int ntypes;
    int umax_num_neigh_atoms;
    int* umax_num_neigh_atoms_lst;
    int* dp_firstneigh;

    static void SetUpTestSuite() {
        std::cout << "DpUtilsTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "DpUtilsTest (TestSuite) is tearing down...\n";
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

        // 42: 0; 16: 1
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
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;
        rcut = 3.3;
        rcut_smooth = 3.0;

        structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = ai2pot::NeighborList<double>(structure, rcut, pbc_xyz, true);
        
        ntypes = 2;
        umax_num_neigh_atoms_lst = (int*)malloc(sizeof(int) * ntypes);
        umax_num_neigh_atoms_lst[0] = 10;
        umax_num_neigh_atoms_lst[1] = 9;
        umax_num_neigh_atoms = 0;
        for (int ii=0; ii<ntypes; ii++)
            umax_num_neigh_atoms += umax_num_neigh_atoms_lst[ii];
        inum = neighbor_list.get_num_center_atoms();
        ilist = (int*)malloc(sizeof(int) * inum);
        numneigh = (int*)malloc(sizeof(int) * inum);
        firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        relative_coords = (double*)malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
        types = (int*)malloc(sizeof(int) * inum);
        dp_firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);

        neighbor_list.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            relative_coords,
            types,
            nghost,
            umax_num_neigh_atoms);
    }

    void TearDown() override {
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(relative_coords);
        free(types);
        free(umax_num_neigh_atoms_lst);
        free(dp_firstneigh);
    }
};


TEST_F(DpUtilsTest, find_dp_firstneigh) {
    ai2pot::deepPotSE::find_dp_firstneigh<double>(
        dp_firstneigh,
        inum,
        ilist,
        numneigh,
        firstneigh,
        relative_coords,
        types,
        ntypes,
        umax_num_neigh_atoms_lst);
    
    neighbor_list.show_in_prim_index();
    for (int ii=0; ii<inum; ii++) {
        printf("%d: ", ii);
        for (int jj=0; jj<umax_num_neigh_atoms; jj++) {
            printf("%d, ", dp_firstneigh[ii*umax_num_neigh_atoms + jj]);
        }
        printf("\n");
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}