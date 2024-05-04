#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../nblist/include/structure.h"
#include "../include/se.h"
#include "../../../core/include/arrayUtils.h"


class PairTildeRTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    //double bin_size_xyz[3];
    bool pbc_xyz[3];

    int center_atomic_number;
    int neigh_atomic_number;
    double rcut_smooth;
    int num_neigh_atoms;    // dpse zero-paddning size

    matersdk::Structure<double> structure;
    matersdk::NeighborList<double> neighbor_list;

    // Variables to simulate info of `LAMMPS_NS::LAMMPS* lmp`
    int inum;           // 中心原子的数目
    int* ilist;         // 中心原子在 supercell 中的 index
    int* numneigh;      // 各个中心原子的近邻原子数目
    int** firstneigh;   // 近邻原子在 supercell 中的 index
    int* types;         // supercell 中所有原子的元素种类
    double** x;         // supercell 中所有原子的位置


    static void SetUpTestSuite() {
        std::cout << "PairTildeRTest TestSuite is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "PairTildeRTest TestSuite is tearing down...\n";
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

        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;

        rcut = 3.3;
        rcut_smooth = 3.0;
        center_atomic_number = 42;
        neigh_atomic_number = 42;
        num_neigh_atoms = 14;

        structure = matersdk::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = matersdk::NeighborList<double>(structure, rcut, pbc_xyz, true);

        // Variables to simulate the info of `LAMMPS_NS::LAMMPS*`
        inum = neighbor_list.get_num_center_atoms();
        
        ilist = (int*)malloc(sizeof(int) * inum);
        int prim_num_atoms = neighbor_list.get_binLinkedList().get_supercell().get_prim_num_atoms();
        int prim_cell_idx = neighbor_list.get_binLinkedList().get_supercell().get_prim_cell_idx();
        for (int ii=0; ii<inum; ii++)
            ilist[ii] = ii + prim_cell_idx * prim_num_atoms;
        
        numneigh = (int*)malloc(sizeof(int) * inum);
        for (int ii=0; ii<inum; ii++) {
            numneigh[ii] = neighbor_list.get_neighbor_lists()[ii].size();
        }

        firstneigh = (int**)malloc(sizeof(int*) * inum);
        for (int ii=0; ii<inum; ii++)
            firstneigh[ii] = (int*)malloc(sizeof(int) * numneigh[ii]);
        for (int ii=0; ii<inum; ii++) {
            for (int jj=0; jj<numneigh[ii]; jj++) {
                firstneigh[ii][jj] = neighbor_list.get_neighbor_lists()[ii][jj];    
                printf("%d, ", firstneigh[ii][jj]);
            }
            printf("\n");
        }

        int supercell_num_atoms = neighbor_list.get_binLinkedList().get_supercell().get_num_atoms();
        x = (double**)neighbor_list.get_binLinkedList().get_supercell().get_structure().get_cart_coords();
        types = (int*)neighbor_list.get_binLinkedList().get_supercell().get_structure().get_atomic_numbers();
    }


    void TearDown() override {
        // Step . Free memory
        free(ilist);
        free(numneigh);
        for (int ii=0; ii<inum; ii++)
            free(firstneigh[ii]);
        free(firstneigh);
    }
};


TEST_F(PairTildeRTest, generate_for_lmp) {
    double*** pair_tilde_r = matersdk::deepPotSE::PairTildeR<double>::generate(
                                inum, ilist, numneigh, firstneigh,
                                x, types,
                                center_atomic_number, neigh_atomic_number, num_neigh_atoms,
                                rcut, rcut_smooth);
    
    int num_center_atoms = 0;
    int center_atom_idx;
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        if (types[center_atom_idx] == center_atomic_number)
            num_center_atoms += 1;
    }

    printf("{ %d - %d }:\n", center_atomic_number, neigh_atomic_number);
    for (int ii=0; ii<num_center_atoms; ii++) {
        for (int jj=0; jj<num_neigh_atoms; jj++) {
            printf("[%10f, %10f, %10f, %10f]\n", pair_tilde_r[ii][jj][0], pair_tilde_r[ii][jj][1], pair_tilde_r[ii][jj][2], pair_tilde_r[ii][jj][3]);
        }
    }

    // Step. Free memory
    for (int ii=0; ii<num_center_atoms; ii++) {
        for (int jj=0; jj<num_neigh_atoms; jj++) {
            free(pair_tilde_r[ii][jj]);
        }
        free(pair_tilde_r[ii]);
    }
    free(pair_tilde_r);
}


TEST_F(PairTildeRTest, deriv_for_lmp) {
    double**** pair_tilde_r_deriv = matersdk::deepPotSE::PairTildeR<double>::deriv(
                                        inum, ilist, numneigh, firstneigh,
                                        x, types,
                                        center_atomic_number, neigh_atomic_number, num_neigh_atoms,
                                        rcut, rcut_smooth);
    
    int num_center_atoms = 0;
    int center_atom_idx = 0;
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        if (types[center_atom_idx] == center_atomic_number)
            num_center_atoms++;
    }


    for (int ii=0; ii<num_center_atoms; ii++) {
        for (int jj=0; jj<num_neigh_atoms; jj++) {
            printf("[%4d, %4d] -- [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f]\n",
                ii, jj,
                pair_tilde_r_deriv[ii][jj][0][0], pair_tilde_r_deriv[ii][jj][0][1], pair_tilde_r_deriv[ii][jj][0][2], 
                pair_tilde_r_deriv[ii][jj][1][0], pair_tilde_r_deriv[ii][jj][1][1], pair_tilde_r_deriv[ii][jj][1][2],
                pair_tilde_r_deriv[ii][jj][2][0], pair_tilde_r_deriv[ii][jj][2][1], pair_tilde_r_deriv[ii][jj][2][2],
                pair_tilde_r_deriv[ii][jj][3][0], pair_tilde_r_deriv[ii][jj][3][1], pair_tilde_r_deriv[ii][jj][3][2]
            );
        }
    }
}




class TildeRTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double rcut_smooth;
    //double bin_size_xyz[3];
    bool pbc_xyz[3];

    int num_center_atomic_numbers;      // 中心原子的种类数     e.g. MoS2 -- 2
    int* center_atomic_numbers_lst;     // 中心原子的元素种类   e.g. MoS2 -- [42, 16]
    int num_neigh_atomic_numbers;       // 近邻原子的种类数     e.g. MoS2 -- 2
    int* neigh_atomic_numbers_lst;      // 近邻原子的元素种类   e.g. MoS2 -- [42, 16]
    int* num_neigh_atoms_lst;           // 近邻原子的数目，决定了 zero-padding 的尺寸   e.g. MoS2 -- [100, 80]

    matersdk::Structure<double> structure;
    matersdk::NeighborList<double> neighbor_list;

    // Variables to simulate info of `LAMMPS_NS::LAMMPS* lmp`
    int inum;               // 中心原子的数目
    int* ilist;             // 中心原子在 supercell 中的index
    int* numneigh;          // 各个中心原子的近邻原子数目
    int** firstneigh;       // 近邻原子在 supercell 中的index
    int* types;             // supercell 中所有院子的元素种类
    double** x;             // supercell中所有原子的位置


    static void SetUpTestSutie() {
        std::cout << "TildeRTest TestSuite is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "TildeRTest TestSuite is tearing down...\n";
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

        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;

        rcut = 3.3;
        rcut_smooth = 3.0;
        num_center_atomic_numbers = 2;
        center_atomic_numbers_lst = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
        center_atomic_numbers_lst[0] = 16;
        center_atomic_numbers_lst[1] = 42;
        num_neigh_atomic_numbers = 2;
        neigh_atomic_numbers_lst = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
        neigh_atomic_numbers_lst[0] = 16;
        neigh_atomic_numbers_lst[1] = 42;
        // 决定了 zero-padding 的尺寸
        num_neigh_atoms_lst = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
        num_neigh_atoms_lst[0] = 8;    
        num_neigh_atoms_lst[1] = 7;

        structure = matersdk::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = matersdk::NeighborList<double>(structure, rcut, pbc_xyz, true);

        // Variables to simulate the info of `LAMMPS_NS::LAMMPS* lmp`
        inum = neighbor_list.get_num_center_atoms();

        ilist = (int*)malloc(sizeof(int) * inum);
        int prim_num_atoms = neighbor_list.get_binLinkedList().get_supercell().get_prim_num_atoms();
        int prim_cell_idx = neighbor_list.get_binLinkedList().get_supercell().get_prim_cell_idx();
        for (int ii=0; ii<inum; ii++) 
            ilist[ii] = prim_cell_idx * prim_num_atoms + ii;
        
        numneigh = (int*)malloc(sizeof(int) * inum);
        for (int ii=0; ii<inum; ii++) 
            numneigh[ii] = neighbor_list.get_neighbor_lists()[ii].size();
        
        firstneigh = (int**)malloc(sizeof(int*) * inum);
        for (int ii=0; ii<inum; ii++)
            firstneigh[ii] = (int*)malloc(sizeof(int) * numneigh[ii]);
        for (int ii=0; ii<inum; ii++) {
            for (int jj=0; jj<numneigh[ii]; jj++) 
                firstneigh[ii][jj] = neighbor_list.get_neighbor_lists()[ii][jj];
        }

        int supercell_num_atoms = neighbor_list.get_binLinkedList().get_supercell().get_num_atoms();
        x = (double**)neighbor_list.get_binLinkedList().get_supercell().get_structure().get_cart_coords();
        types = (int*)neighbor_list.get_binLinkedList().get_supercell().get_structure().get_atomic_numbers();
    }


    void TearDown() override {
        // Step . Free memory
        free(center_atomic_numbers_lst);
        free(neigh_atomic_numbers_lst);
        free(num_neigh_atoms_lst);

        free(ilist);
    }
};



TEST_F(TildeRTest, generate_for_lmp_1) {
    // Step 1. generate()
    double*** tilde_r = matersdk::deepPotSE::TildeR<double>::generate(
                                        inum,
                                        ilist,
                                        numneigh,
                                        firstneigh,
                                        x,
                                        types,
                                        num_center_atomic_numbers,
                                        center_atomic_numbers_lst,
                                        num_neigh_atomic_numbers,
                                        neigh_atomic_numbers_lst,
                                        num_neigh_atoms_lst,
                                        rcut,
                                        rcut_smooth);
    
    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        tot_num_neigh_atoms += num_neigh_atoms_lst[ii];

    /*
    // Step 2. print out
    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
            printf("[%4d, %4d] -- [%10f, %10f, %10f, %10f]\n",
                        ii, jj,
                        tilde_r[ii][jj][0],
                        tilde_r[ii][jj][1],
                        tilde_r[ii][jj][2],
                        tilde_r[ii][jj][3]);
        }
    }
     */

    // Step . Free memory
    matersdk::arrayUtils::free3dArray<double>(tilde_r, inum, tot_num_neigh_atoms);
}



TEST_F(TildeRTest, generate_for_lmp_2) {
    // Step 1. generate()
    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        tot_num_neigh_atoms += num_neigh_atoms_lst[ii];

    double*** tilde_r = matersdk::arrayUtils::allocate3dArray<double>(inum, tot_num_neigh_atoms, 4, true);

    matersdk::deepPotSE::TildeR<double>::generate(
                                tilde_r,
                                inum,
                                ilist,
                                numneigh,
                                firstneigh,
                                x,
                                types,
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut,
                                rcut_smooth);

    /*
    // Step 2. print out
    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
            printf("[%4d, %4d] -- [%10f, %10f, %10f, %10f]\n",
                        ii, jj,
                        tilde_r[ii][jj][0],
                        tilde_r[ii][jj][1],
                        tilde_r[ii][jj][2],
                        tilde_r[ii][jj][3]);
        }
    }
     */

    // Step . Free memory
    matersdk::arrayUtils::free3dArray<double>(tilde_r, inum, tot_num_neigh_atoms);
}


TEST_F(TildeRTest, deriv_for_lmp_1) {
    // Step 1. deriv()
    double**** tilde_r_deriv = matersdk::deepPotSE::TildeR<double>::deriv(
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        x,
                        types,
                        num_center_atomic_numbers,
                        center_atomic_numbers_lst,
                        num_neigh_atomic_numbers,
                        neigh_atomic_numbers_lst,
                        num_neigh_atoms_lst,
                        rcut,
                        rcut_smooth);
    
    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        tot_num_neigh_atoms += num_neigh_atoms_lst[ii];
    
    /*
    // Step 2. Print out
    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
            printf("[%4d, %4d] -- [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f]\n",
                        ii, jj,
                        tilde_r_deriv[ii][jj][0][0],
                        tilde_r_deriv[ii][jj][0][1],
                        tilde_r_deriv[ii][jj][0][2],
                        tilde_r_deriv[ii][jj][1][0],
                        tilde_r_deriv[ii][jj][1][1],
                        tilde_r_deriv[ii][jj][1][2],
                        tilde_r_deriv[ii][jj][2][0],
                        tilde_r_deriv[ii][jj][2][1],
                        tilde_r_deriv[ii][jj][2][2],
                        tilde_r_deriv[ii][jj][3][0],
                        tilde_r_deriv[ii][jj][3][1],
                        tilde_r_deriv[ii][jj][3][2]);
        }
    }
     */

    // Step . Free memory
    matersdk::arrayUtils::free4dArray(tilde_r_deriv, inum, tot_num_neigh_atoms, 4);
}



TEST_F(TildeRTest, deriv_for_lmp_2) {
    // Step 1. deriv()
    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        tot_num_neigh_atoms += num_neigh_atoms_lst[ii];


    double**** tilde_r_deriv = matersdk::arrayUtils::allocate4dArray<double>(inum, tot_num_neigh_atoms, 4, 3, true);
    matersdk::deepPotSE::TildeR<double>::deriv(
                        tilde_r_deriv,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        x,
                        types,
                        num_center_atomic_numbers,
                        center_atomic_numbers_lst,
                        num_neigh_atomic_numbers,
                        neigh_atomic_numbers_lst,
                        num_neigh_atoms_lst,
                        rcut,
                        rcut_smooth);

    
    // Step 2. Print out
    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
            printf("[%4d, %4d] -- [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f]\n",
                        ii, jj,
                        tilde_r_deriv[ii][jj][0][0],
                        tilde_r_deriv[ii][jj][0][1],
                        tilde_r_deriv[ii][jj][0][2],
                        tilde_r_deriv[ii][jj][1][0],
                        tilde_r_deriv[ii][jj][1][1],
                        tilde_r_deriv[ii][jj][1][2],
                        tilde_r_deriv[ii][jj][2][0],
                        tilde_r_deriv[ii][jj][2][1],
                        tilde_r_deriv[ii][jj][2][2],
                        tilde_r_deriv[ii][jj][3][0],
                        tilde_r_deriv[ii][jj][3][1],
                        tilde_r_deriv[ii][jj][3][2]);
        }
    }

    // Step . Free memory
    matersdk::arrayUtils::free4dArray(tilde_r_deriv, inum, tot_num_neigh_atoms, 4);
}



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}