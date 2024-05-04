#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <cmath>

#include "../../../nblist/include/structure.h"
#include "../include/se4pw.h"


class Se4pwTest : public ::testing::Test {
public:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double rcut_smooth;
    bool pbc_xyz[3];

    matersdk::Structure<double> structure;
    matersdk::NeighborList<double> neighbor_list;

    // Variables to simulate info of `LAMMPS_NS::LAMMPS* lmp`
    int inum;           // 中心原子的数目
    int* ilist;         // 中心原子在 supercell 中的 index
    int* numneigh;      // 各个中心原子的近邻原子数目
    int* firstneigh;   // 近邻原子在 supercell 中的 index
    int* types;         // supercell 中所有原子的元素种类
    double** x_2d;         // supercell 中所有原子的位置
    double* x;  // 将 x_2d -> (inum, tot_num_neigh_atoms, 3)
    int ntypes;
    int* num_neigh_atoms_lst;
    int tot_num_neigh_atoms;

    static void SetUpTestSuite() {
        std::cout << "Se4pwTest TestSuite is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "Se4pwTest TestSuite is tearing down...\n";
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


        int supercell_num_atoms = neighbor_list.get_binLinkedList().get_supercell().get_num_atoms();
        types = (int*)neighbor_list.get_binLinkedList().get_supercell().get_structure().get_atomic_numbers();
        ntypes = 2;
        num_neigh_atoms_lst = (int*)malloc(sizeof(int) * ntypes);
        num_neigh_atoms_lst[0] = 12;
        num_neigh_atoms_lst[1] = 8;
        tot_num_neigh_atoms = 0;
        for (int ii=0; ii<ntypes; ii++)
            tot_num_neigh_atoms += num_neigh_atoms_lst[ii];

        firstneigh = (int*)malloc(sizeof(int) * inum * tot_num_neigh_atoms);
        memset(firstneigh, -1, sizeof(int) * inum * tot_num_neigh_atoms);
        for (int ii=0; ii<inum; ii++) {
            for (int jj=0; jj<tot_num_neigh_atoms; jj++) {  
                if (jj < numneigh[ii])
                    firstneigh[ii*tot_num_neigh_atoms+jj] = neighbor_list.get_neighbor_lists()[ii][jj];    
                //printf("%4d, ", firstneigh[ii*tot_num_neigh_atoms+jj]);
            }
            //printf("\n");
        }

        x_2d = (double**)neighbor_list.get_binLinkedList().get_supercell().get_structure().get_cart_coords();
        x = (double*)malloc(sizeof(double) * neighbor_list.get_binLinkedList().get_supercell().get_structure().get_num_atoms() * 3);
        memset(x, 0, sizeof(double) * neighbor_list.get_binLinkedList().get_supercell().get_structure().get_num_atoms() * 3);
        for (int ii=0; ii<neighbor_list.get_binLinkedList().get_supercell().get_structure().get_num_atoms(); ii++) {
            x[ii*3 + 0] = x_2d[ii][0];
            x[ii*3 + 1] = x_2d[ii][1];
            x[ii*3 + 2] = x_2d[ii][2];
        }
    }

    void TearDown() override {
        // Step . Free memory
        free(ilist);
        free(numneigh);
        free(firstneigh);
    }
};


TEST_F(Se4pwTest, generate) {
    double* tilde_r = (double*)malloc(sizeof(double) * (inum*tot_num_neigh_atoms*4));
    double* tilde_r_deriv = (double*)malloc(sizeof(double) * (inum*tot_num_neigh_atoms*4*3));
    double* relative_coords = (double*)malloc(sizeof(double) * (inum*tot_num_neigh_atoms*3));

    matersdk::deepPotSE::Se4pw<double>::generate(
                        tilde_r,
                        tilde_r_deriv,
                        relative_coords,
                        inum,
                        ilist,
                        numneigh,
                        firstneigh,
                        x,
                        types,
                        ntypes,
                        num_neigh_atoms_lst,
                        rcut,
                        rcut_smooth);
    
    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
            printf("[%3d, %3d] -- [%10f, %10f, %10f, %10f]\n", 
                ii, jj,
                tilde_r[ii*tot_num_neigh_atoms*4+jj*4],
                tilde_r[ii*tot_num_neigh_atoms*4+jj*4+1],
                tilde_r[ii*tot_num_neigh_atoms*4+jj*4+2],
                tilde_r[ii*tot_num_neigh_atoms*4+jj*4+3]);
        }
    }

    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
            printf("[%3d, %3d] -- [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f]\n",
                    ii, jj,
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 0*3 + 0],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 0*3 + 1],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 0*3 + 2],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 1*3 + 0],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 1*3 + 1],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 1*3 + 2],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 2*3 + 0],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 2*3 + 1],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 2*3 + 2],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 3*3 + 0],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 3*3 + 1],
                    tilde_r_deriv[ii*tot_num_neigh_atoms*4*3 + jj*4*3 + 3*3 + 2]);
        }
    }
    
    
    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
            printf("[%3d, %3d] -- [%10f, %10f, %10f] : %10f\n",
                    ii, jj,
                    relative_coords[ii*tot_num_neigh_atoms*3 + jj*3 + 0],
                    relative_coords[ii*tot_num_neigh_atoms*3 + jj*3 + 1],
                    relative_coords[ii*tot_num_neigh_atoms*3 + jj*3 + 2],
                    std::sqrt(
                        std::pow(relative_coords[ii*tot_num_neigh_atoms*3 + jj*3 + 0], 2) + 
                        std::pow(relative_coords[ii*tot_num_neigh_atoms*3 + jj*3 + 1], 2) + 
                        std::pow(relative_coords[ii*tot_num_neigh_atoms*3 + jj*3 + 2], 2))
            );
        }
    }
}



TEST_F(Se4pwTest, get_prim_indices_from_matersdk) {
    int* prim_indices = (int*)malloc(sizeof(int) * inum * tot_num_neigh_atoms);

    matersdk::deepPotSE::Se4pw<double>::get_prim_indices_from_matersdk(
            prim_indices,
            inum,
            ilist,
            numneigh,
            firstneigh,
            types,
            ntypes,
            num_neigh_atoms_lst);

    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
            printf("%4d, ", prim_indices[ii*tot_num_neigh_atoms+jj]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}