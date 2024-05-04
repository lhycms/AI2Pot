#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>
#include <stdio.h>

#include "../include/se4pw_op.h"
#include "../../../nblist/include/structure.h"



class Se4pwOpTest : public ::testing::Test {
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
    int batch_size;
    int sinum;          // 超胞中的总原子数
    int inum;           // 中心原子的数目
    int* ilist;         // 中心原子在 supercell 中的 index
    int* numneigh;      // 各个中心原子的近邻原子数目
    int* firstneigh;   // 近邻原子在 supercell 中的 index
    int* types;         // supercell 中所有原子的index，starts from 0
    double** x_2d;         // supercell 中所有原子的位置
    double* x;  // 将 x_2d -> (inum, tot_num_neigh_atoms, 3)
    int ntypes;
    int* num_neigh_atoms_lst;
    int tot_num_neigh_atoms;


    static void SetUpTestSuite() {
        std::cout << "Se4pwOpTest TestSuite is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "Se4pwOpTest TestSuite is tearing down...\n";
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
        neighbor_list = matersdk::NeighborList<double>(structure, rcut, pbc_xyz, false);
        
        // Variables to simulate the info of `LAMMPS_NS::LAMMPS*`
        batch_size = 1;
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

        sinum = neighbor_list.get_binLinkedList().get_supercell().get_structure().get_num_atoms();
        x_2d = (double**)neighbor_list.get_binLinkedList().get_supercell().get_structure().get_cart_coords();
        x = (double*)malloc(sizeof(double) * sinum * 3);
        memset(x, 0, sizeof(double) * sinum * 3);
        for (int ii=0; ii<sinum; ii++) {
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
        free(x);
    }
};



TEST_F(Se4pwOpTest, forward) {
    // Note: It doesn't matter whether input tensor is flatten or not.
    c10::TensorOptions int_tensor_options = c10::TensorOptions().dtype(torch::kInt32).device(c10::kCPU);
    c10::TensorOptions float_tensor_options = c10::TensorOptions().dtype(torch::kFloat64).device(c10::kCPU);
    
    at::Tensor ilist_tensor = torch::from_blob(ilist, {batch_size, inum}, int_tensor_options);
    at::Tensor numneigh_tensor = torch::from_blob(numneigh, {batch_size, inum}, int_tensor_options);
    at::Tensor firstneigh_tensor = torch::from_blob(firstneigh, {batch_size, inum, tot_num_neigh_atoms}, int_tensor_options);
    at::Tensor types_tensor = torch::from_blob(types, {batch_size, sinum}, int_tensor_options);
    at::Tensor num_neigh_atoms_lst_tensor = torch::from_blob(num_neigh_atoms_lst, {ntypes}, int_tensor_options);
    at::Tensor x_tensor = torch::from_blob(x, {batch_size, sinum, 3}, float_tensor_options);

    torch::autograd::variable_list outputs = matersdk::deepPotSE::Se4pwOp::forward(
            batch_size,
            inum,
            ilist_tensor,
            numneigh_tensor,
            firstneigh_tensor,
            x_tensor,
            types_tensor,
            ntypes,
            num_neigh_atoms_lst_tensor,
            rcut,
            rcut_smooth);
    at::Tensor tilde_r = outputs[0];
    at::Tensor tilde_r_deriv = outputs[1];
    at::Tensor relative_coords = outputs[2];
    std::cout << "1. tilde_r : " << tilde_r.sizes() << std::endl;
    std::cout << "2. tilde_r_deriv : " << tilde_r_deriv.sizes() << std::endl;
    std::cout << "3. relative_coords : " << relative_coords.sizes() << std::endl;
}


TEST_F(Se4pwOpTest, get_prim_indices_from_matersdk) {
    c10::TensorOptions int_tensor_options = c10::TensorOptions().dtype(torch::kInt32).device(c10::kCPU);

    at::Tensor ilist_tensor = torch::from_blob(ilist, {batch_size, inum}, int_tensor_options);
    at::Tensor numneigh_tensor = torch::from_blob(numneigh, {batch_size, inum}, int_tensor_options);
    at::Tensor firstneigh_tensor = torch::from_blob(firstneigh, {batch_size, inum, tot_num_neigh_atoms}, int_tensor_options);
    at::Tensor types_tensor = torch::from_blob(types, {batch_size, sinum}, int_tensor_options);
    at::Tensor num_neigh_atoms_lst_tensor = torch::from_blob(num_neigh_atoms_lst, {ntypes}, int_tensor_options);

    at::Tensor prim_indices = matersdk::deepPotSE::Se4pwOp::get_prim_indices_from_matersdk(
            batch_size, 
            inum, 
            ilist_tensor, 
            numneigh_tensor, 
            firstneigh_tensor, 
            types_tensor, 
            ntypes, 
            num_neigh_atoms_lst_tensor);
    
    std::cout << "1. prim_indices.sizes() : " << prim_indices.sizes() << std::endl;
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}