#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
//#include <torch/torch.h>

#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../include/envMatrixOp.h"


class EnvMatrixOpTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double rcut_smooth;
    bool pbc_xyz[3];
    int nghost;
    matersdk::Structure<double> structure;
    matersdk::NeighborList<double> neighbor_list;

    int inum;
    int* ilist;
    int* numneigh;
    int* firstneigh;
    double* relative_coords;
    int* types;
    int ntypes;
    int umax_num_neigh_atoms;
    int* umax_num_neigh_atoms_lst;
    
    int batch_size;

    static void SetUpTestSuite() {
        std::cout << "EnvMatrixOpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "EnvMatrixOpTest (TestSuite) is tearing down...\n";
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

        batch_size = 1;
    }

    void TearDown() override {
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(relative_coords);
        free(types);
        free(umax_num_neigh_atoms_lst);
    }
}; // class : EnvMatrixOpTest


TEST_F(EnvMatrixOpTest, test_EnvMatrixFunction) {
    neighbor_list.find_info4mlff(
        inum,
        ilist,
        numneigh,
        firstneigh,
        relative_coords,
        types,
        nghost,
        umax_num_neigh_atoms);
    c10::TensorOptions int_options = c10::TensorOptions().dtype(torch::kInt32).device(c10::kCPU);
    c10::TensorOptions float_options = c10::TensorOptions().dtype(torch::kFloat64).device(c10::kCPU);
    at::Tensor ilist_tensor = torch::from_blob(
        ilist, 
        {batch_size, inum}, 
        int_options);
    at::Tensor numneigh_tensor = torch::from_blob(
        numneigh, 
        {batch_size, inum}, 
        int_options);
    at::Tensor firstneigh_tensor = torch::from_blob(
        firstneigh, 
        {batch_size, inum, umax_num_neigh_atoms}, 
        int_options);
    at::Tensor relative_coords_tensor = torch::from_blob(
        relative_coords, 
        {batch_size, inum, umax_num_neigh_atoms, 3}, 
        float_options);
    at::Tensor types_tensor = torch::from_blob(
        types, 
        {batch_size, inum}, 
        int_options);
    at::Tensor umax_num_neigh_atoms_lst_tensor = torch::from_blob(
        umax_num_neigh_atoms_lst, 
        {2}, 
        int_options);
    relative_coords_tensor.requires_grad_(true);     // Note!!!
    auto result = matersdk::deepPotSE::EnvMatrixFunction::apply(
        ilist_tensor,
        numneigh_tensor,
        firstneigh_tensor,
        relative_coords_tensor,
        types_tensor,
        umax_num_neigh_atoms_lst_tensor,
        rcut,
        rcut_smooth)[0];
    result.sum().backward();
std::cout << relative_coords_tensor.grad() << std::endl;
    //std::cout << result[0] << std::endl;
    //std::cout << result[1] << std::endl;

    /*
    auto result1 = matersdk::deepPotSE::EnvMatrixOp(
        ilist_tensor,
        numneigh_tensor,
        firstneigh_tensor,
        relative_coords_tensor,
        types_tensor,
        umax_num_neigh_atoms_lst_tensor,
        rcut,
        rcut_smooth);
    */
//std::cout << result1[0][0][11][10][3].item<double>() << std::endl;
//std::cout << result1[0][0][11][15][1].item<double>() << std::endl;
//std::cout << result1[1] << std::endl;
}



int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}