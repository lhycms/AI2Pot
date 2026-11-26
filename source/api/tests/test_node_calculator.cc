#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include <stdio.h>
#include <iostream>
#include <string>

#include "../include/node_calculator.h"
#include "../../nblist/include/structure.h"
#include "../../nblist/include/neighborList.h"


typedef float real;


class AI2PotNodeCalculatorTest : public ::testing::Test {
protected:
    std::string pt_path;
    torch::Device device = torch::kCPU;

    int num_atoms;
    real basis_vectors[3][3];
    int atomic_numbers[12];
    real frac_coords[12][3];
    real rcut;
    real bin_size_xyz[3];
    bool pbc_xyz[3];
    int type_map[2];

    int umax_num_neigh_atoms;
    int inum;
    int *ilist;
    int *numneigh;
    int *firstneigh;
    real *rcs;
    int *types;
    int nghost;

    at::Tensor binum_tensor;
    at::Tensor bilist_tensor;
    at::Tensor bnumneigh_tensor;
    at::Tensor bfirstneigh_tensor;
    at::Tensor brcs_tensor;
    at::Tensor btypes_tensor;
    at::Tensor bnghost_tensor;

    c10::TensorOptions int_options;
    c10::TensorOptions float_options;

    ai2pot::Structure<real> structure;
    ai2pot::NeighborList<real> neighbor_list;

    std::string maxvol_matrix_inv_path;

    static void SetUpTestSuite() {
        std::cout << "AI2PotNodeCalculatorTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "AI2PotNodeCalculatorTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        pt_path = "/data/home/liuhanyu/mycode/AI2Pot/test/test_data/lmp_api/ai2pot_libtorch.pt";
        maxvol_matrix_inv_path = "/data/home/liuhanyu/mycode/AI2Pot/test/test_data/lmp_api/maxvol_matrix_inv.pt";

        // Establish neighbor list
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

        // 42: 0;  16: 1
        type_map[0] = 42;
        type_map[1] = 16;
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

        rcut = 5.0;
        bin_size_xyz[0] = 2.5;
        bin_size_xyz[1] = 2.5;
        bin_size_xyz[2] = 2.5;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;

        structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);

        inum = 12;
        umax_num_neigh_atoms = 200;
        ilist = (int*)malloc(sizeof(int) * inum);
        numneigh = (int*)malloc(sizeof(int) * inum);
        firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        rcs = (real*)malloc(sizeof(real) * inum * umax_num_neigh_atoms * 3);
        types = (int*)malloc(sizeof(int) * inum);
        neighbor_list.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);
        
        int_options = c10::TensorOptions().dtype(torch::kInt32).device(device);
        float_options = c10::TensorOptions().dtype(torch::kFloat32).device(device);
        binum_tensor = torch::zeros({1}, int_options);
        bilist_tensor = torch::zeros({1, inum}, int_options);
        bnumneigh_tensor = torch::zeros({1, inum}, int_options);
        bfirstneigh_tensor = torch::zeros({1, inum, umax_num_neigh_atoms}, int_options);
        brcs_tensor = torch::zeros({1, inum, umax_num_neigh_atoms, 3}, float_options);
        btypes_tensor = torch::zeros({1, inum}, int_options);
        bnghost_tensor = torch::zeros({1}, int_options);

        int *binum_ptr = binum_tensor.data_ptr<int>();
        int *bilist_ptr = bilist_tensor.data_ptr<int>();
        int *bnumneigh_ptr = bnumneigh_tensor.data_ptr<int>();
        int *bfirstneigh_ptr = bfirstneigh_tensor.data_ptr<int>();
        float *brcs_ptr = brcs_tensor.data_ptr<float>();
        int *btypes_ptr = btypes_tensor.data_ptr<int>();
        int *bnghost_ptr = bnghost_tensor.data_ptr<int>();

        binum_ptr[0] = inum;
        for (int ii=0; ii<inum; ii++) {
            bilist_ptr[ii] = ilist[ii];
            bnumneigh_ptr[ii] = numneigh[ii];
            btypes_ptr[ii] = types[ii];

            for (int jj=0; jj<umax_num_neigh_atoms; jj++) {
                bfirstneigh_ptr[ii*umax_num_neigh_atoms + jj] = firstneigh[ii*umax_num_neigh_atoms + jj];
                brcs_ptr[ii*umax_num_neigh_atoms*3 + jj*3 + 0] = rcs[ii*umax_num_neigh_atoms*3 + jj*3 + 0];
                brcs_ptr[ii*umax_num_neigh_atoms*3 + jj*3 + 1] = rcs[ii*umax_num_neigh_atoms*3 + jj*3 + 1];
                brcs_ptr[ii*umax_num_neigh_atoms*3 + jj*3 + 2] = rcs[ii*umax_num_neigh_atoms*3 + jj*3 + 2];
            }
        }
    }

    void TearDown() override {
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);
    }
};  // class : AI2PotNodeCalculatortest


TEST_F(AI2PotNodeCalculatorTest, constructor)
{
    AI2PotNodeCalculator node_calculator = AI2PotNodeCalculator(pt_path, device);
    std::cout << node_calculator.potential_type << std::endl;
}


TEST_F(AI2PotNodeCalculatorTest, predict_ef)
{
    AI2PotNodeCalculator node_calculator = AI2PotNodeCalculator(pt_path, device);
    torch::autograd::variable_list output = node_calculator.predict_ef(
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        bnghost_tensor);
    
    at::Tensor betot_tensor = output[0];
    at::Tensor bforce_tensor = output[1];

    std::cout << betot_tensor << std::endl;
}


TEST_F(AI2PotNodeCalculatorTest, predict_efv)
{
    AI2PotNodeCalculator node_calculator = AI2PotNodeCalculator(pt_path, device);
    torch::autograd::variable_list output = node_calculator.predict_efv(
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        bnghost_tensor);
    
    at::Tensor betot_tensor = output[0];
    at::Tensor bforce_tensor = output[1];
    at::Tensor bvirial_tensor = output[2];

    std::cout << betot_tensor << std::endl;
}


TEST_F(AI2PotNodeCalculatorTest, load_maxvol_matrix_inv)
{
    AI2PotNodeCalculator node_calculator = AI2PotNodeCalculator(pt_path, device);
    at::Tensor maxvol_matrix_inv_tensor = node_calculator.load_matrix_matrix_inv(maxvol_matrix_inv_path);
    std::cout << maxvol_matrix_inv_tensor.sizes() << std::endl;
}



TEST_F(AI2PotNodeCalculatorTest, predict_gamma_global)
{
    AI2PotNodeCalculator node_calculator = AI2PotNodeCalculator(pt_path, device);
    at::Tensor maxvol_matrix_inv_tensor = node_calculator.load_matrix_matrix_inv(maxvol_matrix_inv_path);

    torch::autograd::variable_list output = node_calculator.predict_gamma_gloal(
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        bnghost_tensor,
        maxvol_matrix_inv_tensor);
    
    std::cout << output[0] << std::endl;
}


TEST_F(AI2PotNodeCalculatorTest, predict_gamma_local)
{
    AI2PotNodeCalculator node_calculator = AI2PotNodeCalculator(pt_path, device);
    at::Tensor maxvol_matrix_inv_tensor = node_calculator.load_matrix_matrix_inv(maxvol_matrix_inv_path);

    torch::autograd::variable_list output = node_calculator.predict_gamma_local(
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        bnghost_tensor,
        maxvol_matrix_inv_tensor);
    
    std::cout << output[0] << std::endl;    
    std::cout << output[1] << std::endl; 
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
