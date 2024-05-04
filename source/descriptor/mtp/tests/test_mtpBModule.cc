#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include "../include/mtpBModule.h"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"


class MtpBModuleTest : public ::testing::Test
{
protected:
    int64_t max_level;
    int64_t ntypes;
    int64_t size;
    c10::TensorOptions int_options;
    c10::TensorOptions options;
    at::Tensor rcuts_tensor;

    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];

    int umax_num_neigh_atoms;
    int inum;
    int* ilist;
    int* numneigh;
    int* firstneigh;
    double* rcs;
    int* types;
    int nghost;

    matersdk::Structure<double> structure;
    matersdk::NeighborList<double> neighbor_list;

    at::Tensor ilist_tensor;
    at::Tensor firstneigh_tensor;
    at::Tensor types_tensor;
    at::Tensor rcs_tensor;

    static void SetUpTestSuite()
    {
        std::cout << "MtpBModuleTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite()
    {
        std::cout << "MtpBModuleTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        max_level = 8;
        ntypes = 2;
        size = 8;
        int_options = c10::TensorOptions()
            .dtype(torch::kInt64)
            .device(c10::kCPU);
        options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(c10::kCPU);
        rcuts_tensor = at::zeros({2}, options);
        rcuts_tensor[0] = 5.0;  // -> R_{max}
        rcuts_tensor[1] = 3.0;  // -> R_{min}

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

        rcut = rcuts_tensor[0].item<double>();
        bin_size_xyz[0] = 2.5;
        bin_size_xyz[1] = 2.5;
        bin_size_xyz[2] = 2.5;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;

        structure = matersdk::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = matersdk::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, true);

        umax_num_neigh_atoms = 19;
        inum = 12;
        ilist = (int*)malloc(sizeof(int) * inum);
        numneigh = (int*)malloc(sizeof(int) * inum);
        firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        rcs = (double*)malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
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

        // Key input for `module->forward` function
        ilist_tensor = at::zeros({1, inum}, int_options);
        firstneigh_tensor = at::zeros({1, inum, umax_num_neigh_atoms}, int_options);
        types_tensor = at::zeros({inum}, int_options);
        rcs_tensor = at::zeros({1, inum, umax_num_neigh_atoms, 3}, options);
        int64_t* ilist_ptr = ilist_tensor.data_ptr<int64_t>();
        int64_t* firstneigh_ptr = firstneigh_tensor.data_ptr<int64_t>();
        int64_t* types_ptr = types_tensor.data_ptr<int64_t>();
        double* rcs_ptr = rcs_tensor.data_ptr<double>();
        for (int ii=0; ii<inum; ii++)
            ilist_ptr[ii] = ilist[ii];
        for (int ii=0; ii<inum; ii++) {
            for (int jj=0; jj<umax_num_neigh_atoms; jj++) {
                firstneigh_ptr[ii*umax_num_neigh_atoms + jj] = firstneigh[ii*umax_num_neigh_atoms + jj];
                rcs_ptr[ii*umax_num_neigh_atoms*3 + jj*3 + 0] = rcs[ii*umax_num_neigh_atoms*3 + jj*3 + 0];
                rcs_ptr[ii*umax_num_neigh_atoms*3 + jj*3 + 1] = rcs[ii*umax_num_neigh_atoms*3 + jj*3 + 1];
                rcs_ptr[ii*umax_num_neigh_atoms*3 + jj*3 + 2] = rcs[ii*umax_num_neigh_atoms*3 + jj*3 + 2];
            }
        }
        for (int ii=0; ii<inum; ii++)
            types_ptr[ii] = types[ii];

        rcs_tensor.requires_grad_(true);
    }

    void TearDown() override {
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);
    }
};  // class : MtpBModuleTest


TEST_F(MtpBModuleTest, init)
{
    matersdk::mtp::MtpBModule mtp_b_module(
        max_level,
        ntypes,
        size,
        rcuts_tensor);
    mtp_b_module->to(torch::kFloat64);
//for (const auto& pair : mtp_b_module->named_parameters())
    //std::cout << pair.key() << " : " << pair.value() << std::endl;
    auto time1 = std::chrono::high_resolution_clock::now();
    auto mtp_b_tensor = mtp_b_module->forward(
        ilist_tensor, 
        firstneigh_tensor, 
        rcs_tensor,
        types_tensor);
    auto time2 = std::chrono::high_resolution_clock::now();
    auto time2_time1 = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1);
//std::cout << "Inference cost time : " << time2_time1.count() << " ms.\n";
std::cout << "mtp_b_tensor.sizes() = " << mtp_b_tensor.sizes() << std::endl;
    mtp_b_tensor.sum().backward();
std::cout << "mtp_b_tensor_grad.sizes() = " << rcs_tensor.grad().sizes() << std::endl;

    ASSERT_EQ(max_level, 8);
    ASSERT_EQ(mtp_b_tensor.sizes()[0], 1);
    ASSERT_EQ(mtp_b_tensor.sizes()[1], inum);
    ASSERT_EQ(mtp_b_tensor.sizes()[2], 9);
    ASSERT_EQ(rcs_tensor.grad().sizes()[0], 1);
    ASSERT_EQ(rcs_tensor.grad().sizes()[1], inum);
    ASSERT_EQ(rcs_tensor.grad().sizes()[2], umax_num_neigh_atoms);
    ASSERT_EQ(rcs_tensor.grad().sizes()[3], 3);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
