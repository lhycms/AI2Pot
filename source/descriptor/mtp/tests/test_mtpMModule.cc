#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include "../include/mtpMModule.h"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"


class MtpFModuleTest : public ::testing::Test
{
protected:
    int64_t nmus;
    int64_t ntypes;
    int64_t size;
    at::Tensor rcuts_tensor;    // [rcut, rcut_smooth]
    c10::TensorOptions options;
    c10::TensorOptions int_options;

    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];

    matersdk::Structure<double> structure;
    matersdk::NeighborList<double> neighbor_list;

    int umax_num_neigh_atoms;
    int inum;
    int* ilist;
    int* numneigh;
    int* firstneigh;
    double* relative_coords;
    int* types;
    int nghost;
 
    static void SetUpTestSuite() {
        std::cout << "MtpFModuleTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpFModuleTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        nmus = 2;
        ntypes = 2;
        size = 8;
        options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(c10::kCPU);
        int_options = c10::TensorOptions()
            .dtype(torch::kInt64)
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
        bin_size_xyz[0] = 1.65;
        bin_size_xyz[1] = 1.65;
        bin_size_xyz[2] = 1.65;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = false;

        structure = matersdk::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = matersdk::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, true);

        umax_num_neigh_atoms = 19;
        inum = 12;
        ilist = (int*)malloc(sizeof(int) * inum);
        numneigh = (int*)malloc(sizeof(int) * inum);
        firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        relative_coords = (double*)malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
        types = (int*)malloc(sizeof(int) * inum);
        // nghost
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
    }

};  // class : MtpFModuleTest


TEST_F(MtpFModuleTest, init) {
    matersdk::mtp::MtpMModule mtp_f(
        nmus,
        ntypes,
        size,
        rcuts_tensor);
    mtp_f->to(torch::kFloat64); // Note it!
    for (const auto& pair : mtp_f->named_parameters()) {
        //std::cout << pair.key() << " :\n" << pair.value() << std::endl;
        ASSERT_EQ(pair.value().sizes()[0], size);
    }

    int64_t mu = 0;
    int64_t nu = 3;
    int64_t iidx = 0;
    at::Tensor ifirstneigh_tensor = at::zeros({umax_num_neigh_atoms}, int_options);
    at::Tensor types_tensor = at::zeros({inum}, int_options);
    at::Tensor ircs_tensor = at::zeros({umax_num_neigh_atoms, 3}, options);
    int64_t* ifirstneigh = ifirstneigh_tensor.data_ptr<int64_t>();
    int64_t* types_ = types_tensor.data_ptr<int64_t>();
    double* ircs = ircs_tensor.data_ptr<double>();
    for (int ii=0; ii<umax_num_neigh_atoms; ii++)
        ifirstneigh[ii] = (int64_t)firstneigh[iidx*umax_num_neigh_atoms+ii];
    for (int ii=0; ii<inum; ii++)
        types_[ii] = (int64_t)types[ii];
    for (int ii=0; ii<umax_num_neigh_atoms; ii++) {
        ircs[ii*3 + 0] = relative_coords[iidx*umax_num_neigh_atoms*3 + ii*3 + 0];
        ircs[ii*3 + 1] = relative_coords[iidx*umax_num_neigh_atoms*3 + ii*3 + 1];
        ircs[ii*3 + 2] = relative_coords[iidx*umax_num_neigh_atoms*3 + ii*3 + 2];
    }
    ircs_tensor.requires_grad_(true);   // need to calculate gradients.
    at::Tensor imtp_m_tensor = mtp_f->forward(
        mu,
        nu,
        iidx,
        ifirstneigh_tensor,
        types_tensor,
        ircs_tensor);
//std::cout << imtp_m_tensor << std::endl;
std::cout << "1.1. When mu = " << mu << ", nu = " << nu << ", MtpQModule->forward():\n" << imtp_m_tensor << std::endl;
    imtp_m_tensor.sum().backward();
std::cout << "1.2. MtpQModule.sum() 's derivative wrt. xyz:\n" << ircs_tensor.grad().sizes() << std::endl;
    ASSERT_EQ(ircs_tensor.grad().sizes()[0], umax_num_neigh_atoms);
    ASSERT_EQ(ircs_tensor.grad().sizes()[1], 3);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
