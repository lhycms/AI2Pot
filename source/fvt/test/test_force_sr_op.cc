#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>

#include "../include/force_sr_op.h"
#include "../../nblist/include/structure.h"
#include "../../nblist/include/neighborList.h"


class ForceSrOpTest : public ::testing::Test
{
protected:
    c10::TensorOptions float_options;
    c10::TensorOptions int_options;
    at::Tensor bilist_tensor;
    at::Tensor bnumneigh_tensor;
    at::Tensor bfirstnumneigh_tensor;
    at::Tensor brcs_tensor;
    at::Tensor btypes_tensor;
    int nghost;
    int umax_num_neighs;
    at::Tensor bdei_drij_tensor;

    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];

    int ntypes;
    int inum;
    int *ilist;
    int *numneigh;
    int *firstneigh;
    double *rcs;
    int *types;
    double *dei_drij;

    matersdk::Structure<double> structure;
    matersdk::NeighborList<double> nblist;

    static void SetUpTestSuite() {
        std::cout << "ForceSrOpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "ForceSrOpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        int_options = c10::TensorOptions()
            .dtype(torch::kInt32)
            .device(c10::kCPU);
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(c10::kCPU);
        inum = 12;
        ntypes = 2;
        rcut = 5.0;
        umax_num_neighs = 20;

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

        rcut = 5.0;
        bin_size_xyz[0] = 2.5;
        bin_size_xyz[1] = 2.5;
        bin_size_xyz[2] = 2.5;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;

        structure = matersdk::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        nblist = matersdk::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, true);

        umax_num_neighs = 20;
        inum = 12;
        bilist_tensor = at::zeros({1, inum}, int_options);
        bnumneigh_tensor = at::zeros({1, inum}, int_options);
        bfirstnumneigh_tensor = at::zeros({1, inum, umax_num_neighs}, int_options);
        brcs_tensor = at::zeros({1, inum, umax_num_neighs, 3}, float_options);
        btypes_tensor = at::zeros({1, inum}, int_options);
        bdei_drij_tensor = at::zeros({1, inum, umax_num_neighs, 3}, float_options);
        ilist = bilist_tensor.data_ptr<int>();
        numneigh = bnumneigh_tensor.data_ptr<int>();
        firstneigh = bfirstnumneigh_tensor.data_ptr<int>();
        rcs = brcs_tensor.data_ptr<double>();
        types = btypes_tensor.data_ptr<int>();
        dei_drij = bdei_drij_tensor.data_ptr<double>();

        nblist.find_info4mlff(inum,
                              ilist,
                              numneigh,
                              firstneigh,
                              rcs,
                              types,
                              nghost,
                              umax_num_neighs);
        for (int ii=0; ii<inum*umax_num_neighs*3; ii++)
            dei_drij[ii] = 1.0 + 0.001 * ii;
    }

    void TearDown() override {
    }
};  // class : ForceSrOpTest


TEST_F(ForceSrOpTest, forward_and_backward)
{
    int center_modify = 0;
    int neighbor_modify = 1;
    int direction_modify = 0;
    bdei_drij_tensor.requires_grad_(true);
    at::Tensor force_sr_tensor = matersdk::fvt::ForceSrOp(bilist_tensor,
                             bnumneigh_tensor,
                             bfirstnumneigh_tensor,
                             nghost,
                             umax_num_neighs,
                             bdei_drij_tensor)[0];
    force_sr_tensor.sum().backward();
    at::Tensor grad = bdei_drij_tensor.grad()[0][center_modify][neighbor_modify][direction_modify];

    dei_drij[center_modify*umax_num_neighs*3 + neighbor_modify*3 + direction_modify] += 0.001;
    at::Tensor force_sr_tensor_ = matersdk::fvt::ForceSrOp(bilist_tensor,
                             bnumneigh_tensor,
                             bfirstnumneigh_tensor,
                             nghost,
                             umax_num_neighs,
                             bdei_drij_tensor)[0];
    at::Tensor grad_ = (force_sr_tensor_.sum() - force_sr_tensor.sum()) / 0.001;
    std::cout << force_sr_tensor_.sum() << std::endl;
    std::cout << force_sr_tensor.sum() << std::endl;
    std::cout << grad << std::endl;
    std::cout << grad_ << std::endl;
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}