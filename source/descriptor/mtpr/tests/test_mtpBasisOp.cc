#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>

#include "../include/mtpParamOp.h"
#include "../include/mtpBasisOp.h"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"


class MtpBasisOpTest : public ::testing::Test
{
protected:
    c10::TensorOptions float_options;
    c10::TensorOptions int_options;
    at::Tensor binum_tensor;
    at::Tensor bilist_tensor;
    at::Tensor bnumneigh_tensor;
    at::Tensor bfirstneigh_tensor;
    at::Tensor brcs_tensor;
    at::Tensor btypes_tensor;
    int nghost;
    int umax_num_neighs;

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

    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> nblist;

    int mtp_level;
    int chebyshev_size;
    at::Tensor coeffs_tensor;
    double rmax;
    double rmin;
    at::Tensor alpha_index_basic_tensor;
    at::Tensor alpha_index_times_tensor;
    at::Tensor alpha_moment_mapping_tensor;
    at::Tensor num_mus4moms_tensor;
    at::Tensor mus4moms_tensor;
    at::Tensor nmus_tensor;

    double scaling;

    static void SetUpTestSuite() {
        std::cout << "MtpBasisOpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "MtpBasisOpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        mtp_level = 16;
        chebyshev_size = 8;
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

        structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        nblist = ai2pot::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, true);

        umax_num_neighs = 20;
        inum = 12;
        binum_tensor = at::tensor({inum}, int_options);
        bilist_tensor = at::zeros({1, inum}, int_options);
        bnumneigh_tensor = at::zeros({1, inum}, int_options);
        bfirstneigh_tensor = at::zeros({1, inum, umax_num_neighs}, int_options);
        brcs_tensor = at::zeros({1, inum, umax_num_neighs, 3}, float_options);
        btypes_tensor = at::zeros({1, inum}, int_options);
        ilist = bilist_tensor[0].data_ptr<int>();
        numneigh = bnumneigh_tensor[0].data_ptr<int>();
        firstneigh = bfirstneigh_tensor[0].data_ptr<int>();
        rcs = brcs_tensor[0].data_ptr<double>();
        types = btypes_tensor[0].data_ptr<int>();

        nblist.find_info4mlff(inum,
                              ilist,
                              numneigh,
                              firstneigh,
                              rcs,
                              types,
                              nghost,
                              umax_num_neighs);

        torch::autograd::variable_list mtp_param_info = ai2pot::mtpr::MtpParamOp(mtp_level);
        alpha_index_basic_tensor = mtp_param_info[0];
        alpha_index_times_tensor = mtp_param_info[1];
        alpha_moment_mapping_tensor = mtp_param_info[2];
        num_mus4moms_tensor = mtp_param_info[3];
        mus4moms_tensor = mtp_param_info[4];
        nmus_tensor = mtp_param_info[5];
        coeffs_tensor = at::ones({ntypes*ntypes*nmus_tensor.item<int>()*chebyshev_size}, float_options);
        rmax = 5.0;
        rmin = 2.0;

        scaling = 1.0;
    }

    void TearDown() override {
    }
};  // class : MtpBasisOpTest


TEST_F(MtpBasisOpTest, forward) 
{
    at::Tensor mtp_basis_tensor = ai2pot::mtpr::MtpBasisOp(alpha_index_basic_tensor,
                                                           alpha_index_times_tensor,
                                                           alpha_moment_mapping_tensor,
                                                           num_mus4moms_tensor,
                                                           mus4moms_tensor,
                                                           nmus_tensor,
                                                           ntypes,
                                                           chebyshev_size,
                                                           scaling,
                                                           coeffs_tensor,
                                                           binum_tensor,
                                                           bilist_tensor,
                                                           bnumneigh_tensor,
                                                           bfirstneigh_tensor,
                                                           brcs_tensor,
                                                           btypes_tensor,
                                                           umax_num_neighs,
                                                           rmax,
                                                           rmin)[0];
    
std::cout << "mtp_level = " << mtp_level << ": \n" << mtp_basis_tensor << std::endl;
}


TEST_F(MtpBasisOpTest, backward_der2xyz)
{
    int batch_modify = 5;
    int center_modify = 2;
    int neigh_modify = 12;
    int direction_modify = 0;
    double delta = 1e-3;

    brcs_tensor.requires_grad_(true);
    coeffs_tensor.requires_grad_(true);
    at::Tensor mtp_basis_tensor = ai2pot::mtpr::MtpBasisOp(alpha_index_basic_tensor,
                                                           alpha_index_times_tensor,
                                                           alpha_moment_mapping_tensor,
                                                           num_mus4moms_tensor,
                                                           mus4moms_tensor,
                                                           nmus_tensor,
                                                           ntypes,
                                                           chebyshev_size,
                                                           scaling,
                                                           coeffs_tensor,
                                                           binum_tensor,
                                                           bilist_tensor,
                                                           bnumneigh_tensor,
                                                           bfirstneigh_tensor,
                                                           brcs_tensor,
                                                           btypes_tensor,
                                                           umax_num_neighs,
                                                           rmax,
                                                           rmin)[0];
    at::Tensor result = mtp_basis_tensor.sum();
    result.backward();
    
    at::Tensor brcs_tensor_ = brcs_tensor.clone().detach();
    double *brcs_ = brcs_tensor_.data_ptr<double>();
    brcs_[center_modify*umax_num_neighs*3 + neigh_modify*3 + direction_modify] += delta;
    at::Tensor mtp_basis_tensor_ = ai2pot::mtpr::MtpBasisOp(alpha_index_basic_tensor,
                                                           alpha_index_times_tensor,
                                                           alpha_moment_mapping_tensor,
                                                           num_mus4moms_tensor,
                                                           mus4moms_tensor,
                                                           nmus_tensor,
                                                           ntypes,
                                                           chebyshev_size,
                                                           scaling,
                                                           coeffs_tensor,
                                                           binum_tensor,
                                                           bilist_tensor,
                                                           bnumneigh_tensor,
                                                           bfirstneigh_tensor,
                                                           brcs_tensor_,
                                                           btypes_tensor,
                                                           umax_num_neighs,
                                                           rmax,
                                                           rmin)[0];
    at::Tensor result_ = mtp_basis_tensor_.sum();
    result_.backward();
//std::cout << result << std::endl;
//std::cout << result_ << std::endl;
printf("1.1. Sum of descriptors detivative w.r.t rcs[%d][%d][%d] calculated by custom code:\n", center_modify, neigh_modify, direction_modify);
std::cout << brcs_tensor.grad()[0][center_modify][neigh_modify][direction_modify] << std::endl;
printf("1.2. Sum of descriptors detivative w.r.t rcs[%d][%d][%d] calculated by finite difference method:\n", center_modify, neigh_modify, direction_modify);
std::cout << (result_ - result) / delta << std::endl;
}


TEST_F(MtpBasisOpTest, backward_der2coeffs)
{
    int itype_modify = 1;
    int jtype_modify = 1;
    int mu_modify = 1;
    int xi_modify = 5;
    int coeff_idx_modify = (itype_modify*ntypes + jtype_modify)*nmus_tensor.item<int>()*chebyshev_size
                           + mu_modify*chebyshev_size
                           + xi_modify;
    double delta = 1E-4;

    brcs_tensor.requires_grad_(true);
    coeffs_tensor.requires_grad_(true);
    at::Tensor mtp_basis_tensor = ai2pot::mtpr::MtpBasisOp(alpha_index_basic_tensor,
                                                           alpha_index_times_tensor,
                                                           alpha_moment_mapping_tensor,
                                                           num_mus4moms_tensor,
                                                           mus4moms_tensor,
                                                           nmus_tensor,
                                                           ntypes,
                                                           chebyshev_size,
                                                           scaling,
                                                           coeffs_tensor,
                                                           binum_tensor,
                                                           bilist_tensor,
                                                           bnumneigh_tensor,
                                                           bfirstneigh_tensor,
                                                           brcs_tensor,
                                                           btypes_tensor,
                                                           umax_num_neighs,
                                                           rmax,
                                                           rmin)[0];

    at::Tensor coeffs_tensor_ = coeffs_tensor.clone().detach();
    double *coeffs_ = coeffs_tensor_.data_ptr<double>();
    coeffs_[coeff_idx_modify] += delta;
    at::Tensor mtp_basis_tensor_ = ai2pot::mtpr::MtpBasisOp(alpha_index_basic_tensor,
                                                           alpha_index_times_tensor,
                                                           alpha_moment_mapping_tensor,
                                                           num_mus4moms_tensor,
                                                           mus4moms_tensor,
                                                           nmus_tensor,
                                                           ntypes,
                                                           chebyshev_size,
                                                           scaling,
                                                           coeffs_tensor_,
                                                           binum_tensor,
                                                           bilist_tensor,
                                                           bnumneigh_tensor,
                                                           bfirstneigh_tensor,
                                                           brcs_tensor,
                                                           btypes_tensor,
                                                           umax_num_neighs,
                                                           rmax,
                                                           rmin)[0];

    at::Tensor result = mtp_basis_tensor.sum();
    result.backward();
printf("1.1. Sum of descriptors detivative w.r.t coeffs[%d][%d][%d][%d] calculated by custom code:\n", itype_modify, jtype_modify, mu_modify, xi_modify);
std::cout << coeffs_tensor.grad()[coeff_idx_modify] << std::endl;

    at::Tensor result_ = mtp_basis_tensor_.sum();
    result_.backward();
printf("1.2. Sum of descriptors detivative w.r.t coeffs[%d][%d][%d][%d] calculated by finite difference method:\n", itype_modify, jtype_modify, mu_modify, xi_modify);
std::cout << (result_ - result) / delta << std::endl;
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
