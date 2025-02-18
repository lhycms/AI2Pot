/*
    Copyright 2025 Hanyu Liu
    This file is part of AI2Pot.
    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>

#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../include/mtpParam.h"
#include "../include/linearMtpOp.h"


class LinearMtpOpTest : public ::testing::Test
{
protected:
    std::vector<std::string> filenames;
    ai2pot::mtpr::MtpParam mtp_param;
    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> nblist;

    c10::TensorOptions float_options;
    c10::TensorOptions int_options;
    double e_weight;
    double f_weight;
    double v_weight;
    at::Tensor betot_dft_tensor;
    at::Tensor bforce_dft_tensor;
    at::Tensor bvirial_dft_tensor;
    int chebyshev_size;
    int nmus;
    int alpha_moments_count;

    at::Tensor coeffs_tensor;
    at::Tensor linear_coeffs_tensor;
    at::Tensor type_bias_tensor;
    
    at::Tensor alpha_index_basic_tensor;
    at::Tensor alpha_index_times_tensor;
    at::Tensor alpha_moment_mapping_tensor;

    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];

    int inum;
    at::Tensor binum_tensor;
    at::Tensor bilist_tensor;
    at::Tensor bnumneigh_tensor;
    at::Tensor bfirstneigh_tensor;
    at::Tensor brcs_tensor;
    at::Tensor btypes_tensor;
    int ntypes;
    int nghost;
    int umax_num_neigh_atoms;
    double rmax;
    double rmin;

    static void SetUpTestSuite() {
        std::cout << "LinearMtpOpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "LinearMtpOpTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        filenames = {
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/depreciated-02.almtp", 
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/depreciated-04.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/06.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/08.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/10.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/12.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/14.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/16.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/18.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/20.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/22.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/24.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/26.almtp",
            (std::string)std::getenv("AI2POT_PATH") + "/source/descriptor/mtpr/MTP_templates/28.almtp"
        };
        mtp_param._load(filenames[5]);
//mtp_param.show();

        float_options = c10::TensorOptions()
                            .dtype(torch::kFloat64)
                            .device(c10::kCPU);
        int_options = c10::TensorOptions()
                        .dtype(torch::kInt32)
                        .device(c10::kCPU);

        alpha_index_basic_tensor = at::zeros({mtp_param.alpha_index_basic_count(), 4}, int_options);
        alpha_index_times_tensor = at::zeros({mtp_param.alpha_index_times_count(), 4}, int_options);
        alpha_moment_mapping_tensor = at::zeros({mtp_param.alpha_scalar_moments()}, int_options);
        int (*alpha_index_basic)[4] = (int (*)[4])alpha_index_basic_tensor.data_ptr<int>();
        int (*alpha_index_times)[4] = (int (*)[4])alpha_index_times_tensor.data_ptr<int>();
        int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
        for (int i=0; i<mtp_param.alpha_index_basic_count(); i++)
            for (int aa=0; aa<4; aa++)
                alpha_index_basic[i][aa] = mtp_param.alpha_index_basic()[i][aa];
        for (int i=0; i<mtp_param.alpha_index_times_count(); i++)
            for (int aa=0; aa<4; aa++)
                alpha_index_times[i][aa] = mtp_param.alpha_index_times()[i][aa];
        for (int i=0; i<mtp_param.alpha_scalar_moments(); i++)
            alpha_moment_mapping[i] = mtp_param.alpha_moment_mapping()[i];

        rmax = 5.0;
        rmin = 2.0;

        // Establish neighbor list
        ntypes = 2;
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
        nblist = ai2pot::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, false);
        umax_num_neigh_atoms = 19;

        alpha_moments_count = mtp_param.alpha_moments_count();
        int alpha_index_basic_count = mtp_param.alpha_index_basic_count();
        int alpha_index_times_count = mtp_param.alpha_index_times_count();
        int alpha_scalar_moments = mtp_param.alpha_scalar_moments();
        nmus = mtp_param.nmus();
        chebyshev_size = 8;
        int num_coeffs = ntypes * ntypes * nmus * chebyshev_size;

        e_weight = 1.0;
        f_weight = 1.0;
        v_weight = 0.1;
        betot_dft_tensor = at::zeros({1}, float_options);
        bforce_dft_tensor = at::zeros({1, num_atoms, 3}, float_options);
        bvirial_dft_tensor = at::zeros({1, 9}, float_options);
        coeffs_tensor = at::zeros({num_coeffs}, float_options);
        coeffs_tensor.requires_grad_(true);
        linear_coeffs_tensor = at::zeros({alpha_scalar_moments}, float_options);
        linear_coeffs_tensor.requires_grad_(true);
        type_bias_tensor = at::zeros({ntypes}, float_options);
        type_bias_tensor.requires_grad_(true);
        binum_tensor = at::zeros({1}, int_options);
        bilist_tensor = at::zeros({1, num_atoms}, int_options);
        bnumneigh_tensor = at::zeros({1, num_atoms}, int_options);
        bfirstneigh_tensor = at::zeros({1, num_atoms, umax_num_neigh_atoms}, int_options);
        brcs_tensor = at::zeros({1, num_atoms, umax_num_neigh_atoms, 3}, float_options);
        btypes_tensor = at::zeros({1, num_atoms}, int_options);
        
        double *coeffs = coeffs_tensor.data_ptr<double>();
        double *linear_coeffs = linear_coeffs_tensor.data_ptr<double>();
        double *type_bias = type_bias_tensor.data_ptr<double>();

        for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
            coeffs[ii] = 0.1;
        for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
            linear_coeffs[ii] = 0.1 + ii * 0.01;
        type_bias[0] = -7;
        type_bias[1] = -8;

        binum_tensor[0] = num_atoms;
        int *ilist = bilist_tensor[0].data_ptr<int>();
        int *numneigh = bnumneigh_tensor[0].data_ptr<int>();
        int *firstneigh = bfirstneigh_tensor[0].data_ptr<int>();
        double *rcs = brcs_tensor[0].data_ptr<double>();
        int *types = btypes_tensor[0].data_ptr<int>();
        nblist.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);
//nblist.show_in_distances();
    }

    void TearDown() override {

    }
};


TEST_F(LinearMtpOpTest, apply) {
    at::Tensor loss_tensor = ai2pot::mtpr::LinearMtpToLossOp(
        e_weight,
        f_weight,
        v_weight,
        betot_dft_tensor,
        bforce_dft_tensor,
        bvirial_dft_tensor,
        chebyshev_size,
        coeffs_tensor,
        linear_coeffs_tensor,
        type_bias_tensor,
        alpha_moments_count,
        alpha_index_basic_tensor,
        alpha_index_times_tensor,
        alpha_moment_mapping_tensor,
        nmus,
        binum_tensor,
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        brcs_tensor,
        btypes_tensor,
        ntypes,
        nghost,
        rmax,
        rmin)[0];
std::cout << "loss_tensor:\n" << loss_tensor << std::endl;
    loss_tensor.backward();
std::cout << coeffs_tensor.grad() << std::endl;
std::cout << linear_coeffs_tensor.grad() << std::endl;
std::cout << type_bias_tensor.grad() << std::endl;
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
