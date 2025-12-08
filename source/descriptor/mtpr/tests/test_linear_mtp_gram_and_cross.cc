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
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>

#include "../include/mtpParam.h"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../include/linear_mtp_gram_and_cross.h"


typedef double real;

class LinearMtpGramAndCrossTest : public ::testing::Test {
protected:
    int natoms_pad;
    int inum;
    int num_parameters;

    int chebyshev_size;
    real *coeffs;
    real *linear_coeffs;
    real *type_bias;
    int nmus;
    real rmax;
    real rmin;
    int ntypes;

    std::vector<std::string> filenames;
    ai2pot::mtpr::MtpParam mtp_param;

    int num_atoms;
    real basis_vectors[3][3];
    int atomic_numbers[12];
    real frac_coords[12][3];
    real rcut;
    real bin_size_xyz[3];
    bool pbc_xyz[3];
    int type_map[2];

    int umax_num_neigh_atoms;
    int* ilist;
    int* numneigh;
    int* firstneigh;
    real* rcs;
    int* types;
    int nghost;

    ai2pot::Structure<real> structure;
    ai2pot::NeighborList<real> neighbor_list;

    real *energy_components;
    real *force_components;
    real *virial_components;


    static void SetUpTestSuite() {
        std::cout << "LinearMtpGramAndCrossTest (TestSuite) is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "LinearMtpGramAndCrossTest (TestSuite) is tearing down...\n";
    }


    void SetUp() override {
        natoms_pad = 12;
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
        mtp_param._load(filenames[7]);

        inum = 12;
        ntypes = 2;
        chebyshev_size = 8;
        nmus = mtp_param.nmus();
        rmax = 5.0;
        rmin = 2.0;
        umax_num_neigh_atoms = 20;
        coeffs = (real*)malloc(sizeof(real) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);

        for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
            coeffs[ii] = 0.01 + 0.001 * ii;
    
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

        linear_coeffs = (real*)malloc(sizeof(real) * mtp_param.alpha_scalar_moments());
        type_bias = (real*)malloc(sizeof(real) * ntypes);
        for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
            linear_coeffs[ii] = 0.01 + 0.001 * ii;
        type_bias[0] = -0.1;
        type_bias[1] = -0.2;


        num_parameters = mtp_param.alpha_scalar_moments() + ntypes;
        energy_components = (real*)malloc(sizeof(real) * num_parameters);
        force_components = (real*)malloc(sizeof(real) * natoms_pad * 3 * num_parameters);
        virial_components = (real*)malloc(sizeof(real) * 3 * 3 * num_parameters);
        memset(energy_components, 0, sizeof(real) * num_parameters);
        memset(force_components, 0, sizeof(real) * natoms_pad * 3 * num_parameters);
        memset(virial_components, 0, sizeof(real) * 3 * 3 * num_parameters);
    }


    void TearDown() override {
        free(coeffs);
        free(linear_coeffs);
        free(type_bias);

        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);

        free(energy_components);
        free(force_components);
        free(virial_components);
    }
};  // class : LinearMtpGramAndCrossTest



TEST_F(LinearMtpGramAndCrossTest, accumulate_efv_components) {
    ai2pot::mtpr::LinearMtpGramAndCross<real>::accumulate_structure_efv_components(
        energy_components,
        force_components,
        virial_components,
        chebyshev_size,
        coeffs,
        linear_coeffs,
        type_bias,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        mtp_param.alpha_index_basic(),
        mtp_param.alpha_index_times_count(),
        mtp_param.alpha_index_times(),
        mtp_param.alpha_scalar_moments(),
        mtp_param.alpha_moment_mapping(),
        mtp_param.nmus(),
        natoms_pad,
        inum,
        ilist,
        numneigh,
        firstneigh,
        (real (*)[3])rcs,
        types,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin);
    
    for (int k=0; k<num_parameters; k++)
        printf("%.10lf, ", energy_components[k]);
    printf("\n");
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}