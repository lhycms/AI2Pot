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
#include <stdio.h>
#include <stdlib.h>

#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../include/nep_utilities.h"
#include "../include/nepBasis.h"


class MomsDodsValDerTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    ai2pot::Structure<double> structure;

    double rcut;
    bool pbc_xyz[3];
    ai2pot::NeighborList<double> nblist;
    int umax_num_neigh_atoms;
    int ntypes;
    int inum;
    int *ilist;
    int *numneigh;
    int *firstneigh;
    double *rcs;
    int *types;
    int type_map[2];
    int nghost;

    // NEP parameters
    double rmax_radial;
    double rmax_angular;
    int n_radial_basis;
    int n_angular_basis;
    int l_max;
    int chebyshev_size;
    int num_descriptors;
    int num_Sinlm;
    int num_coeffs;
    double *coeffs;

    double *mom_vals;
    double *dod_vals;
    double (*mom_ders)[3];
    double (*dod_ders)[3];
    double *mom_vals_;
    double *dod_vals_;
    double (*mom_ders_)[3];
    double (*dod_ders_)[3];

    static void SetUpTestSuite() {
        std::cout << "MomsDodsValDer (TestSuite) is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "MomsDodsValDer (TestSuite) is tearing down...\n";
    }


    void SetUp() override {
        // Structure and NeighborList
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
        structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);

        rcut = 5.0;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = true;
        nblist = ai2pot::NeighborList<double>(structure, rcut, pbc_xyz, true);
        inum = 12;
        ntypes = 2;
        umax_num_neigh_atoms = 20;
        nghost = 0;
        ilist = (int*)malloc(sizeof(int) * inum);
        numneigh = (int*)malloc(sizeof(int) * inum);
        firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        rcs = (double*)malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
        types = (int*)malloc(sizeof(int) * inum);
        type_map[0] = 42;
        type_map[1] = 16;
        nblist.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);

        // NEP parameters
        rmax_radial = 5.0;
        rmax_angular = 5.0;
        n_radial_basis = 6;
        n_angular_basis = 4;
        l_max = 3;
        chebyshev_size = 8;
        num_descriptors = ai2pot::nep::NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);
        num_Sinlm = ai2pot::nep::NepIndex::get_num_Sinlm(n_angular_basis, l_max);

        coeffs = (double*)malloc(sizeof(double) * ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size);
        for (int ii=0; ii<ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size; ii++)
            coeffs[ii] = 0.01 + 0.001 * ii;
        

        // mom & dod
        mom_vals = (double*)malloc(sizeof(double) * num_Sinlm);
        dod_vals = (double*)malloc(sizeof(double) * num_descriptors);
        mom_ders = (double (*)[3])malloc(sizeof(double) * num_Sinlm * umax_num_neigh_atoms * 3);
        dod_ders = (double (*)[3])malloc(sizeof(double) * n_radial_basis * umax_num_neigh_atoms * 3);
        mom_vals_ = (double*)malloc(sizeof(double) * num_Sinlm);
        dod_vals_ = (double*)malloc(sizeof(double) * num_descriptors);
        mom_ders_ = (double (*)[3])malloc(sizeof(double) * num_Sinlm * umax_num_neigh_atoms * 3);
        dod_ders_ = (double (*)[3])malloc(sizeof(double) * n_radial_basis * umax_num_neigh_atoms * 3);
    }


    void TearDown() override {        
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);

        free(coeffs);
        free(mom_vals);
        free(dod_vals);
        free(mom_ders);
        free(dod_ders);
        free(mom_vals_);
        free(dod_vals_);
        free(mom_ders_);
        free(dod_ders_);
    }
};  // class : MomsDodsValDerTest


TEST_F(MomsDodsValDerTest, find_val_der) {
    int center_idx_modify = 11;
    int neigh_idx_modify = 0;
    int direction_idx_modify = 2;
    double delta = 1E-7;

    ai2pot::nep::MomsDodsValDer<double>::find_val_der(
        mom_vals,
        dod_vals,
        mom_ders,
        dod_ders,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        coeffs,
        ilist[center_idx_modify],
        numneigh[center_idx_modify],
        &firstneigh[center_idx_modify*umax_num_neigh_atoms],
        (double (*)[3])(&rcs[center_idx_modify*umax_num_neigh_atoms*3]),
        types,
        ntypes,
        umax_num_neigh_atoms,
        rmax_radial,
        rmax_angular);
    
    rcs[center_idx_modify*umax_num_neigh_atoms*3 + neigh_idx_modify*3 + direction_idx_modify] += delta;
    ai2pot::nep::MomsDodsValDer<double>::find_val_der(
        mom_vals_,
        dod_vals_,
        mom_ders_,
        dod_ders_,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        coeffs,
        ilist[center_idx_modify],
        numneigh[center_idx_modify],
        &firstneigh[center_idx_modify*umax_num_neigh_atoms],
        (double (*)[3])(&rcs[center_idx_modify*umax_num_neigh_atoms*3]),
        types,
        ntypes,
        umax_num_neigh_atoms,
        rmax_radial,
        rmax_angular);

for (int i=0; i<num_descriptors; i++)
    printf("%.*lf, ", 15, dod_vals[i]);
printf("\n\n");
printf("1. Check the derivatives of MTP basis wrt. relative coordinates:\n");
printf("1.1. The derivatives of MTP basis wrt. relative coordinates calculated by custom code:\n\t[");
for (int ii=0; ii<num_descriptors; ii++)
    if (ii <= n_radial_basis)
        printf("%10lf, ", mom_ders[ii*umax_num_neigh_atoms + neigh_idx_modify][direction_idx_modify]);
printf("]\n");
printf("1.2. The derivatives of MTP basis wrt. relative coordinates calculated by finite difference method:\n\t[");
for (int ii=0; ii<num_descriptors; ii++)
    if (ii <= n_radial_basis)
        printf("%10lf, ", (mom_vals_[ii] - mom_vals[ii]) / delta);
printf("]\n");
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
