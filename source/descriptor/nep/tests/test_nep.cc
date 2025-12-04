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
#include "../include/nep.h"


class NepTest : public ::testing::Test {
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

    double etot;
    double etot_;
    double (*forces)[3];
    double (*forces_)[3];
    double *virial;
    double *virial_;
    double rmax;
    double rmin;
    int chebyshev_size;
    int n_radial_basis;
    int n_angular_basis;
    int l_max;
    int num_descriptors;
    int num_neurons;
    double *coeffs;
    double *w0;
    double *w1;
    double *type_bias;


    static void SetUpTestSuite() {
        std::cout << "NepTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "NepTest (TestSuite) is tearing down...\n";
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
        nblist = ai2pot::NeighborList<double>(structure, rcut, pbc_xyz, false);
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


        etot = 0.0;
        etot_ = 0.0;
        forces = (double (*)[3])malloc(sizeof(double) * inum * 3);
        memset(forces, 0, sizeof(double) * inum * 3);
        forces_ = (double (*)[3])malloc(sizeof(double) * inum * 3);
        memset(forces_, 0, sizeof(double) * inum * 3);
        virial = (double*)malloc(sizeof(double) * 9);
        memset(virial, 0, sizeof(double) * 9);
        virial_ = (double*)malloc(sizeof(double) * 9);
        memset(virial_, 0, sizeof(double) * 9);

        chebyshev_size = 8;
        num_neurons = 30;
        n_radial_basis = 10;
        n_angular_basis = 8;
        l_max = 3;
        num_descriptors = ai2pot::nep::NepIndex::get_num_descriptors(n_radial_basis, n_angular_basis, l_max);

        rmax = 5.0;
        rmin = 0.0;

        coeffs = (double*)malloc(sizeof(double) * ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size);
        for (int ii=0; ii<ntypes * ntypes * (n_radial_basis+n_angular_basis) * chebyshev_size; ii++)
            coeffs[ii] = 0.01 + 0.001 * ii;
        w0 = (double*)malloc(sizeof(double) * ntypes * num_neurons * num_descriptors);
        for (int ii=0; ii<ntypes * num_neurons * num_descriptors; ii++)
            w0[ii] = 0.1 + 0.001 * ii;
        w1 = (double*)malloc(sizeof(double) * ntypes * num_neurons);
        for (int ii=0; ii<ntypes * num_neurons; ii++)
            w1[ii] = 0.1 + 0.001 * ii;
        type_bias = (double*)malloc(sizeof(double) * ntypes);
        type_bias[0] = -0.1;
        type_bias[1] = -0.2;
    }

    void TearDown() override {
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);
        free(forces);
        free(forces_);
        free(virial);
        free(virial_);
        free(coeffs);
        free(w0);
        free(w1);
        free(type_bias);

        // Loss
    }
};  // class : NepTest


TEST_F(NepTest, find_ef_accuracy) {
    int center_idx_modify = 1;
    int direction1_idx_modify = 2;
    double delta = 1e-7;

    ai2pot::nep::Nep<double>::find_ef(
        etot,
        forces,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        nullptr);

    // *** delta
    double cart_coords[12][3] = {0};
    for (int ii=0; ii<inum; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction1_idx_modify] += delta;
    structure = ai2pot::Structure<double>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);
    nblist = ai2pot::NeighborList<double>(structure, rcut, pbc_xyz, true);
    nblist.find_info4mlff(
            inum,
            ilist,
            numneigh,
            firstneigh,
            rcs,
            types,
            nghost,
            umax_num_neigh_atoms);
    // *** delta

    ai2pot::nep::Nep<double>::find_ef(
        etot_,
        forces_,
        chebyshev_size,
        n_radial_basis,
        n_angular_basis,
        l_max,
        num_neurons,
        coeffs,
        w0,
        w1,
        type_bias,
        inum,
        ilist,
        numneigh,
        firstneigh,
        (double (*)[3])rcs,
        types,
        ntypes,
        type_map,
        umax_num_neigh_atoms,
        nghost,
        rmax,
        rmin,
        nullptr);

printf("1.1. energy = %.15lf\n", etot);
printf("1.1. force[%d][%d] calculated by custom code = %.15lf\n", center_idx_modify, direction1_idx_modify, forces[center_idx_modify][direction1_idx_modify]);
printf("1.2. energy = %.15lf\n", etot_);
printf("1.2. force[%d][%d] calculated by finite difference method = %.15lf\n", center_idx_modify, direction1_idx_modify, -(etot_ - etot) / delta);
printf("2.1. energy = %.15lf\n", etot);
printf("2.2. force=\n");
for (int ii=0; ii<inum; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, forces[ii][0], forces[ii][1], forces[ii][2]);
}


int main(int argc, char **argv)  {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
