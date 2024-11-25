#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#include "../include/virial_sr.cuh"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"


class VirialSrTest : public ::testing::Test
{
protected:
    double *h_virial_sr_val;
    double *h_virial_sr_val_;
    double *h_virial_sr_der1;
    double *h_virial_sr_der1_;
    double *h_virial_sr_der2;
    double *h_virial_sr_der2_;
    
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];

    int ntypes;
    int umax_num_neighs;
    int inum;
    int *h_ilist;
    int *h_numneigh;
    int *h_firstneigh;
    double *h_rcs;
    int *h_types;
    int nghost;

    double *h_dei_drij;

    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> nblist;

    static void SetUpTestSuite() {
        std::cout << "VirialSrTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "VirialSrTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
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
        h_ilist = (int*)malloc(sizeof(int) * inum);
        h_numneigh = (int*)malloc(sizeof(int) * inum);
        h_firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neighs);
        h_rcs = (double*)malloc(sizeof(double) * inum * umax_num_neighs * 3);
        h_types = (int*)malloc(sizeof(int) * (inum+nghost));
        nblist.find_info4mlff(inum,
                              h_ilist,
                              h_numneigh,
                              h_firstneigh,
                              h_rcs,
                              h_types,
                              nghost,
                              umax_num_neighs);
        
        h_virial_sr_val = (double*)malloc(sizeof(double) * 3 * 3);
        h_virial_sr_val_ = (double*)malloc(sizeof(double) * 3 * 3);
        h_virial_sr_der1 = (double*)malloc(sizeof(double) * 3 * 3 * inum * umax_num_neighs);
        h_virial_sr_der1_ = (double*)malloc(sizeof(double) * 3 * 3 * inum * umax_num_neighs);
        h_virial_sr_der2 = (double*)malloc(sizeof(double) * 3 * 3 * inum * umax_num_neighs);
        h_virial_sr_der2_ = (double*)malloc(sizeof(double) * 3 * 3 * inum * umax_num_neighs);

        h_dei_drij = (double*)malloc(sizeof(double) * inum * umax_num_neighs * 3);
        for (int ii=0; ii<inum*umax_num_neighs*3; ii++)
            h_dei_drij[ii] = 1.0 + 0.001 * ii;
    }

    void TearDown() override {
        free(h_ilist);
        free(h_numneigh);
        free(h_firstneigh);
        free(h_rcs);
        free(h_types);

        free(h_virial_sr_val);
        free(h_virial_sr_val_);
        free(h_virial_sr_der1);
        free(h_virial_sr_der1_);
        free(h_virial_sr_der2);
        free(h_virial_sr_der2_);

        free(h_dei_drij);
    }
};  // class : VirialSrTest


TEST_F(VirialSrTest, find_virial_sr_var_der) {
    int center_idx_modify = 0;
    int neighbor_idx_modify = 1;
    int direction_idx_modify = 0;

    ai2pot::fvt::find_virial_sr_val_der_launcher<double>(
        h_virial_sr_val,
        h_virial_sr_der1,
        h_virial_sr_der2,
        inum,
        h_numneigh,
        h_rcs,
        umax_num_neighs,
        h_dei_drij);
    h_dei_drij[center_idx_modify*umax_num_neighs*3 + neighbor_idx_modify*3 + direction_idx_modify] += 0.001;
    ai2pot::fvt::find_virial_sr_val_der_launcher<double>(
        h_virial_sr_val_,
        h_virial_sr_der1_,
        h_virial_sr_der2_,
        inum,
        h_numneigh,
        h_rcs,
        umax_num_neighs,
        h_dei_drij);
    
nblist.show_in_prim_index();
printf("1.1. Value of virial\n\t");
for (int ii=0; ii<9; ii++)
    printf("%6lf, ", h_virial_sr_val[ii]);
printf("\n");
printf("1.2. Value of virial:\n\t");
for (int ii=0; ii<9; ii++)
    printf("%6lf, ", h_virial_sr_val[ii]);
printf("\n");

printf("2.1. Derivatives w.r.t. dei_drij[%d][%d][%d] calculated by custom code:\n\t", center_idx_modify, neighbor_idx_modify, direction_idx_modify);
for (int bb=0; bb<3; bb++)
    printf("%6lf, ", h_virial_sr_der1[(center_idx_modify*3+bb)*inum*umax_num_neighs + center_idx_modify*umax_num_neighs + neighbor_idx_modify]);
printf("\n");
printf("2.2. Derivatives w.r.t. dei_drij[%d][%d][%d] calculated by finite difference method:\n\t", center_idx_modify, neighbor_idx_modify, direction_idx_modify);
for (int ii=0; ii<9; ii++)
    printf("%6lf, ", (h_virial_sr_val_[ii] - h_virial_sr_val[ii]) / 0.001);
printf("\n");
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}