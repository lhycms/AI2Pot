#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>

#include "../include/force_sr.cuh"
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"


class ForceSrTest : public ::testing::Test
{
protected:
    double *h_force_sr_val;
    double *d_force_sr_val;
    double *h_force_sr_val_;
    double *d_force_sr_val_;
    double *h_force_sr_der;
    double *d_force_sr_der;
    double *h_force_sr_der_;
    double *d_force_sr_der_;
    double *h_dei_drij;
    double *d_dei_drij;

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
    int *d_ilist;
    int *h_numneigh;
    int *d_numneigh;
    int *h_firstneigh;
    int *d_firstneigh;
    double *h_rcs;
    double *d_rcs;
    int *h_types;
    int *d_types;
    int nghost;

    ai2pot::Structure<double> structure;
    ai2pot::NeighborList<double> nblist;

    static void SetUpTestSuite() {
        std::cout << "ForceSrTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "ForceSrTest (TestSuite) is tearing down...\n";
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
        CHECK( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
        h_numneigh = (int*)malloc(sizeof(int) * inum);
        CHECK( cudaMalloc((void**)&d_numneigh, sizeof(int) * inum) );
        h_firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neighs);
        CHECK( cudaMalloc((void**)&d_firstneigh, sizeof(int) * inum * umax_num_neighs) );
        h_rcs = (double*)malloc(sizeof(double) * inum * umax_num_neighs * 3);
        CHECK( cudaMalloc((void**)&d_rcs, sizeof(double) * inum * umax_num_neighs * 3) );
        h_types = (int*)malloc(sizeof(int) * (inum + nghost));
        CHECK( cudaMalloc((void**)&d_types, sizeof(int) * (inum + nghost)) );

        nblist.find_info4mlff(
            inum,
            h_ilist,
            h_numneigh,
            h_firstneigh,
            h_rcs,
            h_types,
            nghost,
            umax_num_neighs);

        h_force_sr_val = (double*)malloc(sizeof(double) * (inum+nghost) * 3);
        CHECK( cudaMalloc((void**)&d_force_sr_val, sizeof(double) * (inum+nghost) * 3) );
        h_force_sr_der = (double*)malloc(sizeof(double) * (inum+nghost) * 3 * inum * umax_num_neighs);
        CHECK( cudaMalloc((void**)&d_force_sr_der, sizeof(double) * (inum+nghost) * 3 * inum * umax_num_neighs) );
        h_force_sr_val_ = (double*)malloc(sizeof(double) * (inum+nghost) * 3);
        CHECK( cudaMalloc((void**)&d_force_sr_val_, sizeof(double) * (inum+nghost) * 3) );
        h_force_sr_der_ = (double*)malloc(sizeof(double) * (inum+nghost) * 3 * inum * umax_num_neighs);
        CHECK( cudaMalloc((void**)&d_force_sr_der_, sizeof(double) * (inum+nghost) * 3 * inum * umax_num_neighs) );
        h_dei_drij = (double*)malloc(sizeof(double) * inum * umax_num_neighs * 3);
        CHECK( cudaMalloc((void**)&d_dei_drij, sizeof(double) * inum * umax_num_neighs * 3) );
        for (int ii=0; ii<inum*umax_num_neighs*3; ii++)
            h_dei_drij[ii] = 1.0 + 0.001 * ii;
    }

    void TearDown() override {
        free(h_ilist);
        CHECK( cudaFree(d_ilist) );
        free(h_numneigh);
        CHECK( cudaFree(d_numneigh) );
        free(h_firstneigh);
        CHECK( cudaFree(d_firstneigh) );
        free(h_rcs);
        CHECK( cudaFree(d_rcs) );
        free(h_types);
        CHECK( cudaFree(d_types) );

        free(h_force_sr_val);
        CHECK( cudaFree(d_force_sr_val) );
        free(h_force_sr_der);
        CHECK( cudaFree(d_force_sr_der) );
        free(h_force_sr_val_);
        CHECK( cudaFree(d_force_sr_val_) );
        free(h_force_sr_der_);
        CHECK( cudaFree(d_force_sr_der_) );
        free(h_dei_drij);
        CHECK( cudaFree(d_dei_drij) );
    }
};  // class : ForceSrTest


TEST_F(ForceSrTest, find_force_sr_val_der)
{
    int center_idx_modify = 0;
    int neigh_idx_modify = 1;
    int direction_modify = 1;

    ai2pot::fvt::find_force_sr_val_der_launcher(
        h_force_sr_val,
        h_force_sr_der,
        inum,
        h_ilist,
        h_numneigh,
        h_firstneigh,
        nghost,
        umax_num_neighs,
        h_dei_drij);
    CHECK( cudaDeviceSynchronize() );
    h_dei_drij[center_idx_modify*umax_num_neighs*3 + neigh_idx_modify*3 + direction_modify] += 0.001;
    ai2pot::fvt::find_force_sr_val_der_launcher(
        h_force_sr_val_,
        h_force_sr_der_,
        inum,
        h_ilist,
        h_numneigh,
        h_firstneigh,
        nghost,
        umax_num_neighs,
        h_dei_drij);
    CHECK( cudaDeviceSynchronize() );
nblist.show_in_prim_index();

printf("1.1. Value of force:\n\t");
for (int ii=0; ii<(inum+nghost)*3; ii++)
    printf("%6lf, ", h_force_sr_val[ii]);
std::cout << "\n";
printf("1.2. Value of force:\n\t");
for (int ii=0; ii<(inum+nghost)*3; ii++)
    printf("%6lf, ", h_force_sr_val_[ii]);
std::cout << "\n";

printf("2.1. Derivatives w.r.t. dei_drij[%d][%d][%d] calculated by custom code:\n\t", center_idx_modify, neigh_idx_modify, direction_modify);
for (int ii=0; ii<(inum+nghost); ii++) {
    printf("%6lf, ", h_force_sr_der[(ii*3+direction_modify)*inum*umax_num_neighs + center_idx_modify*umax_num_neighs + neigh_idx_modify]);
}
std::cout << "\n";
printf("2.2. Derivatives w.r.t. dei_drij[%d][%d][%d] calculated by finite difference method:\n\t");
for (int ii=0; ii<(inum+nghost)*3; ii++)
    printf("%6lf, ", (h_force_sr_val_[ii] - h_force_sr_val[ii]) / 0.001);
printf("\n");
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
