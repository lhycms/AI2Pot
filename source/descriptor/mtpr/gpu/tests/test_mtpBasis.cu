#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <string>

#include "../include/mtp_utilities.cuh"
#include "../include/mtpBasis.cuh"
#include "../../include/mtpParam.h"
#include "../../../../nblist/include/structure.h"
#include "../../../../nblist/include/neighborList.h"


class BasisUtilitesTest : public ::testing::Test 
{
protected:
    double rmax;
    double rmin;
    double distance_ij;

    double h_switch_func_val;
    double h_switch_func_der2r;

    int chebyshev_size;
    double *h_rb_chebyshev_vals;
    double *h_rb_chebyshev_ders2r;

    double *h_rq_chebyshev_vals;
    double *h_rq_chebyshev_ders2r;

    static void SetUpTestSuite() {
        std::cout << "BasisUtilitiesTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "BasisUtilitiesTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rmax = 5.0;
        rmin = 2.0;
        distance_ij = 3.14;

        chebyshev_size = 8;
        h_rb_chebyshev_vals = (double*)malloc(chebyshev_size * sizeof(double));
        h_rb_chebyshev_ders2r = (double*)malloc(chebyshev_size * sizeof(double));

        h_rq_chebyshev_vals = (double*)malloc(chebyshev_size * sizeof(double));
        h_rq_chebyshev_ders2r = (double*)malloc(chebyshev_size * sizeof(double));
    }

    void TearDown() override {
        free(h_rb_chebyshev_vals);
        free(h_rb_chebyshev_ders2r);

        free(h_rq_chebyshev_vals);
        free(h_rq_chebyshev_ders2r);
    }
};  // class : BasisUtilitiesTest


class MtpBasisTest : public ::testing::Test
{
protected:
    double *h_mtp_basis_val;
    double *d_mtp_basis_val;
    double (*h_mtp_basis_der);
    double (*d_mtp_basis_der);
    double *h_mtp_basis_der2coeffs;
    double *d_mtp_basis_der2coeffs;
    double *h_mtp_basis_val_;
    double *d_mtp_basis_val_;
    double (*h_mtp_basis_der_);
    double (*d_mtp_basis_der_);
    double *h_mtp_basis_der2coeffs_;
    double *d_mtp_basis_der2coeffs_;
    int chebyshev_size;
    double *h_coeffs;
    double *d_coeffs;
    int nmus;
    double rmax;
    double rmin;
    int ntypes;
    double *d_moms_vals;
    double *d_moms_ders;
    double *d_moms_ders2coeffs;

    std::vector<std::string> filenames;
    ai2pot::mtpr::MtpParam mtp_param;
    int *d_alpha_index_basic;
    int *d_alpha_index_times;
    int *d_alpha_moment_mapping;
    int *d_num_mus4moms;
    int *d_mus4moms_ptr;
    
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];

    int umax_num_neigh_atoms;
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
    ai2pot::NeighborList<double> neighbor_list;

    static void SetUpTestSuite() {
        std::cout << "MtpBasisTest (TestSuite) is setting up...\n";

    }

    static void TearDownTestSuite() {
        std::cout << "MtpBasisTest (TestSuite) is tearing down...\n";
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
        mtp_param._load(filenames[4]);
//mtp_param.show();
        CHECK( cudaMalloc((void**)&d_alpha_index_basic, sizeof(int) * mtp_param.alpha_index_basic_count() * 4) );
        CHECK( cudaMalloc((void**)&d_alpha_index_times, sizeof(int) * mtp_param.alpha_index_times_count() * 4) );
        CHECK( cudaMalloc((void**)&d_alpha_moment_mapping, sizeof(int) * mtp_param.alpha_scalar_moments()) );
        CHECK( cudaMalloc((void**)&d_num_mus4moms, sizeof(int) * mtp_param.alpha_moments_count()) );
        CHECK( cudaMalloc((void**)&d_mus4moms_ptr, sizeof(int) * mtp_param.alpha_moments_count() * mtp_param.max_num_mus4mom()) );
        CHECK( cudaMemcpy(d_alpha_index_basic, 
                          (int*)mtp_param.alpha_index_basic(), 
                          sizeof(int)*mtp_param.alpha_index_basic_count()*4,
                          cudaMemcpyHostToDevice) 
        );
        CHECK( cudaMemcpy(d_alpha_index_times,
                          (int*)mtp_param.alpha_index_times(),
                          sizeof(int)*mtp_param.alpha_index_times_count()*4,
                          cudaMemcpyHostToDevice) 
        );
        CHECK( cudaMemcpy(d_alpha_moment_mapping, 
                          mtp_param.alpha_moment_mapping(),
                          sizeof(int)*mtp_param.alpha_scalar_moments(),
                          cudaMemcpyHostToDevice)
        );

        inum = 12;
        ntypes = 2;
        chebyshev_size = 8;
        nmus = mtp_param.nmus();
        rmax = 5.0;
        rmin = 2.0;
        umax_num_neigh_atoms = 20;
        h_mtp_basis_val = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments());
        CHECK( cudaMalloc((void**)&d_mtp_basis_val, sizeof(double) * inum * mtp_param.alpha_scalar_moments()) );
        h_mtp_basis_der = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments() * umax_num_neigh_atoms * 3);
        CHECK( cudaMalloc((void**)&d_mtp_basis_der, sizeof(double) * inum * mtp_param.alpha_scalar_moments() * umax_num_neigh_atoms * 3) );
        h_mtp_basis_der2coeffs = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments() * ntypes * ntypes * nmus * chebyshev_size);
        CHECK( cudaMalloc((void**)&d_mtp_basis_der2coeffs, sizeof(double) * inum * mtp_param.alpha_scalar_moments() * ntypes * ntypes * nmus * chebyshev_size) );
        h_mtp_basis_val_ = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments());
        CHECK( cudaMalloc((void**)&d_mtp_basis_val_, sizeof(double) * inum * mtp_param.alpha_scalar_moments()) );
        h_mtp_basis_der_ = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments() * umax_num_neigh_atoms * 3);
        CHECK( cudaMalloc((void**)&d_mtp_basis_der_, sizeof(double) * inum * mtp_param.alpha_scalar_moments() * umax_num_neigh_atoms * 3) );
        h_mtp_basis_der2coeffs_ = (double*)malloc(sizeof(double) * inum * mtp_param.alpha_scalar_moments() * ntypes * ntypes * nmus * chebyshev_size);
        CHECK( cudaMalloc((void**)&d_mtp_basis_der2coeffs_, sizeof(double) * inum * mtp_param.alpha_scalar_moments() * ntypes * ntypes * nmus * chebyshev_size) );

        h_coeffs = (double*)malloc(sizeof(double) * ntypes * ntypes * nmus * chebyshev_size);
        cudaMalloc((void**)&d_coeffs, sizeof(double) * ntypes * ntypes * nmus * chebyshev_size);
        for (int ii=0; ii<ntypes*ntypes*nmus*chebyshev_size; ii++)
            h_coeffs[ii] = 1.0;
        CHECK( cudaMemcpy(d_coeffs, h_coeffs, sizeof(double)*ntypes*ntypes*nmus*chebyshev_size, cudaMemcpyHostToDevice) );
        CHECK( cudaMalloc((void**)&d_moms_vals, sizeof(double) * inum * mtp_param.alpha_moments_count()) );
        CHECK( cudaMalloc((void**)&d_moms_ders, sizeof(double) * inum * mtp_param.alpha_moments_count() * umax_num_neigh_atoms * 3) );
        CHECK( cudaMalloc((void**)&d_moms_ders2coeffs, sizeof(double) * inum * mtp_param.alpha_moments_count() * ntypes * ntypes * nmus * chebyshev_size) );


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
        neighbor_list = ai2pot::NeighborList<double>(structure, rcut, bin_size_xyz, pbc_xyz, true);

        //umax_num_neigh_atoms = 20;
        inum = 12;
        h_ilist = (int*)malloc(sizeof(int) * inum);
        CHECK( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
        h_numneigh = (int*)malloc(sizeof(int) * inum);
        CHECK( cudaMalloc((void**)&d_numneigh, sizeof(int) * inum) );
        h_firstneigh = (int*)malloc(sizeof(int) * inum * umax_num_neigh_atoms);
        CHECK( cudaMalloc((void**)&d_firstneigh, sizeof(int) * inum * umax_num_neigh_atoms) );
        h_rcs = (double*)malloc(sizeof(double) * inum * umax_num_neigh_atoms * 3);
        CHECK( cudaMalloc((void**)&d_rcs, sizeof(double) * inum * umax_num_neigh_atoms * 3) );
        h_types = (int*)malloc(sizeof(int) * inum);
        CHECK( cudaMalloc((void**)&d_types, sizeof(int) * inum) );
        neighbor_list.find_info4mlff(
            inum,
            h_ilist,
            h_numneigh,
            h_firstneigh,
            h_rcs,
            h_types,
            nghost,
            umax_num_neigh_atoms);

        CHECK( cudaMemcpy(d_ilist, h_ilist, sizeof(int)*inum, cudaMemcpyHostToDevice) );
        CHECK( cudaMemcpy(d_numneigh, h_numneigh, sizeof(int)*inum, cudaMemcpyHostToDevice) );
        CHECK( cudaMemcpy(d_firstneigh, h_firstneigh, sizeof(int)*inum*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
        CHECK( cudaMemcpy(d_rcs, h_rcs, sizeof(double)*inum*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
        CHECK( cudaMemcpy(d_types, h_types, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    }

    void TearDown() override {
        CHECK( cudaFree(d_alpha_index_basic) );
        CHECK( cudaFree(d_alpha_index_times) );
        CHECK( cudaFree(d_alpha_moment_mapping) );
        CHECK( cudaFree(d_num_mus4moms) );
        CHECK( cudaFree(d_mus4moms_ptr) );

        free(h_mtp_basis_val);
        cudaFree(d_mtp_basis_val);
        free(h_mtp_basis_der);
        cudaFree(d_mtp_basis_der);
        free(h_mtp_basis_der2coeffs);
        CHECK( cudaFree(d_mtp_basis_der2coeffs) );
        free(h_mtp_basis_val_);
        CHECK( cudaFree(d_mtp_basis_val_) );
        free(h_mtp_basis_der_);
        CHECK( cudaFree(d_mtp_basis_der_) );
        free(h_mtp_basis_der2coeffs_);
        CHECK( cudaFree(d_mtp_basis_der2coeffs_) );

        CHECK( cudaFree(d_moms_vals) );
        CHECK( cudaFree(d_moms_ders) );
        CHECK( cudaFree(d_moms_ders2coeffs) );

        free(h_coeffs);
        CHECK( cudaFree(d_coeffs) );

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
    }
};  // class : MtpBasisTest


TEST_F(BasisUtilitesTest, switch_func_test) {
    distance_ij = 3.14;
    ai2pot::mtpr::find_switch_func<double>(h_switch_func_val,
                                           h_switch_func_der2r,
                                           rmax,
                                           rmin, 
                                           distance_ij);
printf("BasisUtilitiesTest.switch_func_test:\n");
printf("\t1. h_switch_func_val = %g\n", h_switch_func_val);
printf("\t2. h_switch_func_der2r = %g\n", h_switch_func_der2r);
}


TEST_F(BasisUtilitesTest, rb_chebyshev_test) {
    distance_ij = 3.14;
    ai2pot::mtpr::find_rb_chebyshev<double>(h_rb_chebyshev_vals,
                                            h_rb_chebyshev_ders2r,
                                            chebyshev_size,
                                            rmax,
                                            rmin,
                                            distance_ij);
printf("BasisUtilitiesTest.rb_chebyshev_test:\n");
printf("\t1. rb_chebyshev_vals = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rb_chebyshev_vals[ii]);
printf("\n");
printf("\t1. rb_chebyshev_der2r = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rb_chebyshev_ders2r[ii]);
printf("\n");
}


TEST_F(BasisUtilitesTest, rq_chebyshev_test) {
    distance_ij = 3.14;
    ai2pot::mtpr::find_rq_chebyshev<double>(h_rq_chebyshev_vals,
                                            h_rq_chebyshev_ders2r,
                                            chebyshev_size,
                                            rmax,
                                            rmin,
                                            distance_ij);
printf("BasisUtilitiesTest.rq_chebyshev_test:\n");
printf("\t1. rq_chebyshev_vals = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rq_chebyshev_vals[ii]);
printf("\n");
printf("\t1. rq_chebyshev_der2r = ");
for (int ii=0; ii<chebyshev_size; ii++)
    printf("%g, ", h_rq_chebyshev_ders2r[ii]);
printf("\n");
}


TEST_F(MtpBasisTest, find_mtp_basis_val_der) {
    ai2pot::mtpr::find_mtp_basis_val_der_cuda_kernel<double> KERNEL_ARG2(2, 12) (
        d_mtp_basis_val,
        (double (*)[3])d_mtp_basis_der,
        d_mtp_basis_der2coeffs,
        chebyshev_size,
        d_coeffs,
        mtp_param.alpha_moments_count(),
        mtp_param.alpha_index_basic_count(),
        (int (*)[4])d_alpha_index_basic,
        mtp_param.alpha_index_times_count(),
        (int (*)[4])d_alpha_index_times,
        mtp_param.alpha_scalar_moments(),
        d_alpha_moment_mapping,
        mtp_param.max_num_mus4mom(),
        d_num_mus4moms,
        d_mus4moms_ptr,
        nmus,
        inum,
        d_ilist,
        d_numneigh,
        d_firstneigh,
        (double (*)[3])d_rcs,
        d_types,
        ntypes,
        umax_num_neigh_atoms,
        rmax,
        rmin,
        d_moms_vals,
        (double (*)[3])d_moms_ders,
        d_moms_ders2coeffs);

    CHECK( cudaDeviceSynchronize() );
    CHECK( cudaMemcpy(h_mtp_basis_val, d_mtp_basis_val, sizeof(double)*mtp_param.alpha_scalar_moments(), cudaMemcpyDeviceToHost) );
for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
    printf("%10lf, ", h_mtp_basis_val[0*mtp_param.alpha_scalar_moments() + ii]);
printf("\n");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
