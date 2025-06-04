#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>

#include "../include/mtp_utilities.cuh"
#include "../include/linearMtp.cuh"
#include "../include/linearMtpLoss.cuh"
#include "../../include/mtpParam.h"
#include "../../../../nblist/include/structure.h"
#include "../../../../nblist/include/neighborList.h"


typedef double real;

class LinearMtpTest : public ::testing::Test
{
protected:
    std::vector<std::string> filenames;
    ai2pot::mtpr::MtpParam mtp_param;
    int chebyshev_size;
    int ntypes;
    real zbl_rmax;
    real zbl_rmin;
    real *zbl_cks;
    real *zbl_dks;

    real *coeffs;
    real *linear_coeffs;
    real *type_bias;
    real rmax;
    real rmin;

    int num_atoms;
    real basis_vectors[3][3];
    int atomic_numbers[12];
    real frac_coords[12][3];
    real rcut;
    real bin_size_xyz[3];
    bool pbc_xyz[3];
    int type_map[2];

    int umax_num_neigh_atoms;
    int inum;
    int *ilist;
    int *numneigh;
    int *firstneigh;
    real *rcs;
    int *types;
    int nghost;

    ai2pot::Structure<real> structure;
    ai2pot::NeighborList<real> neighbor_list;


    real etot;
    real etot_;
    real (*force)[3];
    real (*force_)[3];
    real *virial;
    real *virial_;


    real *loss_der2coeffs;
    real *loss_der2linear_coeffs;
    real *loss_der2type_bias;

    real etot_dft;
    real (*force_dft)[3];
    real *virial_dft;


    static void SetUpTestSuite() {
        std::cout << "LinearMtpTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "LinearMtpTest (TestSuite) is tearing down...\n";
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
        chebyshev_size = 8;

        ntypes = 2;
        zbl_rmin = 1.0;
        zbl_rmax = 2.0;
        zbl_cks = (real*)malloc(sizeof(real) * ntypes * ntypes * 4);
        zbl_dks = (real*)malloc(sizeof(real) * ntypes * ntypes * 4);
        for (int ii=0; ii<ntypes; ii++) {
            for (int jj=0; jj<ntypes; jj++) {
                zbl_cks[(ii*ntypes + jj)*4 + 0] = 0.18175;
                zbl_cks[(ii*ntypes + jj)*4 + 1] = 0.50986;
                zbl_cks[(ii*ntypes + jj)*4 + 2] = 0.28022;
                zbl_cks[(ii*ntypes + jj)*4 + 3] = 0.02817;
                zbl_dks[(ii*ntypes + jj)*4 + 0] = 3.1998;
                zbl_dks[(ii*ntypes + jj)*4 + 1] = 0.94229;
                zbl_dks[(ii*ntypes + jj)*4 + 2] = 0.4029;
                zbl_dks[(ii*ntypes + jj)*4 + 3] = 0.20162;
            }
        }

        coeffs = (real*)malloc(sizeof(real) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        linear_coeffs = (real*)malloc(sizeof(real) * mtp_param.alpha_scalar_moments());
        type_bias = (real*)malloc(sizeof(real) * ntypes);
        for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
            coeffs[ii] = 0.1 + 0.01 * ii;
        for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
            linear_coeffs[ii] = 0.1 + 0.01 * ii;
        type_bias[0] = -0.1;
        type_bias[1] = -0.2;
        rmax = 5.0;
        rmin = 2.0;

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
        
        type_map[0] = 42;
        type_map[1] = 16;

        structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);
        umax_num_neigh_atoms = 20;
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
        

        etot = 0.0;
        force = (real (*)[3])malloc(sizeof(real) * (inum+nghost) * 3);
        memset(force, 0.0, sizeof(real)*(inum+nghost)*3);
        virial = (real*)malloc(sizeof(real) * 9);
        memset(virial, 0.0, sizeof(real)*9);

        etot_ = 0.0;
        force_ = (real (*)[3])malloc(sizeof(real) * (inum+nghost) * 3);
        memset(force_, 0.0, sizeof(real)*(inum+nghost)*3);
        virial_ = (real*)malloc(sizeof(real) * 9);
        memset(virial_, 0.0, sizeof(real) * 9);

        loss_der2coeffs = (real*)malloc(sizeof(real) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        memset(loss_der2coeffs, 0, sizeof(real) * ntypes * ntypes * mtp_param.nmus() * chebyshev_size);
        loss_der2linear_coeffs = (real*)malloc(sizeof(real) * mtp_param.alpha_scalar_moments());
        memset(loss_der2linear_coeffs, 0, sizeof(real) * mtp_param.alpha_scalar_moments());
        loss_der2type_bias = (real*)malloc(sizeof(real) * ntypes);
        memset(loss_der2type_bias, 0, sizeof(real) * ntypes);

        etot_dft = 0.0;
        force_dft = (real (*)[3])malloc(sizeof(real) * inum * 3);
        memset(force_dft, 0, sizeof(real)*inum*3);
        virial_dft = (real*)malloc(sizeof(real) * 9);
        memset(virial_dft, 0, sizeof(real)*9);
    }

    void TearDown() override {
        free(zbl_cks);
        free(zbl_dks);
        free(ilist);
        free(numneigh);
        free(firstneigh);
        free(rcs);
        free(types);

        free(coeffs);
        free(linear_coeffs);
        free(type_bias);

        free(force);
        free(virial);
        free(force_);
        free(virial_);

        free(loss_der2coeffs);
        free(loss_der2linear_coeffs);
        free(loss_der2type_bias);
        free(force_dft);
        free(virial_dft);
    }
};  // class : LinearMtpTest



TEST_F(LinearMtpTest, find_efv_launcher)
{
    real delta = 1e-6;
    int center_idx_modify = 1;
    int direction_idx_modify = 2;

    ai2pot::mtpr::find_efv_launcher<real>(
        etot,
        force,
        virial,
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
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks,
        zbl_dks);

    // *** delta
    real cart_coords[inum][3] = {0.};
    for (int ii=0; ii<inum; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction_idx_modify] += delta;
    structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);
    neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);
    neighbor_list.find_info4mlff(
        inum,
        ilist,
        numneigh,
        firstneigh,
        rcs,
        types,
        nghost,
        umax_num_neigh_atoms);
    
    ai2pot::mtpr::find_efv_launcher<real>(
        etot_,
        force_,
        virial_,
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
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks,
        zbl_dks);


printf("1.1. energy = %.15lf\n", etot);
printf("1.1. force[%d][%d] calculated by custom code = %.15lf\n", center_idx_modify, direction_idx_modify, force[center_idx_modify][direction_idx_modify]);
printf("1.2. energy = %.15lf\n", etot_);
printf("1.2. force[%d][%d] calculated by finite difference method = %.15lf\n", center_idx_modify, direction_idx_modify, -(etot_ - etot) / delta);
printf("\n\nfind_efv_launcher:\n");
printf("\t2.1. Energy = %.15f\n", etot);
printf("\t2.2. Force =\n");
for (int ii=0; ii<inum; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, force[ii][0], force[ii][1], force[ii][2]);

}


TEST_F(LinearMtpTest, find_ef_launcher)
{
    real delta = 1e-8;
    int center_idx_modify = 3;
    int direction_idx_modify = 1;

    ai2pot::mtpr::find_ef_launcher<real>(
        etot,
        force,
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
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks,
        zbl_dks);



    // *** delta
    real cart_coords[inum][3] = {0.};
    for (int ii=0; ii<inum; ii++)
        for (int aa=0; aa<3; aa++)
            cart_coords[ii][aa] = structure.get_cart_coords()[ii][aa];
    cart_coords[center_idx_modify][direction_idx_modify] += delta;
    structure = ai2pot::Structure<real>(num_atoms, basis_vectors, atomic_numbers, cart_coords, true);   
    neighbor_list = ai2pot::NeighborList<real>(structure, rcut, bin_size_xyz, pbc_xyz, true);
    neighbor_list.find_info4mlff(
        inum,
        ilist,
        numneigh,
        firstneigh,
        rcs,
        types,
        nghost,
        umax_num_neigh_atoms);
    
    ai2pot::mtpr::find_ef_launcher<real>(
        etot_,
        force_,
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
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks,
        zbl_dks);

printf("1.1. energy = %.15lf\n", etot);
printf("1.1. force[%d][%d] calculated by custom code = %.15lf\n", center_idx_modify, direction_idx_modify, force[center_idx_modify][direction_idx_modify]);
printf("1.2. energy = %.15lf\n", etot_);
printf("1.2. force[%d][%d] calculated by finite difference method = %.15lf\n", center_idx_modify, direction_idx_modify, -(etot_ - etot) / delta);

printf("\n\nfind_ef_launcher():\n");
printf("\t2.1. Energy = %.15f\n", etot);
printf("\t2.2. Force =\n");
for (int ii=0; ii<inum; ii++)
    printf("\t\t%3d: [%.15f, %.15f, %.15f]\n", ii, force[ii][0], force[ii][1], force[ii][2]);

}


TEST_F(LinearMtpTest, find_loss_backward_launcer)
{
    real e_weight = 1.0;
    real f_weight = 0.0;
    real v_weight = 0.0;
    
    ai2pot::mtpr::find_ef_launcher<real>(
        etot,
        force,
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
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks,
        zbl_dks);

    ai2pot::mtpr::find_loss_backward_launcher<real>(
        loss_der2coeffs,
        loss_der2linear_coeffs,
        loss_der2type_bias,
        e_weight,
        f_weight,
        v_weight,
        etot,
        etot_dft,
        force,
        force_dft,
        virial,
        virial_dft,
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
        rmin,
        zbl_rmax,
        zbl_rmin,
        zbl_cks,
        zbl_dks);

printf("1. loss_der2coeffs:\n");
for (int ii=0; ii<ntypes*ntypes*mtp_param.nmus()*chebyshev_size; ii++)
    printf("%.15f, ", loss_der2coeffs[ii]);
printf("\n");
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
