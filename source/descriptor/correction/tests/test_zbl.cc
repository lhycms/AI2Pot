#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>

#include "../include/zbl.h"


class ZBLTest : public ::testing::Test {
protected:
    int Zi;
    int Zj;
    double rmax;
    double rmin;
    double *ck;
    double *dk;
    double neigh_vec[3];
    double force[3];

    static void SetUpTestSuite() {
        std::cout << "ZBLTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "ZBLTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        Zi = 16;
        Zj = 42;
        rmax = 2.0;
        rmin = 1.0;
        ck = (double*)malloc(sizeof(double) * 4);
        dk = (double*)malloc(sizeof(double) * 4);
        ck[0] = 0.18175;
        ck[1] = 0.50986;
        ck[2] = 0.28022;
        ck[3] = 0.02817;

        dk[0] = 3.1998;
        dk[1] = 0.94229;
        dk[2] = 0.4029;
        dk[3] = 0.20162;

        neigh_vec[0] = 1.01;
        neigh_vec[1] = 1.01;
        neigh_vec[2] = 1.01;

        force[0] = 0;
        force[1] = 0;
        force[2] = 0;
    }

    void TearDown() override {
        free(ck);
        free(dk);
    }
};  // class : ZBLTest


TEST_F(ZBLTest, find_switch_func) {
    ai2pot::correction::ZBL<double> zbl(Zi, Zj, rmax, rmin, ck, dk);
    double result = zbl.find_switch_func(1.0);
    ASSERT_EQ(result, 1.0);
    result = zbl.find_switch_func(2.0);
    ASSERT_EQ(result, 0.0);
}


TEST_F(ZBLTest, find_switch_gradient) {
    double delta = 1e-5;

    ai2pot::correction::ZBL<double> zbl(Zi, Zj, rmax, rmin, ck, dk);
    double result = zbl.find_switch_func(1.5);
    double gradient = zbl.find_switch_func_der2rij(1.5);
    double result_ = zbl.find_switch_func(1.5 + delta);

printf("\t1. Gradient calculated by custom code = %.10f\n", gradient);
printf("\t2. Gradient calculated by finite difference method = %.10f\n", (result_ - result) / delta);
}


TEST_F(ZBLTest, find_phi_gradient) {
    double delta = 1e-5;

    ai2pot::correction::ZBL<double> zbl(Zi, Zj, rmax, rmin, ck, dk);
    double result = zbl.find_phi_func(1.1);
    double gradient = zbl.find_phi_func_der2rij(1.1);
    double result_ = zbl.find_phi_func(1.1 + delta);

printf("\t1. Gradient calculated by custom code = %.10f\n", gradient);
printf("\t2. Gradient calculated by finite difference method = %.10f\n", (result_ - result) / delta);
}


TEST_F(ZBLTest, find_zbl_pair_energy) {
    double delta = 1e-6;
    double distance_ij = std::sqrt(std::pow(neigh_vec[0], 2)
                                   + std::pow(neigh_vec[1], 2)
                                   + std::pow(neigh_vec[2], 2));
    printf("distance_ij = %g\n", distance_ij);
    ai2pot::correction::ZBL<double> zbl(Zi, Zj, rmax, rmin, ck, dk);
    double pair_energy = zbl.find_pair_energy(distance_ij);
    double pair_energy_ = zbl.find_pair_energy(distance_ij + delta);

    zbl.add_atomic_force(force, neigh_vec);

printf("\t1. Gradient calculated by custom code = %.10f\n", force[0] * std::sqrt(3));
printf("\t2. Gradient calculated by finite difference method = %.10f\n", (pair_energy_ - pair_energy) / delta);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
