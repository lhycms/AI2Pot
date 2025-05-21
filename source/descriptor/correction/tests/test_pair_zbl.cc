#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>

#include "../include/zbl.h"


class PairZBLTest : public ::testing::Test {
protected:
    int Zi;
    int Zj;
    double rmax;
    double rmin;
    double *ck;
    double *dk;
    double coord_1[3];
    double coord_2[3];
    double neigh_vec[3];
    double force[3];
    double virial[9];

    static void SetUpTestSuite() {
        std::cout << "PairZBLTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "PairZBLTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        Zi = 20;
        Zj = 24;
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

        coord_1[0] = 1.0;
        coord_1[1] = 1.0;
        coord_1[2] = 1.0;
        coord_2[0] = 2.01;
        coord_2[1] = 2.02;
        coord_2[2] = 2.03;
        neigh_vec[0] = coord_2[0] - coord_1[0];
        neigh_vec[1] = coord_2[1] - coord_1[1];
        neigh_vec[2] = coord_2[2] - coord_1[2];

        force[0] = 0;
        force[1] = 0;
        force[2] = 0;

        for (int ii=0; ii<9; ii++)
            virial[ii] = 0.0;
    }

    void TearDown() override {
        free(ck);
        free(dk);
    }
};  // class : PairZBLTest


TEST_F(PairZBLTest, find_switch_func) {
    ai2pot::correction::PairZBL<double> pair_zbl(Zi, Zj, rmax, rmin, ck, dk);
    double result = pair_zbl.find_switch_func(1.0);
    ASSERT_EQ(result, 1.0);
    result = pair_zbl.find_switch_func(2.0);
    ASSERT_EQ(result, 0.0);
}


TEST_F(PairZBLTest, find_switch_gradient) {
    double delta = 1e-5;

    ai2pot::correction::PairZBL<double> pair_zbl(Zi, Zj, rmax, rmin, ck, dk);
    double result = pair_zbl.find_switch_func(1.5);
    double gradient = pair_zbl.find_switch_func_der2rij(1.5);
    double result_ = pair_zbl.find_switch_func(1.5 + delta);

printf("\t1. Gradient calculated by custom code = %.10f\n", gradient);
printf("\t2. Gradient calculated by finite difference method = %.10f\n", (result_ - result) / delta);
}


TEST_F(PairZBLTest, find_phi_gradient) {
    double delta = 1e-5;

    ai2pot::correction::PairZBL<double> pair_zbl(Zi, Zj, rmax, rmin, ck, dk);
    double result = pair_zbl.find_phi_func(1.1);
    double gradient = pair_zbl.find_phi_func_der2rij(1.1);
    double result_ = pair_zbl.find_phi_func(1.1 + delta);

printf("\t1. Gradient calculated by custom code = %.10f\n", gradient);
printf("\t2. Gradient calculated by finite difference method = %.10f\n", (result_ - result) / delta);
}


TEST_F(PairZBLTest, find_force_accuracy) {
    double delta = 1e-6;
    double distance_ij = std::sqrt(std::pow(neigh_vec[0], 2)
                                   + std::pow(neigh_vec[1], 2)
                                   + std::pow(neigh_vec[2], 2));

    ai2pot::correction::PairZBL<double> pair_zbl(Zi, Zj, rmax, rmin, ck, dk);
    double pair_energy = pair_zbl.find_pair_energy(distance_ij);
    double pair_energy_ = pair_zbl.find_pair_energy(distance_ij + delta);

    pair_zbl.add_atomic_force_one(force, neigh_vec);

printf("Pair Energy = %.10lf\n", pair_energy);
printf("\t1. Gradient calculated by custom code = %.10f\n", force[0] * std::sqrt(3));
printf("\t2. Gradient calculated by finite difference method = %.10f\n", (pair_energy_ - pair_energy) / delta);
}


TEST_F(PairZBLTest, virial_accuracy) {
    double distance_ij = std::sqrt(std::pow(neigh_vec[0], 2)
                                   + std::pow(neigh_vec[1], 2)
                                   + std::pow(neigh_vec[2], 2));
    
    ai2pot::correction::PairZBL<double> pair_zbl(Zi, Zj, rmax, rmin, ck, dk);


    double calculated_virial[9];
    memset(calculated_virial, 0, sizeof(double) * 9);
    // Atom 1.
    memset(force, 0, sizeof(double) * 3);
    pair_zbl.add_atomic_force_one(force, neigh_vec);
    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            calculated_virial[aa*3 + bb] += coord_1[aa] * force[bb];
        }
    }
    pair_zbl.add_virial_one(virial, neigh_vec);
    // Atom 2.
    neigh_vec[0] = -neigh_vec[0];
    neigh_vec[1] = -neigh_vec[1];
    neigh_vec[2] = -neigh_vec[2];
    memset(force, 0, sizeof(double) * 3);
    pair_zbl.add_atomic_force_one(force, neigh_vec);
    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            calculated_virial[aa*3 + bb] += coord_2[aa] * force[bb];
        }
    }
    pair_zbl.add_virial_one(virial, neigh_vec);


printf("\t1. Virial calculated by custom code:\n");
for (int aa=0; aa<3; aa++) {
    for (int bb=0; bb<3; bb++) {
        printf("%.10f, ", calculated_virial[aa*3 + bb]);
    }
}
printf("\n");
printf("\t2. Virial calculated by definition:\n");
for (int aa=0; aa<3; aa++) {
    for (int bb=0; bb<3; bb++) {
        printf("%.10f, ", virial[aa*3 + bb]);
    }
}
printf("\n");
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
