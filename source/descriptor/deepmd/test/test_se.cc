#include <gtest/gtest.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../nblist/include/structure.h"
#include "../include/se.h"
#include "../../../core/include/arrayUtils.h"


class SwitchFuncTest : public ::testing::Test {
protected:
    double rcut;
    double rcut_smooth;

    static void SetUpTestSuite() {
        std::cout << "SwitchFuncTest (TestSuite) is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "SwitchFuncTest (TestSuite) is tearing down...\n";
    }

    void SetUp() override {
        rcut = 3.5;
        rcut_smooth = 3.0;
    }

    void TearDown() override {

    }
};  // class : SwitchFuncTest



TEST_F(SwitchFuncTest, show) {
    matersdk::deepPotSE::SwitchFunc<double> switch_func(rcut, rcut_smooth);
    //switch_func.show();
}


TEST_F(SwitchFuncTest, get_result) {
    rcut = 3.5;
    rcut_smooth = 3.0;

    matersdk::deepPotSE::SwitchFunc<double> switch_func(rcut, rcut_smooth);
    // Case 1
    EXPECT_FLOAT_EQ(switch_func.get_result(3.0), 1.0);      // Case 3
    EXPECT_FLOAT_EQ(switch_func.get_result(2.9), 1.0);      // Case 3
    // Case 2
    EXPECT_FLOAT_EQ(switch_func.get_result(3.2), 0.68256);  // Case 2
    // Case 3
    EXPECT_FLOAT_EQ(switch_func.get_result(3.6), 0.0);      // Case 1
    EXPECT_FLOAT_EQ(switch_func.get_result(3.5), 0.0);      // Case 1
}


TEST_F(SwitchFuncTest, get_deriv2r) {
    matersdk::deepPotSE::SwitchFunc<double> switch_func(rcut, rcut_smooth);

    // Case 1
    EXPECT_FLOAT_EQ(switch_func.get_deriv2r(2.9), 0.0);
    EXPECT_FLOAT_EQ(switch_func.get_deriv2r(3.0), 0.0);

    // Case 2
    EXPECT_FLOAT_EQ(switch_func.get_deriv2r(3.2), -3.456);

    // Case 3
    EXPECT_FLOAT_EQ(switch_func.get_deriv2r(3.5), 0.0);
    EXPECT_FLOAT_EQ(switch_func.get_deriv2r(3.6), 0.0);
}




class SmoothFuncTest : public ::testing::Test {
protected:
    double rcut;
    double rcut_smooth;

    double distance_ji;


    static void SetUpTestSuite() {
        std::cout << "SmoothFuncTest TestSuite is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "SmoothFuncTest TestSuite is tearing down...\n";
    }

    void SetUp() override {
        
    }

    void TearDown() override {

    }
};


TEST_F(SmoothFuncTest, operation) {
    rcut = 3.3;
    rcut_smooth = 3.0;
    distance_ji = 3.3;
    double result = matersdk::deepPotSE::smooth_func<double>(distance_ji, rcut, rcut_smooth);
    EXPECT_FLOAT_EQ(result, 0);

    distance_ji = 3.0;
    result = matersdk::deepPotSE::smooth_func<double>(distance_ji, rcut, rcut_smooth);
    EXPECT_FLOAT_EQ(result, 1.0/3.0);   //  `= 1/r`
}




class RecipTest : public ::testing::Test {
protected:
    double value;
    double value_recip;

    static void SetUpTestSuite() {
        std::cout << "RecipTest TestSuite is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "RecipTest TestSuite is tearing down...\n";
    }
};  // class : RecipTest


TEST_F(RecipTest, operation) {
    value = 1;
    value_recip = matersdk::deepPotSE::recip(value);
    EXPECT_FLOAT_EQ(value_recip, 1.0);

    value = 2;
    value_recip = matersdk::deepPotSE::recip(value);
    EXPECT_FLOAT_EQ(value_recip, 0.5);
}




class PairTildeRTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];  

    int center_atomic_number;
    int neigh_atomic_number;
    int num_neigh_atoms;
    double rcut_smooth;

    matersdk::Structure<double> structure;
    matersdk::NeighborList<double> neighbor_list;


    static void SetUpTestSuite() {
        std::cout << "PairTildeR TestSuite is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "PairTildeR TestSuite is tearing down...\n";
    }


    void SetUp() override {
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

        atomic_numbers[0] = 42;
        atomic_numbers[1] = 16;
        atomic_numbers[2] = 16;
        atomic_numbers[3] = 42;
        atomic_numbers[4] = 16;
        atomic_numbers[5] = 16;
        atomic_numbers[6] = 42;
        atomic_numbers[7] = 16;
        atomic_numbers[8] = 16;
        atomic_numbers[9] = 42; 
        atomic_numbers[10] = 16;
        atomic_numbers[11] = 16;

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

        rcut = 3.3;
        bin_size_xyz[0] = 3.0;
        bin_size_xyz[1] = 3.0;
        bin_size_xyz[2] = 3.0;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = false; 


        structure = matersdk::Structure<double>(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
        neighbor_list = matersdk::NeighborList<double>(structure, rcut, pbc_xyz, true);
    }

    void TearDown() override {

    }
};


TEST_F(PairTildeRTest, default_constructor) {
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r;
    //pair_tilde_r.show();
}


TEST_F(PairTildeRTest, constructor_1) {
    center_atomic_number = 42;
    neigh_atomic_number = 16;
    num_neigh_atoms = 24;
    rcut_smooth = 3.0;
    
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r(neighbor_list, center_atomic_number, neigh_atomic_number, num_neigh_atoms, rcut_smooth);
    // pair_tilde_r.show();
}


TEST_F(PairTildeRTest, constructor_2) {
    center_atomic_number = 42;
    neigh_atomic_number = 16;
    int num_center_atoms = 4;   // 中心原子的数量
    num_neigh_atoms = 7;
    rcut_smooth = 3.0;

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r(neighbor_list, center_atomic_number, neigh_atomic_number, num_center_atoms, num_neigh_atoms, rcut_smooth);
    //pair_tilde_r.show_in_deriv();
}


TEST_F(PairTildeRTest, constructor_3) {
    center_atomic_number = 42;
    neigh_atomic_number = 16;
    rcut_smooth = 3.0;

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r(neighbor_list, center_atomic_number, neigh_atomic_number, rcut_smooth);
    //pair_tilde_r.show();
}


TEST_F(PairTildeRTest, constructor_4) {
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;
    bool sort = true;

    center_atomic_number = 42;
    neigh_atomic_number = 16;
    num_neigh_atoms = 100;
    rcut = 3.3;
    rcut_smooth = 3.0;

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r(structure, rcut, pbc_xyz, sort, center_atomic_number, neigh_atomic_number, num_neigh_atoms, rcut_smooth);
    //pair_tilde_r.show();
}


TEST_F(PairTildeRTest, constructor_5) {
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;
    bool sort = true;

    center_atomic_number = 16;
    neigh_atomic_number = 16;
    rcut = 3.3;
    rcut_smooth = 3.0;

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r(structure, rcut, pbc_xyz, sort, center_atomic_number, neigh_atomic_number, rcut_smooth);
    pair_tilde_r.show();
}


TEST_F(PairTildeRTest, copy_constructor) {
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = true;
    bool sort = true;

    center_atomic_number = 16;
    neigh_atomic_number = 16;
    rcut = 3.3;
    rcut_smooth = 3.0;

    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, sort);

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_1;
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_2(neighbor_list, center_atomic_number, neigh_atomic_number, rcut_smooth);
    
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_3(pair_tilde_r_1);
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_4(pair_tilde_r_2);
    //pair_tilde_r_3.show();
    //pair_tilde_r_4.show();
}


TEST_F(PairTildeRTest, assignment_operator) {
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;

    center_atomic_number = 16;
    neigh_atomic_number = 16;
    rcut = 3.3;
    rcut_smooth = 3.0;

    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_1;
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_2(neighbor_list, center_atomic_number, neigh_atomic_number, rcut_smooth);

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_3;
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_4(neighbor_list, center_atomic_number, neigh_atomic_number, rcut_smooth);

    pair_tilde_r_3 = pair_tilde_r_1;
    //pair_tilde_r_3.show();
    //pair_tilde_r_3.show_in_value();
    //pair_tilde_r_3.show_in_deriv();

    pair_tilde_r_3 = pair_tilde_r_2;
    //pair_tilde_r_3.show();
    //pair_tilde_r_3.show_in_value();
    //pair_tilde_r_3.show_in_deriv();

    pair_tilde_r_4 = pair_tilde_r_1;
    //pair_tilde_r_4.show();
    //pair_tilde_r_4.show_in_value();
    //pair_tilde_r_4.show_in_deriv();

    pair_tilde_r_4 = pair_tilde_r_2;
    //pair_tilde_r_4.show();
    //pair_tilde_r_4.show_in_value();
    //pair_tilde_r_4.show_in_deriv();
}


TEST_F(PairTildeRTest, get_num_center_atoms) {
    // Case 1.
    center_atomic_number = 42;
    neigh_atomic_number = 16;
    rcut_smooth = 3.0;

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_42_16(neighbor_list, center_atomic_number, neigh_atomic_number, rcut_smooth);
    double num_center_atoms_42_16 = pair_tilde_r_42_16.get_num_center_atoms();
    EXPECT_EQ(num_center_atoms_42_16, 4);

    // Case 2.
    center_atomic_number = 16;
    neigh_atomic_number = 16;
    rcut_smooth = 3.0;
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_16_16(neighbor_list, center_atomic_number, neigh_atomic_number, rcut_smooth);
    double num_center_atoms_16_16 = pair_tilde_r_16_16.get_num_center_atoms();
    EXPECT_EQ(num_center_atoms_16_16, 8);
}


TEST_F(PairTildeRTest, get_num_neigh_atoms) {
    // Case 1.
    center_atomic_number = 16;
    neigh_atomic_number = 16;
    rcut_smooth = 3.0;

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_16_16(neighbor_list, center_atomic_number, neigh_atomic_number, rcut_smooth);
    double num_neigh_atoms_16_16 = pair_tilde_r_16_16.get_num_neigh_atoms();
    EXPECT_EQ(num_neigh_atoms_16_16, 7);

    // Case 2.
    center_atomic_number = 42;
    neigh_atomic_number = 16;
    num_neigh_atoms = 100;
    rcut_smooth = 3.0;

    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r_16_16_(neighbor_list, center_atomic_number, neigh_atomic_number, num_neigh_atoms, rcut_smooth);
    double num_neigh_atoms_16_16_ = pair_tilde_r_16_16_.get_num_neigh_atoms();
    EXPECT_EQ(num_neigh_atoms_16_16_, 100);
}


TEST_F(PairTildeRTest, generate) {
    center_atomic_number = 42;
    neigh_atomic_number = 16;
    rcut = 3.3;
    rcut_smooth = 3.0;
    num_neigh_atoms = 14;

    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r(neighbor_list, center_atomic_number, neigh_atomic_number, num_neigh_atoms, rcut_smooth);
    double*** pair_tilde_r_matrix = pair_tilde_r.generate();
    
    pair_tilde_r.show_in_value();

    for (int ii=0; ii<pair_tilde_r.get_num_center_atoms(); ii++) {
        for (int jj=0; jj<pair_tilde_r.get_num_neigh_atoms(); jj++) {
            free(pair_tilde_r_matrix[ii][jj]);
        }
        free(pair_tilde_r_matrix[ii]);
    }
    free(pair_tilde_r_matrix);
}


TEST_F(PairTildeRTest, deriv) {
    center_atomic_number = 42;
    neigh_atomic_number = 42;
    rcut = 3.3;
    rcut_smooth = 3.0;
    num_neigh_atoms = 7;

    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::PairTildeR<double> pair_tilde_r(neighbor_list, center_atomic_number, neigh_atomic_number, num_neigh_atoms, rcut_smooth);
    //double**** pair_tilde_r_deriv = pair_tilde_r.deriv();
    pair_tilde_r.show_in_deriv();
}



class TildeRTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];

    // Step 2. For NeighborList
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];  

    // Step 3. For TildeR
    int num_center_atomic_numbers;
    int* center_atomic_numbers_lst;
    int num_neigh_atomic_numbers;
    int* neigh_atomic_numbers_lst;
    int* num_neigh_atoms_lst;
    double rcut_smooth;


    static void SetUpTestSuite() {
        std::cout << "TildeR TestSuite is setting up...\n";
    }

    static void TearDownTestSuite() {
        std::cout << "TildeR TestSuite is tearing down...\n";
    }

    void SetUp() override {
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

        atomic_numbers[0] = 42;
        atomic_numbers[1] = 16;
        atomic_numbers[2] = 16;
        atomic_numbers[3] = 42;
        atomic_numbers[4] = 16;
        atomic_numbers[5] = 16;
        atomic_numbers[6] = 42;
        atomic_numbers[7] = 16;
        atomic_numbers[8] = 16;
        atomic_numbers[9] = 42; 
        atomic_numbers[10] = 16;
        atomic_numbers[11] = 16;

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


        // Step 2. For NeighborList
        rcut = 3.3;
        bin_size_xyz[0] = 3.0;
        bin_size_xyz[1] = 3.0;
        bin_size_xyz[2] = 3.0;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = false; 

        
        // Step 3. For TildeR
        num_center_atomic_numbers = 2;
        center_atomic_numbers_lst = (int*)malloc(sizeof(int) * 2);
        center_atomic_numbers_lst[0] = 42;
        center_atomic_numbers_lst[1] = 16;

        num_neigh_atomic_numbers = 2;
        neigh_atomic_numbers_lst = (int*)malloc(sizeof(int) * 2);
        neigh_atomic_numbers_lst[0] = 42;
        neigh_atomic_numbers_lst[1] = 16;
        
        num_neigh_atoms_lst = (int*)malloc(sizeof(int) * 2);
        num_neigh_atoms_lst[0] = 14;
        num_neigh_atoms_lst[1] = 16;
        
        rcut_smooth = 3.0;
    }

    void TearDown() override {
        free(center_atomic_numbers_lst);
        free(neigh_atomic_numbers_lst);
        free(num_neigh_atoms_lst);
    }
};


TEST_F(TildeRTest, constructor_1) {
    num_neigh_atoms_lst[0] = 14;
    num_neigh_atoms_lst[1] = 16;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                neighbor_list, 
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut_smooth);
    
    //tilde_r.show();
}


TEST_F(TildeRTest, constructor_2) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                neighbor_list, 
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                rcut_smooth);
    
    //tilde_r.show();
}


TEST_F(TildeRTest, constructor_3) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                structure,
                                rcut,
                                pbc_xyz,
                                true,
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut_smooth);
    //tilde_r.show();
}


TEST_F(TildeRTest, constructor_4) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                structure,
                                rcut,
                                pbc_xyz,
                                true,
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                rcut_smooth);
    //tilde_r.show();
}


TEST_F(TildeRTest, copy_constructor) {
    num_center_atomic_numbers = 2;
    center_atomic_numbers_lst[0] = 42;
    center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 2;
    neigh_atomic_numbers_lst[0] = 42;
    neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10;
    num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);

    matersdk::deepPotSE::TildeR<double> tilde_r_1;
    matersdk::deepPotSE::TildeR<double> tilde_r_2(
                                        neighbor_list, 
                                        num_center_atomic_numbers, 
                                        center_atomic_numbers_lst, 
                                        num_neigh_atomic_numbers, 
                                        neigh_atomic_numbers_lst, 
                                        num_neigh_atoms_lst,
                                        rcut_smooth);
    
    matersdk::deepPotSE::TildeR<double> tilde_r_3(tilde_r_1);
    matersdk::deepPotSE::TildeR<double> tilde_r_4(tilde_r_2);

    //tilde_r_3.show();
    //tilde_r_3.show_in_value();
    //tilde_r_3.show_in_deriv();

    //tilde_r_4.show();
    //tilde_r_4.show_in_value();
    //tilde_r_4.show_in_deriv();
}


TEST_F(TildeRTest, assignment_operator) {
    num_center_atomic_numbers = 2;
    center_atomic_numbers_lst[0] = 42;
    center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 2;
    neigh_atomic_numbers_lst[0] = 42;
    neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10;
    num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    
    matersdk::deepPotSE::TildeR<double> tilde_r_1;
    matersdk::deepPotSE::TildeR<double> tilde_r_2(
                                        neighbor_list,
                                        num_center_atomic_numbers,
                                        center_atomic_numbers_lst,
                                        num_neigh_atomic_numbers,
                                        neigh_atomic_numbers_lst,
                                        num_neigh_atoms_lst,
                                        rcut_smooth);
    
    matersdk::deepPotSE::TildeR<double> tilde_r_3;
    matersdk::deepPotSE::TildeR<double> tilde_r_4(
                                        neighbor_list,
                                        num_center_atomic_numbers,
                                        center_atomic_numbers_lst,
                                        num_neigh_atomic_numbers,
                                        neigh_atomic_numbers_lst,
                                        num_neigh_atoms_lst,
                                        rcut_smooth);
    
    tilde_r_3 = tilde_r_1;
    tilde_r_3 = tilde_r_2;
    tilde_r_4 = tilde_r_1;
    tilde_r_4 = tilde_r_2;
}


TEST_F(TildeRTest, get_num_center_atoms_and_neigh_atoms) {
    num_center_atomic_numbers = 2;
    center_atomic_numbers_lst[0] = 42;
    center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 2;
    neigh_atomic_numbers_lst[0] = 42;
    neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10;
    num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                        neighbor_list, 
                                        num_center_atomic_numbers,
                                        center_atomic_numbers_lst,
                                        num_neigh_atomic_numbers,
                                        neigh_atomic_numbers_lst,
                                        num_neigh_atoms_lst,
                                        rcut_smooth);
    
    EXPECT_EQ(tilde_r.get_num_center_atoms(), 12);
    EXPECT_EQ(tilde_r.get_num_neigh_atoms(), 25);
}



TEST_F(TildeRTest, generate_c2_n2) {
    num_center_atomic_numbers = 2;
    center_atomic_numbers_lst[0] = 42;
    center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 2;
    neigh_atomic_numbers_lst[0] = 42;
    neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10;
    num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                neighbor_list, 
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut_smooth);
    
    double*** tilde_r_value = tilde_r.generate();
    
    //tilde_r.show_in_value();
    
    matersdk::arrayUtils::free3dArray<double>(
                                        tilde_r_value, 
                                        tilde_r.get_num_center_atoms(),
                                        tilde_r.get_num_neigh_atoms());
}


TEST_F(TildeRTest, generate_c2_n1) {
    num_center_atomic_numbers = 2;
    center_atomic_numbers_lst[0] = 42;
    center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 1;
    neigh_atomic_numbers_lst[0] = 42;
    //neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10;
    //num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                neighbor_list, 
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut_smooth);
    
    double*** tilde_r_value = tilde_r.generate();
    
    //tilde_r.show_in_value();
    
    matersdk::arrayUtils::free3dArray<double>(
                                        tilde_r_value, 
                                        tilde_r.get_num_center_atoms(),
                                        tilde_r.get_num_neigh_atoms());
}


TEST_F(TildeRTest, generate_c1_n2) {
    num_center_atomic_numbers = 1;
    center_atomic_numbers_lst[0] = 42;
    //center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 2;
    neigh_atomic_numbers_lst[0] = 42;
    neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10;
    num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                neighbor_list, 
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut_smooth);
    
    double*** tilde_r_value = tilde_r.generate();
    
    //tilde_r.show_in_value();
    
    matersdk::arrayUtils::free3dArray<double>(
                                        tilde_r_value, 
                                        tilde_r.get_num_center_atoms(),
                                        tilde_r.get_num_neigh_atoms());
}


TEST_F(TildeRTest, deriv_c2_n2) {
    num_center_atomic_numbers = 2;
    center_atomic_numbers_lst[0] = 42;
    center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 2;
    neigh_atomic_numbers_lst[0] = 42;
    neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10; 
    num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                neighbor_list,
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut_smooth);
    
    double**** tilde_r_deriv = tilde_r.deriv();
    
    //tilde_r.show_in_deriv();

    matersdk::arrayUtils::free4dArray(
                            tilde_r_deriv,
                            tilde_r.get_num_center_atoms(),
                            tilde_r.get_num_neigh_atoms(),
                            4);
}


TEST_F(TildeRTest, deriv_c2_n1) {
    num_center_atomic_numbers = 2;
    center_atomic_numbers_lst[0] = 42;
    center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 1;
    neigh_atomic_numbers_lst[0] = 42;
    //neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10; 
    //num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                neighbor_list,
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut_smooth);
    
    double**** tilde_r_deriv = tilde_r.deriv();
    
    //tilde_r.show_in_deriv();

    matersdk::arrayUtils::free4dArray(
                            tilde_r_deriv,
                            tilde_r.get_num_center_atoms(),
                            tilde_r.get_num_neigh_atoms(),
                            4);
}


TEST_F(TildeRTest, deriv_c1_n2) {
    num_center_atomic_numbers = 1;
    center_atomic_numbers_lst[0] = 42;
    //center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 2;
    neigh_atomic_numbers_lst[0] = 42;
    neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10; 
    num_neigh_atoms_lst[1] = 15;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::NeighborList<double> neighbor_list(structure, rcut, pbc_xyz, true);
    matersdk::deepPotSE::TildeR<double> tilde_r(
                                neighbor_list,
                                num_center_atomic_numbers,
                                center_atomic_numbers_lst,
                                num_neigh_atomic_numbers,
                                neigh_atomic_numbers_lst,
                                num_neigh_atoms_lst,
                                rcut_smooth);
    
    double**** tilde_r_deriv = tilde_r.deriv();
    
    //tilde_r.show_in_deriv();

    matersdk::arrayUtils::free4dArray(
                            tilde_r_deriv,
                            tilde_r.get_num_center_atoms(),
                            tilde_r.get_num_neigh_atoms(),
                            4);
}




int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}