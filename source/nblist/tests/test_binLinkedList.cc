#include <gtest/gtest.h>
#include <iostream>

#include "../include/binLinkedList.h"


class SupercellTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    int scaling_matrix[3];

    static void SetUpTestSuite() {
        std::cout << "SupercellTest is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "SupercellTest is tearing down...\n";
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

        scaling_matrix[0] = 3;
        scaling_matrix[1] = 3;
        scaling_matrix[2] = 1;
    }


    void TearDown() override {
        
    }
}; // class: Supercell class



TEST_F(SupercellTest, default_constructor) {
    matersdk::Supercell<double> supercell;
    //supercell.show();
}


TEST_F(SupercellTest, constuctor_1) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    //supercell.show();
}


TEST_F(SupercellTest, copy_constructor) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::Supercell<double> supercell_1(structure, scaling_matrix);
    matersdk::Supercell<double> supercell_2;

    matersdk::Supercell<double> supercell_3(supercell_1);
    matersdk::Supercell<double> supercell_4(supercell_2);

    //supercell_3.show();
    //supercell_4.show();
}


TEST_F(SupercellTest, assignment_operator) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::Supercell<double> supercell_1(structure, scaling_matrix);
    matersdk::Supercell<double> supercell_(structure, scaling_matrix);
    matersdk::Supercell<double> supercell_2;
    
    matersdk::Supercell<double> supercell_3;
    matersdk::Supercell<double> supercell_4;

    supercell_3 = supercell_1;
    supercell_4 = supercell_2;
    //supercell_3.show();
    //supercell_4.show();

    //supercell_1 = supercell_2;
    //supercell_1.show();
}


TEST_F(SupercellTest, calc_prim_cell_idx_xyz_even) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    // Test 1: Scaling factor 是奇数
    scaling_matrix[0] = 5;
    scaling_matrix[1] = 7;
    scaling_matrix[2] = 9;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    // supercell.calc_prim_cell_idx_xyz();
    // supercell.calc_prim_cell_idx();
    const int* prim_cell_idx_xyz = supercell.get_prim_cell_idx_xyz();

    EXPECT_EQ(prim_cell_idx_xyz[0], 2);
    EXPECT_EQ(prim_cell_idx_xyz[1], 3);
    EXPECT_EQ(prim_cell_idx_xyz[2], 4);
}


TEST_F(SupercellTest, calc_prim_cell_idx_xyz_odd) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    // Test 2: Scaling factor 是偶数
    scaling_matrix[0] = 6;
    scaling_matrix[1] = 8;
    scaling_matrix[2] = 10;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    // supercell.calc_prim_cell_idx_xyz();
    // supercell.calc_prim_cell_idx();
    const int* prim_cell_idx_xyz = supercell.get_prim_cell_idx_xyz();
    EXPECT_EQ(prim_cell_idx_xyz[0], 2);
    EXPECT_EQ(prim_cell_idx_xyz[1], 3);
    EXPECT_EQ(prim_cell_idx_xyz[2], 4);
}


TEST_F(SupercellTest, calc_prim_cell_idx_xyz_even_odd) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    // Test 3: Scaling factor 既有奇数，也有偶数
    scaling_matrix[0] = 5;
    scaling_matrix[1] = 6;
    scaling_matrix[2] = 7;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    // supercell.calc_prim_cell_idx_xyz();
    // supercell.calc_prim_cell_idx();
    const int* prim_cell_idx_xyz = supercell.get_prim_cell_idx_xyz();
    EXPECT_EQ(prim_cell_idx_xyz[0], 2);
    EXPECT_EQ(prim_cell_idx_xyz[1], 2);
    EXPECT_EQ(prim_cell_idx_xyz[2], 3);
}


TEST_F(SupercellTest, calc_prim_cell_idx_even) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    // Test 1: Scaling factor 是奇数
    scaling_matrix[0] = 5;
    scaling_matrix[1] = 7;
    scaling_matrix[2] = 9;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    // supercell.calc_prim_cell_idx_xyz();
    // supercell.calc_prim_cell_idx();
    const int* prim_cell_idx_xyz = supercell.get_prim_cell_idx_xyz();
    const int prim_cell_idx = supercell.get_prim_cell_idx();

    EXPECT_EQ(
        prim_cell_idx,
        prim_cell_idx_xyz[0] + 
        prim_cell_idx_xyz[1] * scaling_matrix[0] + 
        prim_cell_idx_xyz[2] * scaling_matrix[0] * scaling_matrix[1]
    );
}


TEST_F(SupercellTest, calc_prim_cell_idx_odd) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    // Test 2: Scaling factor 是偶数
    scaling_matrix[0] = 6;
    scaling_matrix[1] = 8;
    scaling_matrix[2] = 10;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    // supercell.calc_prim_cell_idx_xyz();
    // supercell.calc_prim_cell_idx();
    const int* prim_cell_idx_xyz = supercell.get_prim_cell_idx_xyz();
    const int prim_cell_idx = supercell.get_prim_cell_idx();
    
    EXPECT_EQ(
        prim_cell_idx, 
        prim_cell_idx_xyz[0] + 
        prim_cell_idx_xyz[1] * scaling_matrix[0] + 
        prim_cell_idx_xyz[2] * scaling_matrix[0] * scaling_matrix[1]
    );
}


TEST_F(SupercellTest, calc_prim_cell_even_odd) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    // Test 3: Scaling factor 既有奇数，也有偶数
    scaling_matrix[0] = 5;
    scaling_matrix[1] = 6;
    scaling_matrix[2] = 7;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    // supercell.calc_prim_cell_idx_xyz();
    // supercell.calc_prim_cell_idx();
    const int* prim_cell_idx_xyz = supercell.get_prim_cell_idx_xyz();
    const int prim_cell_idx = supercell.get_prim_cell_idx();

    EXPECT_EQ(
        prim_cell_idx,
        prim_cell_idx_xyz[0] + 
        prim_cell_idx_xyz[1] * scaling_matrix[0] + 
        prim_cell_idx_xyz[2] * scaling_matrix[0] * scaling_matrix[1]
    );
}


TEST_F(SupercellTest, get_structure) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    scaling_matrix[0] = 7;
    scaling_matrix[1] = 8;
    scaling_matrix[2] = 9;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);

    //supercell.get_structure().show();
}


TEST_F(SupercellTest, get_scaling_matrix) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    scaling_matrix[0] = 7;
    scaling_matrix[1] = 8;
    scaling_matrix[2] = 18;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    const int* scaling_matrix = supercell.get_scaling_matrix();

    EXPECT_EQ(scaling_matrix[0], 7);
    EXPECT_EQ(scaling_matrix[1], 8);
    EXPECT_EQ(scaling_matrix[2], 18);
}



TEST_F(SupercellTest, get_num_atoms) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    scaling_matrix[0] = 6;
    scaling_matrix[1] = 8;
    scaling_matrix[2] = 10;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    
    EXPECT_EQ(supercell.get_prim_num_atoms(), 12);
    EXPECT_EQ(supercell.get_num_atoms(), 12 * scaling_matrix[0] * scaling_matrix[1] * scaling_matrix[2]);
}


TEST_F(SupercellTest, get_owned_atom_idxs) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    scaling_matrix[0] = 3;
    scaling_matrix[1] = 3;
    scaling_matrix[2] = 1;
    matersdk::Supercell<double> supercell(structure, scaling_matrix);
    /*
        72~83   84~95   96~107
        36~47   48~59   60~71
         0~11   12~23   24~35
    */
    const int* owned_atom_idxs = supercell.get_owned_atom_idxs();
    EXPECT_EQ(owned_atom_idxs[0], 48);
    EXPECT_EQ(owned_atom_idxs[1], 49);
    EXPECT_EQ(owned_atom_idxs[2], 50);
    EXPECT_EQ(owned_atom_idxs[3], 51);
    EXPECT_EQ(owned_atom_idxs[4], 52);
    EXPECT_EQ(owned_atom_idxs[5], 53);
    EXPECT_EQ(owned_atom_idxs[6], 54);
    EXPECT_EQ(owned_atom_idxs[7], 55);
    EXPECT_EQ(owned_atom_idxs[8], 56);
    EXPECT_EQ(owned_atom_idxs[9], 57);
    EXPECT_EQ(owned_atom_idxs[10], 58);
    EXPECT_EQ(owned_atom_idxs[11], 59);
}






class BinLinkedListTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];



    static void SetUpTestSuite() {
        std::cout << "BinLinkedListTest is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "BinLinkedListTest is tearing down...\n";
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

        rcut = 3.2;
        bin_size_xyz[0] = 2.8;
        bin_size_xyz[1] = 2.8;
        bin_size_xyz[2] = 2.8;
        pbc_xyz[0] = true;
        pbc_xyz[1] = true;
        pbc_xyz[2] = false;
    }


    void TearDown() override {

    }
};  // class BinLinkedListTest


TEST_F(BinLinkedListTest, default_constructor) {
    matersdk::BinLinkedList<double> bin_linked_list;
    //bin_linked_list.show();
}

TEST_F(BinLinkedListTest, constructor_1_case_1) {
    rcut = 3.0;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, bin_size_xyz, pbc_xyz);
    //bin_linked_list.show();

    // Step 1. 验证 `extending_matrix`, `scaling_matrix`
    const int* scaling_matrix = bin_linked_list.get_supercell().get_scaling_matrix();
    double* prim_interplanar_distances = (double *)structure.get_interplanar_distances();
    int* standard_scaling_matrix = (int*)malloc(sizeof(int) * 3);
    for (int ii=0; ii<3; ii++) {
        standard_scaling_matrix[ii] = std::ceil(rcut / prim_interplanar_distances[ii]);
        standard_scaling_matrix[ii] = standard_scaling_matrix[ii] * 2 + 1;

        if (pbc_xyz[ii] != true) 
            standard_scaling_matrix[ii] = 1;
    }
    EXPECT_EQ(scaling_matrix[0], standard_scaling_matrix[0]);
    EXPECT_EQ(scaling_matrix[1], standard_scaling_matrix[1]);
    EXPECT_EQ(scaling_matrix[2], standard_scaling_matrix[2]);


    // Step 2. 验证 `num_bin_xyz`
    const int* num_bin_xyz = bin_linked_list.get_num_bin_xyz();
    double* projected_lengths = (double*)bin_linked_list.get_supercell().get_structure().get_projected_lengths();
    int* standard_num_bin_xyz = (int*)malloc(sizeof(int) * 3);
    for (int ii=0; ii<3; ii++) {
        standard_num_bin_xyz[ii] = std::ceil( projected_lengths[ii] / bin_size_xyz[ii] );
    }
    EXPECT_EQ(num_bin_xyz[0], standard_num_bin_xyz[0]);
    EXPECT_EQ(num_bin_xyz[1], standard_num_bin_xyz[1]);
    EXPECT_EQ(num_bin_xyz[2], standard_num_bin_xyz[2]);


    // Step . Free memory
    free(prim_interplanar_distances);
    free(standard_scaling_matrix);
    free(projected_lengths);
    free(standard_num_bin_xyz);
}


TEST_F(BinLinkedListTest, constructor_2) {
    rcut = 3.0;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, pbc_xyz);
    //bin_linked_list.show();


    /*
    // Step 1. 验证 `extending_matrix`, `scaling_matrix`
    const int* scaling_matrix = bin_linked_list.get_supercell().get_scaling_matrix();
    double* prim_interplanar_distances = (double *)structure.get_interplanar_distances();
    int* standard_scaling_matrix = (int*)malloc(sizeof(int) * 3);
    for (int ii=0; ii<3; ii++) {
        standard_scaling_matrix[ii] = std::ceil(rcut / prim_interplanar_distances[ii]);
        standard_scaling_matrix[ii] = standard_scaling_matrix[ii] * 2 + 1;

        if (pbc_xyz[ii] != true) 
            standard_scaling_matrix[ii] = 1;
    }
    EXPECT_EQ(scaling_matrix[0], standard_scaling_matrix[0]);
    EXPECT_EQ(scaling_matrix[1], standard_scaling_matrix[1]);
    EXPECT_EQ(scaling_matrix[2], standard_scaling_matrix[2]);


    // Step 2. 验证 `num_bin_xyz`
    const int* num_bin_xyz = bin_linked_list.get_num_bin_xyz();
    double* projected_lengths = (double*)bin_linked_list.get_supercell().get_structure().get_projected_lengths();
    int* standard_num_bin_xyz = (int*)malloc(sizeof(int) * 3);
    for (int ii=0; ii<3; ii++) {
        standard_num_bin_xyz[ii] = std::ceil( projected_lengths[ii] / (rcut/2) );
    }
    EXPECT_EQ(num_bin_xyz[0], standard_num_bin_xyz[0]);
    EXPECT_EQ(num_bin_xyz[1], standard_num_bin_xyz[1]);
    EXPECT_EQ(num_bin_xyz[2], standard_num_bin_xyz[2]);


    // Step . Free memory
    free(prim_interplanar_distances);
    free(standard_scaling_matrix);
    free(projected_lengths);
    free(standard_num_bin_xyz);
    */
}


TEST_F(BinLinkedListTest, copy_constructor) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list_1(structure, rcut, bin_size_xyz, pbc_xyz);
    matersdk::BinLinkedList<double> bin_linked_list_2;

    matersdk::BinLinkedList<double> bin_linked_list_3(bin_linked_list_1);
    matersdk::BinLinkedList<double> bin_linked_list_4(bin_linked_list_2);
    //bin_linked_list_3.show();
    //bin_linked_list_4.show();
}


TEST_F(BinLinkedListTest, assignment_operator) {
    rcut = 3.0;
    bin_size_xyz[0] = 3.0;
    bin_size_xyz[1] = 3.0;
    bin_size_xyz[2] = 3.0;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list_1(structure, rcut, bin_size_xyz, pbc_xyz);
    matersdk::BinLinkedList<double> bin_linked_list_2;
    
    matersdk::BinLinkedList<double> bin_linked_list_3 = bin_linked_list_1;
    matersdk::BinLinkedList<double> bin_linked_list_4 = bin_linked_list_2;

    //bin_linked_list_2 = bin_linked_list_1;
    //bin_linked_list_2.show();
}


TEST_F(BinLinkedListTest, _build) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, bin_size_xyz, pbc_xyz);
    // bin_linked_list._build();
    //bin_linked_list.show();
}


TEST_F(BinLinkedListTest, get_bin_size) {
    rcut = 3.2;
    bin_size_xyz[0] = 2.8;
    bin_size_xyz[1] = 2.8;
    bin_size_xyz[2] = 2.8;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, bin_size_xyz, pbc_xyz);

    const double* bin_sizes = bin_linked_list.get_bin_size_xyz();

    EXPECT_EQ(bin_sizes[0], bin_size_xyz[0]);
    EXPECT_EQ(bin_sizes[1], bin_size_xyz[1]);
    EXPECT_EQ(bin_sizes[2], bin_size_xyz[2]);
}


TEST_F(BinLinkedListTest, get_bin_idx) {
    rcut = 3.2;
    bin_size_xyz[0] = 2.8;
    bin_size_xyz[1] = 2.8;
    bin_size_xyz[2] = 2.8;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;
    int prim_atom_idx = 3;

    // Step 1. 得到 `bin_idx`
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, bin_size_xyz, pbc_xyz);
    //bin_linked_list.show();

    int bin_idx = bin_linked_list.get_bin_idx(prim_atom_idx);

    // Step 2. 得到 `num_bin_xyz`
    const int* num_bin_xyz = bin_linked_list.get_num_bin_xyz();
    //printf("num_bin_xyz = [%d, %d, %d]\n", num_bin_xyz[0], num_bin_xyz[1], num_bin_xyz[2]);

    // Step 3. 得到 `bin_idx_xyz`
    int bin_idx_xyz[3];
    bin_idx_xyz[0] = bin_idx % num_bin_xyz[0];
    bin_idx_xyz[1] = (bin_idx / num_bin_xyz[0]) % num_bin_xyz[1];
    bin_idx_xyz[2] = bin_idx / (num_bin_xyz[0] * num_bin_xyz[1]);
    //printf("bin_idx_xyz = [%d, %d, %d]\n", bin_idx_xyz[0], bin_idx_xyz[1], bin_idx_xyz[2]);

    // Step 4.
    // Step 4.1. 得到 `cart_coords[atom_idx]` -- `atom_idx` 对应 `prim_atom_idx`
    int atom_idx = (
        prim_atom_idx + 
        bin_linked_list.get_supercell().get_prim_num_atoms() * bin_linked_list.get_supercell().get_prim_cell_idx()
    );
    const double* cart_coord = bin_linked_list.get_supercell().get_structure().get_cart_coords()[atom_idx];
    const double* min_limit_xyz = bin_linked_list.get_min_limit_xyz();
    const double* bin_size_xyz = bin_linked_list.get_bin_size_xyz();

    int standard_bin_idx_xyz[3];
    standard_bin_idx_xyz[0] = ( cart_coord[0] - min_limit_xyz[0] ) / bin_size_xyz[0];
    standard_bin_idx_xyz[1] = ( cart_coord[1] - min_limit_xyz[1] ) / bin_size_xyz[1];
    standard_bin_idx_xyz[2] = ( cart_coord[2] - min_limit_xyz[2] ) / bin_size_xyz[2];
    //printf("standard_bin_idx_xyz = [%d, %d, %d]\n", standard_bin_idx_xyz[0], standard_bin_idx_xyz[1], standard_bin_idx_xyz[2]);


    EXPECT_EQ(bin_idx_xyz[0], standard_bin_idx_xyz[0]);
    EXPECT_EQ(bin_idx_xyz[1], standard_bin_idx_xyz[1]);
    EXPECT_EQ(bin_idx_xyz[2], standard_bin_idx_xyz[2]);

    // Step .Free memory
}


TEST_F(BinLinkedListTest, get_num_neigh_bins) {
    rcut = 9.0;
    bin_size_xyz[0] = 3.0;
    bin_size_xyz[1] = 3.0;
    bin_size_xyz[2] = 3.0;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, bin_size_xyz, pbc_xyz);

    EXPECT_EQ(bin_linked_list.get_num_neigh_bins(), 343);
}


TEST_F(BinLinkedListTest, get_neigh_bins) {
    rcut = 9.0;
    bin_size_xyz[0] = 3.0;
    bin_size_xyz[1] = 3.0;
    bin_size_xyz[2] = 3.0;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;

    // Step 1. 初始化 structure, bin_linked_list
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);    
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, bin_size_xyz, pbc_xyz);

    // Step 1.1. 得到 prim_atom_idx 对应的 atom_idx
    int center_bin_idx = bin_linked_list.get_bin_idx(0);
    printf("center_bin_idx = %d\n", center_bin_idx);

    // Step 1.2. 得到 prim_atom_idx 对应的 atom_idx 的所有近邻 bins
    int summation = 0;
    int* neigh_bin_idxs = bin_linked_list.get_neigh_bins(0);
    int exist_neigh_bin_idxs[7];
    exist_neigh_bin_idxs[0] = center_bin_idx - 3;
    exist_neigh_bin_idxs[1] = center_bin_idx - 2;
    exist_neigh_bin_idxs[2] = center_bin_idx - 1;
    exist_neigh_bin_idxs[3] = center_bin_idx;
    exist_neigh_bin_idxs[4] = center_bin_idx + 1;
    exist_neigh_bin_idxs[5] = center_bin_idx + 2;
    exist_neigh_bin_idxs[6] = center_bin_idx + 3;
    for (int ii=0; ii<343; ii++) {
        for (int jj=0; jj<7; jj++) {
            if (neigh_bin_idxs[ii] == exist_neigh_bin_idxs[jj])
                summation++;
        }
    }
    EXPECT_EQ(summation, 7);

    // Step . Free memory
    free(neigh_bin_idxs);
}


TEST_F(BinLinkedListTest, get_neigh_atoms) {
    rcut = 2.8;
    bin_size_xyz[0] = 2.8;
    bin_size_xyz[1] = 2.8;
    bin_size_xyz[2] = 2.8;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false;

    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, bin_size_xyz, pbc_xyz);
    std::vector<int> neigh_atom_idxs = bin_linked_list.get_neigh_atoms(0);
    //for (int neigh_atom_idx: neigh_atom_idxs) {
        //printf("%d, ", neigh_atom_idx);
    //}
    //printf("\n");
}


TEST_F(BinLinkedListTest, get_supercell) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::BinLinkedList<double> bin_linked_list(structure, rcut, bin_size_xyz, pbc_xyz);

}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}