#include <gtest/gtest.h>
#include <iostream>
#include <stdlib.h>

#include "../include/structure.h"


// Part I. 
class StructureArrayTest : public ::testing::Test {
protected:
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];


    static void SetUpTestSuite() {
        std::cout << "StructureArrayTest is setting up...\n";
    }


    static void TearDownTestSuite() {
        std::cout << "StructureArrayTest is tearing down...\n";
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
    };
    


    void TearDown() override {
        
    }
};


//TEST_F(StructureArrayTest, default_constructor) {
//    matersdk::Structure<double> structure;
//}


TEST_F(StructureArrayTest, init) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    //structure.show();
}


TEST_F(StructureArrayTest, copy_constructor) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);

    matersdk::Structure<double> structure_1(structure);     // Note: You should init `this->num_atoms` in private region
    //structure_1.show();
}


TEST_F(StructureArrayTest, copy_assignment) {
    matersdk::Structure<double> structure_1(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::Structure<double> structure_2;
    matersdk::Structure<double> structure_3;

    structure_2 = structure_1;
    //structure_2.show();

    structure_2 = structure_3;
    //structure_3.show();
}


TEST_F(StructureArrayTest, calc_cart_coords_array) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    //structure.show();
}


TEST_F(StructureArrayTest, make_supercell) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    int scaling_matrix[3] = {3, 3, 1};
    structure.make_supercell(scaling_matrix);
    //structure.show();
}


TEST_F(StructureArrayTest, get_num_atoms) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    EXPECT_EQ(structure.get_num_atoms(), 12);

    int scaling_matrix[3] = {3, 3, 1};
    structure.make_supercell(scaling_matrix);
    EXPECT_EQ(structure.get_num_atoms(), 108);
}


TEST_F(StructureArrayTest, get_basis_vectors) {
    matersdk::Structure<double> structure_1(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    const double** basis_vectors_1 = structure_1.get_basis_vectors();
    
    EXPECT_EQ(basis_vectors_1[0][0], basis_vectors[0][0]);
    EXPECT_EQ(basis_vectors_1[0][1], basis_vectors[0][1]);
    EXPECT_EQ(basis_vectors_1[0][2], basis_vectors[0][2]);
    EXPECT_EQ(basis_vectors_1[1][0], basis_vectors[1][0]);
    EXPECT_EQ(basis_vectors_1[1][1], basis_vectors[1][1]);
    EXPECT_EQ(basis_vectors_1[1][2], basis_vectors[1][2]);
    EXPECT_EQ(basis_vectors_1[2][0], basis_vectors[2][0]);
    EXPECT_EQ(basis_vectors_1[2][1], basis_vectors[2][1]);
    EXPECT_EQ(basis_vectors_1[2][2], basis_vectors[2][2]);


    matersdk::Structure<double> structure_2;
    const void* basis_vectors_2 = structure_2.get_basis_vectors();
    EXPECT_EQ(basis_vectors_2, nullptr);
}


TEST_F(StructureArrayTest, get_basis_vectors_4_supercell) {
    matersdk::Structure<double> prim_structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    int scaling_matrix[3] = {3, 3, 1};    

    // Step 1. 记录原胞的 basis_vectors
    const double** prim_basis_vectors = prim_structure.get_basis_vectors(); 

    // Step 2. make_supercell()， 记录超胞的 basis_vectors
    structure.make_supercell(scaling_matrix);
    const double** supercell_basis_vectors = structure.get_basis_vectors();
    
    // Step 3. 
    EXPECT_EQ(supercell_basis_vectors[0][0], prim_basis_vectors[0][0] * scaling_matrix[0]);
    EXPECT_EQ(supercell_basis_vectors[0][1], prim_basis_vectors[0][1] * scaling_matrix[0]);
    EXPECT_EQ(supercell_basis_vectors[0][2], prim_basis_vectors[0][2] * scaling_matrix[0]);
    EXPECT_EQ(supercell_basis_vectors[1][0], prim_basis_vectors[1][0] * scaling_matrix[1]);
    EXPECT_EQ(supercell_basis_vectors[1][1], prim_basis_vectors[1][1] * scaling_matrix[1]);
    EXPECT_EQ(supercell_basis_vectors[1][2], prim_basis_vectors[1][2] * scaling_matrix[1]);
    EXPECT_EQ(supercell_basis_vectors[2][0], prim_basis_vectors[2][0] * scaling_matrix[2]);
    EXPECT_EQ(supercell_basis_vectors[2][1], prim_basis_vectors[2][1] * scaling_matrix[2]);
    EXPECT_EQ(supercell_basis_vectors[2][2], prim_basis_vectors[2][2] * scaling_matrix[2]);
}


TEST_F(StructureArrayTest, get_atomic_numbers) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    const int* atomic_numbers_1 = structure.get_atomic_numbers();
    for (int ii=0; ii<structure.get_num_atoms(); ii++) {
        EXPECT_EQ(atomic_numbers_1[ii], atomic_numbers[ii]);
    }

    // int* atomic_numbers_1_nonconst = (int*)structure.get_atomic_numbers();
    // free(atomic_numbers_1_nonconst);


    matersdk::Structure<double> structure_2;
    const void* atomic_numbers_2 = structure_2.get_atomic_numbers();
    EXPECT_EQ(atomic_numbers_2, nullptr);
}


TEST_F(StructureArrayTest, get_cart_coords) {
    matersdk::Structure<double> structure_1(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    const double** cart_coords_1 = structure_1.get_cart_coords();
    //structure.show();
    for (int ii=0; ii<structure_1.get_num_atoms(); ii++) {
        //printf("%-12.6f\t%-12.6f\t%-12.6f\n", cart_coords_[ii][0], cart_coords_[ii][1], cart_coords_[ii][2]);
    }

    matersdk::Structure<double> structure_2;
    EXPECT_EQ(structure_2.get_cart_coords(), nullptr);

}


TEST_F(StructureArrayTest, get_projected_lengths) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    
    // Step 1. 得到 `projected_lengths`
    double* projected_lengths = structure.get_projected_lengths();
    const double** basis_vectors = structure.get_basis_vectors();
    //printf("basis_vectors:\n");
    //for (int ii=0; ii<3; ii++)
    //    printf("[%15f, %15f, %15f]\n", structure.get_basis_vectors()[ii][0], structure.get_basis_vectors()[ii][1], structure.get_basis_vectors()[ii][2]);

    //printf("projected_lengths:\n");
    //printf("[%15f, %15f, %15f]\n", projected_lengths[0], projected_lengths[1], projected_lengths[2]);


    // Step 2. 得到 `standard_projected_lengths`
    double* standard_projected_lengths = (double*)malloc(sizeof(double) * 3);
    double** limit_xyz = structure.get_limit_xyz();
    standard_projected_lengths[0] = limit_xyz[0][1] - limit_xyz[0][0];
    standard_projected_lengths[1] = limit_xyz[1][1] - limit_xyz[1][0];
    standard_projected_lengths[2] = limit_xyz[2][1] - limit_xyz[2][0];

    // Step 3. 判断
    // Step 3.1. Way 1
    EXPECT_DOUBLE_EQ(
        projected_lengths[0], 
        std::abs(basis_vectors[0][0]) + std::abs(basis_vectors[1][0]) + std::abs(basis_vectors[2][0])
    );
    EXPECT_DOUBLE_EQ(
        projected_lengths[1],
        std::abs(basis_vectors[0][1]) + std::abs(basis_vectors[1][1]) + std::abs(basis_vectors[1][2])
    );
    EXPECT_DOUBLE_EQ(
        projected_lengths[2],
        std::abs(basis_vectors[0][2]) + std::abs(basis_vectors[1][2]) + std::abs(basis_vectors[2][2])
    );
    // Step 3.2. Way 2
    EXPECT_DOUBLE_EQ(standard_projected_lengths[0], projected_lengths[0]);
    EXPECT_DOUBLE_EQ(standard_projected_lengths[1], projected_lengths[1]);
    EXPECT_DOUBLE_EQ(standard_projected_lengths[2], projected_lengths[2]);

    // Step . Free memory
    free(projected_lengths);
    free(standard_projected_lengths);
    for (int ii=0; ii<3; ii++) 
        free(limit_xyz[ii]);
    free(limit_xyz);

    matersdk::Structure<double> structure_2;
    EXPECT_EQ(structure_2.get_projected_lengths(), nullptr);
}


TEST_F(StructureArrayTest, get_projected_lengths4supercell) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    int scaling_matrix[3] = {3, 3, 1};
    structure.make_supercell(scaling_matrix);
    double* projected_lengths = structure.get_projected_lengths();

    // Step 1. 使用 `limit_xyz` 计算 `standard_projected_lengths`
    double* standard_projected_lengths = (double*)malloc(sizeof(double) * 3);
    double** limit_xyz = structure.get_limit_xyz();
    for (int ii=0; ii<3; ii++) {
        standard_projected_lengths[ii] = limit_xyz[ii][1] - limit_xyz[ii][0];
    }

    EXPECT_EQ(projected_lengths[0], standard_projected_lengths[0]);
    EXPECT_EQ(projected_lengths[1], standard_projected_lengths[1]);
    EXPECT_EQ(projected_lengths[2], standard_projected_lengths[2]);

    // Step . Free memory
    free(projected_lengths);
    for (int ii=0; ii<3; ii++) {
        free(limit_xyz[ii]);
    }
    free(limit_xyz);
    free(standard_projected_lengths);
}



TEST_F(StructureArrayTest, get_interplanar_distances) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    const double* interplanar_distances = structure.get_interplanar_distances();

    //printf("Interplanar distances:\n");
    //printf("[%15f, %15f, %15f]\n", interplanar_distances[0], interplanar_distances[1], interplanar_distances[2]);

    double* interplanar_distances_noconst = (double*)interplanar_distances;
    free(interplanar_distances_noconst);

    matersdk::Structure<double> structure_2;
    EXPECT_EQ(structure_2.get_interplanar_distances(), nullptr);
}


TEST_F(StructureArrayTest, get_interplanar_distances4supercell) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    double* interplanar_distances = structure.get_interplanar_distances();

    int scaling_matrix[3] = {3, 3, 1};
    structure.make_supercell(scaling_matrix);
    double* supercell_interplanar_distances = structure.get_interplanar_distances();

    EXPECT_DOUBLE_EQ(supercell_interplanar_distances[0], interplanar_distances[0] * (double)scaling_matrix[0]);
    EXPECT_DOUBLE_EQ(supercell_interplanar_distances[1], interplanar_distances[1] * (double)scaling_matrix[1]);
    EXPECT_DOUBLE_EQ(supercell_interplanar_distances[2], interplanar_distances[2] * (double)scaling_matrix[2]);


    // Step . Free memory
    free(interplanar_distances);
    free(supercell_interplanar_distances);
}


TEST_F(StructureArrayTest, get_pseudo_origin) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    const double* pseudo_origin = structure.get_pseudo_origin();

    EXPECT_DOUBLE_EQ(pseudo_origin[0], 0);
    EXPECT_DOUBLE_EQ(pseudo_origin[1], 0);
    EXPECT_DOUBLE_EQ(pseudo_origin[2], 0);
}


TEST_F(StructureArrayTest, get_pseudo_origin4supercell) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    int scaling_matrix[3] = {3, 3, 1};
    // Note: structure.make_supercell(scaling_matrix) 之后，`prim_basis_vectors_` 也会发生变化
    const double** prim_basis_vectors_ = structure.get_basis_vectors();
    double prim_basis_vectors[3][3];
    for (int ii=0; ii<3; ii++) {
        for (int jj=0; jj<3; jj++) {
            prim_basis_vectors[ii][jj] = prim_basis_vectors_[ii][jj];
        }
    }

    // Step 1. make_supercell
    structure.make_supercell(scaling_matrix);
    // Step 1.1. 
    const double* pseudo_origin = structure.get_pseudo_origin();

    // Step 2. 
    // Step 2.1. 根据 `scaling_matrix` 获取 `range`
    int range[3][2];
    for (int ii=0; ii<3; ii++) {
        if (scaling_matrix[ii] % 2 ==0) {   // 偶数
            range[ii][0] = -(scaling_matrix[ii] / 2 - 1);
            range[ii][1] = scaling_matrix[ii] / 2;
        } else {    // 奇数
            range[ii][0] = -(scaling_matrix[ii] - 1) / 2;
            range[ii][1] = (scaling_matrix[ii] - 1) / 2;
        }
    }

    // Step 2.2. 得到 `standard_pseudo_origin`
    double standard_pseudo_origin[3];
    for (int ii=0; ii<3; ii++) {
        standard_pseudo_origin[ii] = (
            range[0][0] * prim_basis_vectors[0][ii] + 
            range[1][0] * prim_basis_vectors[1][ii] + 
            range[2][0] * prim_basis_vectors[2][ii]
        );
    }

    //printf("pseudo_origin = [%f, %f, %f]\n", pseudo_origin[0], pseudo_origin[1], pseudo_origin[2]);
    //printf("standard_pseudo_origin = [%f, %f, %f]\n", standard_pseudo_origin[0], standard_pseudo_origin[1], standard_pseudo_origin[2]);
    EXPECT_DOUBLE_EQ(pseudo_origin[0], standard_pseudo_origin[0]);
    EXPECT_DOUBLE_EQ(pseudo_origin[1], standard_pseudo_origin[1]);
    EXPECT_DOUBLE_EQ(pseudo_origin[2], standard_pseudo_origin[2]);
}


TEST_F(StructureArrayTest, get_vertexes) {
    // Step 1.
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    double** vertexes = structure.get_vertexes();
    
    printf("scaling_matrix[3] = {1, 1, 1}. Primitive cell's Vertexes:\n");
    for (int ii=0; ii<8; ii++)
        printf("\t[%f, %f, %f]\n", vertexes[ii][0], vertexes[ii][1], vertexes[ii][2]);
    
    // Step . Free memory
    for (int ii=0; ii<8; ii++) {
        free(vertexes[ii]);
    }
    free(vertexes);
}


TEST_F(StructureArrayTest, get_vertexes4supercell) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    int scaling_matix[3] = {3, 3, 1};
    structure.make_supercell(scaling_matix);
    
    double** vertexes = structure.get_vertexes();
    printf("scaling_matrix[3] = {3, 3, 1}. Supercell's Vertexes:\n");
    for (int ii=0; ii<8; ii++) {
        printf("\t[%f, %f, %f]\n", vertexes[ii][0], vertexes[ii][1], vertexes[ii][2]);
    }

    // Step . Free memory
    for (int ii=0; ii<8; ii++) {
        free(vertexes[ii]);
    }
    free(vertexes);
}


TEST_F(StructureArrayTest, get_limit_xyz) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    double** limit_xyz = structure.get_limit_xyz();

    printf("scaling_matrix[3] = {1, 1, 1}. Primitive cell's limit_xyz:\n");
    printf("\t[%f, %f]\n", limit_xyz[0][0], limit_xyz[0][1]);
    printf("\t[%f, %f]\n", limit_xyz[1][0], limit_xyz[1][1]);
    printf("\t[%f, %f]\n", limit_xyz[2][0], limit_xyz[2][1]);

    for (int ii=0; ii<3; ii++) {
        free(limit_xyz[ii]);
    }
    free(limit_xyz);
}


TEST_F(StructureArrayTest, get_limit_xyz4supercell) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    int scaling_matrix[3] = {3, 3, 1};
    structure.make_supercell(scaling_matrix);
    double** limit_xyz = structure.get_limit_xyz();

    printf("scaling_matrix[3] = {3, 3, 1}. Primitive cell's limit_xyz:\n");
    printf("\t[%f, %f]\n", limit_xyz[0][0], limit_xyz[0][1]);
    printf("\t[%f, %f]\n", limit_xyz[1][0], limit_xyz[1][1]);
    printf("\t[%f, %f]\n", limit_xyz[2][0], limit_xyz[2][1]);

    for (int ii=0; ii<3; ii++) {
        free(limit_xyz[ii]);
    }
    free(limit_xyz);
}


// Part II.
class StructurePointerTest : public ::testing::Test {
protected:
    int num_atoms;
    double **basis_vectors;
    int *atomic_numbers;
    double **frac_coords;

    static void SetUpTestSuite() {
        std::cout << "StructurePointerTest is setting up...\n";
    }

    static void TearDownSuite() {
        std::cout << "StructurePointerTest is tearing down...\n";
    }


    void SetUp() override {

        // Step 1. Allocate memory 
        num_atoms = 12;
        basis_vectors = (double**)malloc(sizeof(double*) * 3);
        for (int ii=0; ii<3; ii++) {
            basis_vectors[ii] = (double*)malloc(sizeof(double) * 3);
        }
        atomic_numbers = (int*)malloc(sizeof(int) * num_atoms);
        frac_coords = (double**)malloc(sizeof(double*) * num_atoms);
        for (int ii=0; ii<num_atoms; ii++) {
            frac_coords[ii] = (double*)malloc(sizeof(double) * 3);
        }
        // Step 2. Assign
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

    }


    void TearDown() override {
        for (int ii=0; ii<3; ii++) {
            free(basis_vectors[ii]);
        }
        free(basis_vectors);
        free(atomic_numbers);
        for (int ii=0; ii<num_atoms; ii++) {
            free(frac_coords[ii]);
        }
        free(frac_coords);
    }

};  // class StructurePointerTest


TEST_F(StructurePointerTest, init) {
    matersdk::Structure<double> structure(num_atoms, basis_vectors, atomic_numbers, frac_coords, false);
    //structure.show();
}



// Part III. 
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}