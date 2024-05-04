#include <chrono>
#include <iostream>

#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../include/se.h"
#include "../../../core/include/arrayUtils.h"


int main() {
    // Step 1. Declaration
    int num_atoms;
    double basis_vectors[3][3];
    int atomic_numbers[12];
    double frac_coords[12][3];
    double rcut;
    double bin_size_xyz[3];
    bool pbc_xyz[3];  

    int num_center_atomic_numbers;
    int center_atomic_numbers_lst[2];
    int num_neigh_atomic_numbers;
    int neigh_atomic_numbers_lst[2];
    int num_neigh_atoms_lst[2];
    double rcut_smooth;


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

    rcut = 3.3;
    bin_size_xyz[0] = 3.0;
    bin_size_xyz[1] = 3.0;
    bin_size_xyz[2] = 3.0;
    pbc_xyz[0] = true;
    pbc_xyz[1] = true;
    pbc_xyz[2] = false; 

    num_center_atomic_numbers = 2;
    center_atomic_numbers_lst[0] = 42;
    center_atomic_numbers_lst[1] = 16;
    num_neigh_atomic_numbers = 2;
    neigh_atomic_numbers_lst[0] = 42;
    neigh_atomic_numbers_lst[1] = 16;
    num_neigh_atoms_lst[0] = 10; 
    num_neigh_atoms_lst[1] = 15;

    // Step 3. 构建 matersdk::Structure 对象
    matersdk::Structure<double> structure(
                    num_atoms,          // 体系的原子数
                    basis_vectors,      // 基矢:  3*3的数组
                    atomic_numbers,     // 原子序数: 长度等于体系的原子数
                    frac_coords,        // num_atoms * 3 的数组
                    false               // coords 是否为笛卡尔坐标
    );
    const int scaling_matrix[3] = {1000, 100, 1};
    structure.make_supercell(scaling_matrix);



    auto start = std::chrono::high_resolution_clock::now();
    // Step 4. 构建 matersdk::NeighborList 对象
    matersdk::NeighborList<double> neighbor_list(
                    structure,          // Structure 对象
                    rcut,               // 截断半径
                    pbc_xyz,            // [true, true, false]，在各个方向上周期性
                    false                // 是否按照近邻原子的距离排序
    );
    
    
    // Step 5. 构建 matersdk::deepPotSE::TildeR 对象
    matersdk::deepPotSE::TildeR<double> tilde_r(
                    neighbor_list,          // neighbor_list 对象
                    num_center_atomic_numbers,   
                    center_atomic_numbers_lst,
                    num_neigh_atomic_numbers,      
                    neigh_atomic_numbers_lst,
                    num_neigh_atoms_lst,        
                    rcut_smooth);
    double*** tilde_r_value = tilde_r.generate();

    matersdk::arrayUtils::free3dArray(tilde_r_value, tilde_r.get_num_center_atoms(), tilde_r.get_num_neigh_atoms());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "Program took " << duration << " seconds." << std::endl;

    //tilde_r.show();
    //tilde_r.show_in_value();           // 输出特征的值
    //printf("\n\n");
    //tilde_r.show_in_deriv();           // 输出特征的导数
}