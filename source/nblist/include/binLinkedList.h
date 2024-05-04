#ifndef MATERSDK_BIN_LINKED_LIST_H
#define MATERSDK_BIN_LINKED_LIST_H


#include <vector>
#include <cassert>
#include "./structure.h"
#include "../../core/include/vec3Operation.h"


namespace matersdk {

const double EPSILON = 1e-3;

template <typename CoordType>
class BinLinkedList;


template <typename CoordType>
class BasicStructureInfo {
public:
    BasicStructureInfo();

    BasicStructureInfo(Structure<CoordType> structure);

    BasicStructureInfo& operator=(const BasicStructureInfo& rhs);

    BasicStructureInfo(const BasicStructureInfo& rhs);

    ~BasicStructureInfo();

    void show() const;

    friend class Supercell<CoordType>;
    friend class BinLinkedList<CoordType>;

private:
    int num_atoms = 0;
    CoordType projected_lengths[3] = {0, 0, 0};
    CoordType interplanar_distances[3] = {0, 0, 0};
};



template <typename CoordType>
class Supercell {
public:
    Supercell();

    Supercell(Structure<CoordType> structure, int *scaling_matrix); // Note `Structure<CoordType> &structure` is a reference.

    //Supercell(Structure<CoordType> Structure, int scaling_matrix[3]);
    
    Supercell(const Supercell &rhs);

    Supercell& operator=(const Supercell &rhs);

    ~Supercell();

    void calc_prim_cell_idx_xyz();

    void calc_prim_cell_idx();      // Call this function after `this->calc_prim_cell_idx_xyz()`

    void calc_owned_atom_idxs();    // Call this function after `this->`

    void show() const;              // Print out information

    const Structure<CoordType>& get_structure() const;

    const int* get_scaling_matrix() const;

    const int get_prim_num_atoms() const;

    const int* get_prim_cell_idx_xyz() const;

    const int get_prim_cell_idx() const;

    const int get_num_atoms() const;

    const int* get_owned_atom_idxs() const;

    friend class BinLinkedList<CoordType>;


private:
    Structure<CoordType> structure;
    BasicStructureInfo<CoordType> prim_structure_info;
    int scaling_matrix[3] = {1, 1, 1};      // 扩包倍数；x, y, z 方向上的 primitive_cell 个数
    int num_atoms = 0;
    int prim_cell_idx = 0;          // primitive cell 对应的 cell index
    int prim_cell_idx_xyz[3] = {0, 0, 0};   // 
    int *owned_atom_idxs;           // 
}; // class: Supercell




template <typename CoordType>
class BinLinkedList {
public:
    BinLinkedList();

    BinLinkedList(Structure<CoordType> structure, CoordType rcut, CoordType* bin_size_xyz, bool* pbc_xyz);

    BinLinkedList(Structure<CoordType> structure, CoordType rcut, bool* pbc_xyz);

    BinLinkedList(const BinLinkedList& rhs);

    BinLinkedList& operator=(const BinLinkedList& rhs);

    ~BinLinkedList();

    void _build();

    int get_bin_idx(int prim_atom_idx) const;

    int get_num_neigh_bins() const;

    int* get_neigh_bins(int prim_atom_idx) const;

    std::vector<int> get_neigh_atoms(int prim_atom_idx) const;  // Unfinished!

    void show() const;

    const Supercell<CoordType>& get_supercell() const;

    const CoordType* get_bin_size_xyz() const;

    const int* get_num_bin_xyz() const;

    const CoordType* get_min_limit_xyz() const;

private:
    Supercell<CoordType> supercell;             // 超胞 `matersdk::Supercell 对象`
    CoordType rcut = 0;                         // 截断半径
    CoordType bin_size_xyz[3] = {0, 0, 0};      // bin 在 x, y, z 方向上的尺寸
    bool pbc_xyz[3] = {false, false, false};
    int num_bin_xyz[3] = {0, 0, 0};             // bin 在 x, y, z 方向上的数量
    CoordType min_limit_xyz[3] = {0, 0, 0};     // box 在 x, y, z 方向上的最小坐标值
    int* heads_lst;                             // length = number of bins.
    int* nexts_lst;                             // length = number of atoms in supercell.

};  // class BinLinkedList








/**
 * @brief Construct a new Basic Structure Info< Coord Type>:: Basic Structure Info object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
BasicStructureInfo<CoordType>::BasicStructureInfo() {
    this->num_atoms = 0;

    this->projected_lengths[0] = 0;
    this->projected_lengths[1] = 0;
    this->projected_lengths[2] = 0;

    this->interplanar_distances[0] = 0;
    this->interplanar_distances[1] = 0;
    this->interplanar_distances[2] = 0;
}


/**
 * @brief Construct a new Basic Structure Info< Coord Type>:: Basic Structure Info object
 * 
 * @tparam CoordType 
 * @param structure 
 */
template <typename CoordType>
BasicStructureInfo<CoordType>::BasicStructureInfo(Structure<CoordType> structure) {
    this->num_atoms = structure.num_atoms;

    if (this->num_atoms == 0) {
        this->projected_lengths[0] = 0;
        this->projected_lengths[1] = 0;
        this->projected_lengths[2] = 0;

        this->interplanar_distances[0] = 0;
        this->interplanar_distances[1] = 0;
        this->interplanar_distances[2] = 0;
    } else {
        CoordType* projected_lengths = (CoordType*)structure.get_projected_lengths();
        CoordType* interplanar_distances_nonconst = (CoordType*)structure.get_interplanar_distances();
        
        this->projected_lengths[0] = projected_lengths[0];
        this->projected_lengths[1] = projected_lengths[1];
        this->projected_lengths[2] = projected_lengths[2];
        this->interplanar_distances[0] = interplanar_distances_nonconst[0];
        this->interplanar_distances[1] = interplanar_distances_nonconst[1];
        this->interplanar_distances[2] = interplanar_distances_nonconst[2];

        free(projected_lengths);
        free(interplanar_distances_nonconst);
    }
}


/**
 * @brief Construct a new Basic Structure Info< Coord Type>:: Basic Structure Info object
 * 
 * @tparam CoordType 
 * @param rhs 
 */
template <typename CoordType>
BasicStructureInfo<CoordType>::BasicStructureInfo(const BasicStructureInfo& rhs) {
    this->num_atoms = rhs.num_atoms;
    
    if (this->num_atoms == 0) {
        this->projected_lengths[0] = 0;
        this->projected_lengths[1] = 0;
        this->projected_lengths[2] = 0;
        this->interplanar_distances[0] = 0;
        this->interplanar_distances[1] = 0;
        this->interplanar_distances[2] = 0;
    } else {
        this->projected_lengths[0] = rhs.projected_lengths[0];
        this->projected_lengths[1] = rhs.projected_lengths[1];
        this->projected_lengths[2] = rhs.projected_lengths[2];
        this->interplanar_distances[0] = rhs.interplanar_distances[0];
        this->interplanar_distances[1] = rhs.interplanar_distances[1];
        this->interplanar_distances[2] = rhs.interplanar_distances[2];
    }
}


/**
 * @brief Overload assignment operator.
 * 
 * @tparam CoordType 
 * @param rhs 
 * @return BasicStructureInfo<CoordType>& 
 */
template <typename CoordType>
BasicStructureInfo<CoordType>& BasicStructureInfo<CoordType>::operator=(const BasicStructureInfo& rhs) {
    this->num_atoms = rhs.num_atoms;

    if (this->num_atoms == 0) {
        this->projected_lengths[0] = 0;
        this->projected_lengths[1] = 0;
        this->projected_lengths[2] = 0;
        this->interplanar_distances[0] = 0;
        this->interplanar_distances[1] = 0;
        this->interplanar_distances[2] = 0;
    } else {
        this->projected_lengths[0] = rhs.projected_lengths[0];
        this->projected_lengths[1] = rhs.projected_lengths[1];
        this->projected_lengths[2] = rhs.projected_lengths[2];
        this->interplanar_distances[0] = rhs.interplanar_distances[0];
        this->interplanar_distances[1] = rhs.interplanar_distances[1];
        this->interplanar_distances[2] = rhs.interplanar_distances[2];
    }
}


/**
 * @brief Destroy the Basic Structure Info< Coord Type>:: Basic Structure Info object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
BasicStructureInfo<CoordType>::~BasicStructureInfo() {
    if (this->num_atoms != 0)
        this->num_atoms = 0;
}


/**
 * @brief Print out the information about `BasicStructureInfo`.
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
void BasicStructureInfo<CoordType>::show() const{
    printf("num_atoms = %15d\n", this->num_atoms);
    if (this->num_atoms != 0) {
        printf("Projected Lengths : \n");
        printf("[%15f, %15f, %15f]\n", this->projected_lengths[0], this->projected_lengths[1], this->projected_lengths[2]);
        printf("Inter Planar Distances : \n");
        printf("[%15f, %15f, %15f]\n", this->interplanar_distances[0], this->interplanar_distances[1], this->interplanar_distances[2]);
    } else {
        printf("This is a null basic_structure_info.\n");
    }
}







/**
 * @brief Construct a new matersdk::Supercell<Coord Type>::Supercell object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
matersdk::Supercell<CoordType>::Supercell() {
    this->structure = Structure<CoordType>();
    this->prim_structure_info = BasicStructureInfo<CoordType>();

    this->scaling_matrix[0] = 1;
    this->scaling_matrix[1] = 1;
    this->scaling_matrix[2] = 1;
    
    this->num_atoms = 0;
    this->prim_cell_idx = 0;

    this->prim_cell_idx_xyz[0] = 0;
    this->prim_cell_idx_xyz[1] = 0;
    this->prim_cell_idx_xyz[2] = 0;

    // owned_atom_idxs
}


/**
 * @brief Construct a new matersdk::Supercell<Coord Type>::Supercell object
 * 
 * @tparam CoordType 
 * @param structure 
 * @param scaling_matrix 
 */
template <typename CoordType>
matersdk::Supercell<CoordType>::Supercell(Structure<CoordType> structure, int *scaling_matrix)
{
    this->structure = structure;
    this->prim_structure_info = BasicStructureInfo<CoordType>(structure);
    for (int ii=0; ii<3; ii++) {
        this->scaling_matrix[ii] = scaling_matrix[ii];
    }
    this->num_atoms = this->prim_structure_info.num_atoms * this->scaling_matrix[0] * this->scaling_matrix[1] * this->scaling_matrix[2];

    this->calc_prim_cell_idx_xyz();             // Assign `this->prim_cell_idx_xyz`
    this->calc_prim_cell_idx();                 // Assign `this->prim_cell_idx`
    this->owned_atom_idxs = (int*)malloc(sizeof(int) * this->prim_structure_info.num_atoms);
    this->calc_owned_atom_idxs();               // Assign `this->prim_owned_atom_idxs`

    // Step 3. make_supercell
    this->structure.make_supercell(this->scaling_matrix);
}


/**
 * @brief Construct a new Supercell< Coord Type>:: Supercell object
 * 
 * @tparam CoordType 
 * @param rhs 
 */
template <typename CoordType>
Supercell<CoordType>::Supercell(const Supercell &rhs) {
    this->num_atoms = rhs.num_atoms;

    if (this->num_atoms == 0) {
        this->structure = Structure<CoordType>();
        this->prim_structure_info = BasicStructureInfo<CoordType>();
        this->scaling_matrix[0] = this->scaling_matrix[1] = this->scaling_matrix[2] = 1;
        
        this->prim_cell_idx = 0;
        this->prim_cell_idx_xyz[0] = this->prim_cell_idx_xyz[1] = this->prim_cell_idx_xyz[2] = 0;
    } else {
        this->structure = rhs.structure;
        this->prim_structure_info = rhs.prim_structure_info;
        for (int ii=0; ii<3; ii++) {
            this->scaling_matrix[ii] = rhs.scaling_matrix[ii];
        }

        this->prim_cell_idx = rhs.prim_cell_idx;
        this->prim_cell_idx_xyz[0] = rhs.prim_cell_idx_xyz[0];
        this->prim_cell_idx_xyz[1] = rhs.prim_cell_idx_xyz[1];
        this->prim_cell_idx_xyz[2] = rhs.prim_cell_idx_xyz[2];
        this->owned_atom_idxs = (int*)malloc(sizeof(int) * rhs.prim_structure_info.num_atoms);    
        for (int ii=0; ii<rhs.prim_structure_info.num_atoms; ii++) {
            this->owned_atom_idxs[ii] = rhs.owned_atom_idxs[ii];
        }
    }

}


/**
 * @brief Overloading assignment operator.
 * 
 * @tparam CoordType 
 * @param rhs 
 * @return Supercell<CoordType>& 
 */
template <typename CoordType>
Supercell<CoordType>& Supercell<CoordType>::operator=(const Supercell<CoordType> &rhs) {
    if (this->num_atoms != 0) {
        free(this->owned_atom_idxs);
    }
 
    this->num_atoms = rhs.num_atoms;

    if (this->num_atoms == 0) {
        this->structure = Structure<CoordType>();
        this->prim_structure_info = BasicStructureInfo<CoordType>();
        this->scaling_matrix[0] = this->scaling_matrix[1] = this->scaling_matrix[2] = 1;
        
        this->prim_cell_idx = 0;
        this->prim_cell_idx_xyz[0] = this->prim_cell_idx_xyz[1] = this->prim_cell_idx_xyz[2] = 0;
    } else {
        this->structure = rhs.structure;
        this->prim_structure_info = rhs.prim_structure_info;
        for (int ii=0; ii<3; ii++) {
            this->scaling_matrix[ii] = rhs.scaling_matrix[ii];
        }

        this->prim_cell_idx = rhs.prim_cell_idx;
        this->prim_cell_idx_xyz[0] = rhs.prim_cell_idx_xyz[0];
        this->prim_cell_idx_xyz[1] = rhs.prim_cell_idx_xyz[1];
        this->prim_cell_idx_xyz[2] = rhs.prim_cell_idx_xyz[2];
        this->owned_atom_idxs = (int*)malloc(sizeof(int) * rhs.prim_structure_info.num_atoms);
        for (int ii=0; ii<rhs.prim_structure_info.num_atoms; ii++) {
            this->owned_atom_idxs[ii] = rhs.owned_atom_idxs[ii];
        }
    }
    
    
    return *this;
}


/**
 * @brief Destroy the Supercell< Coord Type>:: Supercell object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
Supercell<CoordType>::~Supercell() {
    if (this->num_atoms != 0) {
        free(this->owned_atom_idxs);
    }
}


/**
 * @brief Calculate the `this->prim_cell_idx_xyz` and assign it.
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
void Supercell<CoordType>::calc_prim_cell_idx_xyz() {
    /*
        1. scaling_factor = 奇数
            - central_idx = \frac{scaling_factor - 1}{2}
        2. scaling_factor = 偶数
            - central_idx = \frac{scaling_factor}{2} - 1
    */
    for (int ii=0; ii<3; ii++) {
        if (this->scaling_matrix[ii] % 2 != 0) {    // 奇数
            this->prim_cell_idx_xyz[ii] = (this->scaling_matrix[ii] - 1) / 2;
        } else {    // 偶数
            this->prim_cell_idx_xyz[ii] = this->scaling_matrix[ii] / 2 - 1;
        }
    }
}


/**
 * @brief Calculate the `this->prim_cell_idx` and assign it.
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
void Supercell<CoordType>::calc_prim_cell_idx() {
    this->prim_cell_idx = (
                this->prim_cell_idx_xyz[0] + 
                this->prim_cell_idx_xyz[1] * this->scaling_matrix[0] + 
                this->prim_cell_idx_xyz[2] * this->scaling_matrix[0] * this->scaling_matrix[1]
    );
}


/**
 * @brief Calculate the `this->owned_atom_idxs` and assign it.
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
void Supercell<CoordType>::calc_owned_atom_idxs() {
    for (int ii=0; ii<this->prim_structure_info.num_atoms; ii++) {
        this->owned_atom_idxs[ii] = this->prim_cell_idx * this->prim_structure_info.num_atoms + ii;
    }
}


/**
 * @brief Print out the information of `this` (class = Supercell)
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
void Supercell<CoordType>::show() const {
    printf("Prmitive Cell:\n");
    this->prim_structure_info.show();

    printf("\nSuper Cell:\n");
    this->structure.show();
    printf("prim_num_atoms = %15d\n", this->prim_structure_info.num_atoms);
    printf("prim_cell_idx = %15d\n", this->prim_cell_idx);
    printf("prim_cell_idx_xyz = [%4d, %4d, %4d]\n", this->prim_cell_idx_xyz[0], this->prim_cell_idx_xyz[1], this->prim_cell_idx_xyz[2]);
    if (this->num_atoms != 0)
        printf("owned_atom_idxs range = %15d ~ %15d\n", this->owned_atom_idxs[0], this->owned_atom_idxs[this->prim_structure_info.num_atoms-1]);
}


template <typename CoordType>
const Structure<CoordType>& Supercell<CoordType>::get_structure() const {
    return this->structure;
}


template <typename CoordType>
const int* Supercell<CoordType>::get_scaling_matrix() const {
    return (const int*)this->scaling_matrix;
}


template <typename CoordType>
const int* Supercell<CoordType>::get_prim_cell_idx_xyz() const {
    return (const int*)this->prim_cell_idx_xyz;   // Type conversion : `int[]` -> `int*`
}


template <typename CoordType>
const int Supercell<CoordType>::get_prim_cell_idx() const {
    return (const int)this->prim_cell_idx;
}


template <typename CoordType>
const int Supercell<CoordType>::get_prim_num_atoms() const {
    return (const int)(this->prim_structure_info.num_atoms);
}


template <typename CoordType>
const int Supercell<CoordType>::get_num_atoms() const {
    return (const int)(this->structure.num_atoms);
}


template <typename CoordType>
const int* Supercell<CoordType>::get_owned_atom_idxs() const {
    return (const int*)this->owned_atom_idxs;
}






/**
 * @brief Construct a new Bin Linked List< Coord Type>:: Bin Linked List object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
BinLinkedList<CoordType>::BinLinkedList() {
    this->supercell = Supercell<CoordType>();
    this->rcut = 0;
    this->bin_size_xyz[0] = this->bin_size_xyz[1] = this->bin_size_xyz[2] = 0;
    this->pbc_xyz[0] = this->pbc_xyz[1] = this->pbc_xyz[2] = false;
    this->num_bin_xyz[0] = this->num_bin_xyz[1] = this->num_bin_xyz[2] = 0;
    this->min_limit_xyz[0] = this->min_limit_xyz[1] = this->min_limit_xyz[2] = 0;
}


/**
 * @brief Construct a new Bin Linked List< Coord Type>:: Bin Linked List object
 * 
 * @tparam CoordType 
 * @param structure 
 * @param rcut 截断半径
 * @param bin_size_xyz bin_size
 * @param pbc_xyz 是否满足周期性边界条件
 */
template <typename CoordType>
BinLinkedList<CoordType>::BinLinkedList(Structure<CoordType> structure, CoordType rcut, CoordType* bin_size_xyz, bool* pbc_xyz) {
    assert(structure.get_num_atoms() > 0);

    // Step 1. 计算 `scaling_matrix` -- 根据 `rcut` 和 `interplanar_distances`
    this->rcut = rcut;
    this->bin_size_xyz[0] = bin_size_xyz[0];
    this->bin_size_xyz[1] = bin_size_xyz[1];
    this->bin_size_xyz[2] = bin_size_xyz[2];
    this->pbc_xyz[0] = pbc_xyz[0];
    this->pbc_xyz[1] = pbc_xyz[1];
    this->pbc_xyz[2] = pbc_xyz[2];
    CoordType* prim_interplanar_distances = (CoordType*)structure.get_interplanar_distances();
    int* scaling_matrix = (int*)malloc(sizeof(int) * 3);
    int* extending_matrix = (int*)malloc(sizeof(int) * 3);
    for (int ii=0; ii<3; ii++) {
        extending_matrix[ii] = std::ceil(rcut / prim_interplanar_distances[ii]);
        scaling_matrix[ii] = extending_matrix[ii] * 2 + 1;
        if (this->pbc_xyz[ii] == false) {
            scaling_matrix[ii] = 1;
        }
    }

    // Step 2. 初始化 supercell
    Supercell<CoordType> supercell(structure, scaling_matrix);
    this->supercell = supercell;
    CoordType* projected_lengths = (CoordType*)supercell.structure.get_projected_lengths();
    for (int ii=0; ii<3; ii++) {
        this->num_bin_xyz[ii] = std::ceil( projected_lengths[ii] / this->bin_size_xyz[ii] );
    }

    // Step 3. 计算 `min_limit_xyz`  --  Note!!!!
    CoordType** limit_xyz = this->supercell.get_structure().get_limit_xyz();
    this->min_limit_xyz[0] = limit_xyz[0][0];
    this->min_limit_xyz[1] = limit_xyz[1][0];
    this->min_limit_xyz[2] = limit_xyz[2][0];

    // Step 4. 构建 BinLinkedList -- 将 atoms 分配到不同的 bins 中
    this->_build();

    // Step . Free memory
    free(prim_interplanar_distances);
    free(scaling_matrix);
    free(projected_lengths);
    free(extending_matrix);
    for (int ii=0; ii<3; ii++) {
        free(limit_xyz[ii]);
    }
    free(limit_xyz);
}


/**
 * @brief Construct a new Bin Linked List< Coord Type>:: Bin Linked List object
 * 
 * @tparam CoordType 
 * @param structure 
 * @param rcut 
 * @param pbc_xyz 
 */
template <typename CoordType>
BinLinkedList<CoordType>::BinLinkedList(Structure<CoordType> structure, CoordType rcut, bool* pbc_xyz) {
    assert(structure.get_num_atoms() > 0);

    // Step 1. 计算 `scaling_matrix` -- 根据 `rcut` 和 `interplanar_distances`
    this->rcut = rcut;
    this->bin_size_xyz[0] = this->bin_size_xyz[1] = this->bin_size_xyz[2] = this->rcut / 2;
    this->pbc_xyz[0] = pbc_xyz[0];
    this->pbc_xyz[1] = pbc_xyz[1];
    this->pbc_xyz[2] = pbc_xyz[2];
    CoordType* prim_interplanar_distances = (CoordType*)structure.get_interplanar_distances();
    int* scaling_matrix = (int*)malloc(sizeof(int) * 3);
    int* extending_matrix = (int*)malloc(sizeof(int) * 3);
    for (int ii=0; ii<3; ii++) {
        extending_matrix[ii] = std::ceil( this->rcut / prim_interplanar_distances[ii] );
        scaling_matrix[ii] = extending_matrix[ii] * 2 + 1;
        if (this->pbc_xyz[ii] == false) 
            scaling_matrix[ii] = 1;
    }
    
    // Step 2. 初始化 Supercell 
    Supercell<CoordType> supercell(structure, scaling_matrix);
    this->supercell = supercell;
    CoordType* projected_lengths = (CoordType*)supercell.structure.get_projected_lengths();
    for (int ii=0; ii<3; ii++) {
        this->num_bin_xyz[ii] = std::ceil( projected_lengths[ii] / this->bin_size_xyz[ii] );
    }

    // Step 3. 计算 `min_limit_xyz` -- Note!!!
    CoordType** limit_xyz = this->supercell.get_structure().get_limit_xyz();
    this->min_limit_xyz[0] = limit_xyz[0][0];
    this->min_limit_xyz[1] = limit_xyz[1][0];
    this->min_limit_xyz[2] = limit_xyz[2][0];

    // Step 4. 构建 BinLinkedList -- 将 atoms 分配到不同的 bins 中
    this->_build();

    // Step . Free memory
    free(prim_interplanar_distances);
    free(scaling_matrix);
    free(extending_matrix);
    free(projected_lengths);
    for (int ii=0; ii<3; ii++) {
        free(limit_xyz[ii]);
    }
    free(limit_xyz);
}


/**
 * @brief Construct a new Bin Linked List< Coord Type>:: Bin Linked List object
 * 
 * @tparam CoordType 
 * @param rhs 
 */
template <typename CoordType>
BinLinkedList<CoordType>::BinLinkedList(const BinLinkedList<CoordType>& rhs) {
    this->rcut = rhs.rcut;

    if (this->rcut == 0) {
        this->supercell = Supercell<CoordType>();
        this->bin_size_xyz[0] = this->bin_size_xyz[1] = this->bin_size_xyz[2] = 0;
        this->pbc_xyz[0] = this->pbc_xyz[1] = this->pbc_xyz[2] = false;
        this->num_bin_xyz[0] = this->num_bin_xyz[1] = this->num_bin_xyz[2] = 0;
        this->min_limit_xyz[0] = this->min_limit_xyz[1] = this->min_limit_xyz[2] = 0;
        this->heads_lst = nullptr;
        this->nexts_lst = nullptr;
    } else {
        this->supercell = rhs.supercell;
        for (int ii=0; ii<3; ii++) {
            this->bin_size_xyz[ii] = rhs.bin_size_xyz[ii];
            this->pbc_xyz[ii] = rhs.pbc_xyz[ii];
            this->num_bin_xyz[ii] = rhs.num_bin_xyz[ii];
            this->min_limit_xyz[ii] = rhs.min_limit_xyz[ii];
        }

        this->heads_lst = (int*)malloc(sizeof(int) * this->num_bin_xyz[0] * this->num_bin_xyz[1] * this->num_bin_xyz[2]);
        for (int ii=0; ii<this->num_bin_xyz[0] * this->num_bin_xyz[1] * this->num_bin_xyz[2]; ii++) {
            this->heads_lst[ii] = rhs.heads_lst[ii];
        }

        this->nexts_lst = (int*)malloc(sizeof(int) * this->supercell.get_num_atoms());
        for (int ii=0; ii<this->supercell.get_num_atoms(); ii++) {
            this->nexts_lst[ii] = rhs.nexts_lst[ii];
        }
    }
}


/**
 * @brief Overloading the assignment operator
 * 
 * @tparam CoordType 
 * @param rhs 
 * @return BinLinkedList<CoordType>& 
 */
template <typename CoordType>
BinLinkedList<CoordType>& BinLinkedList<CoordType>::operator=(const BinLinkedList<CoordType>& rhs) {
    if (this->rcut != 0) {
        free(this->heads_lst);
        free(this->nexts_lst);
    }
    
    this->rcut = rhs.rcut;

    if (this->rcut == 0) {
        this->supercell = Supercell<CoordType>();
        this->bin_size_xyz[0] = this->bin_size_xyz[1] = this->bin_size_xyz[2] = 0;
        this->pbc_xyz[0] = this->pbc_xyz[1] = this->pbc_xyz[2] = false;
        this->num_bin_xyz[0] = this->num_bin_xyz[1] = this->num_bin_xyz[2] = 0;
        this->min_limit_xyz[0] = this->min_limit_xyz[1] = this->min_limit_xyz[2] = 0;
        this->heads_lst = nullptr;
        this->nexts_lst = nullptr;
    } else {
        this->supercell = rhs.supercell;
        for (int ii=0; ii<3; ii++) {
            this->bin_size_xyz[ii] = rhs.bin_size_xyz[ii];
            this->pbc_xyz[ii] = rhs.pbc_xyz[ii];
            this->num_bin_xyz[ii] = rhs.num_bin_xyz[ii];
            this->min_limit_xyz[ii] = rhs.min_limit_xyz[ii];
        }
        this->heads_lst = (int*)malloc(sizeof(int) * this->num_bin_xyz[0] * this->num_bin_xyz[1] * this->num_bin_xyz[2]);
        for (int ii=0; ii<this->num_bin_xyz[0] * this->num_bin_xyz[1] * this->num_bin_xyz[2]; ii++) {
            this->heads_lst[ii] = rhs.heads_lst[ii];
        }

        this->nexts_lst = (int*)malloc(sizeof(int) * this->supercell.get_num_atoms());
        for (int ii=0; ii<this->supercell.get_num_atoms(); ii++) {
            this->nexts_lst[ii] = rhs.nexts_lst[ii];
        }
    }

    return *this;
}


/**
 * @brief Destroy the Bin Linked List< Coord Type>:: Bin Linked List object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
BinLinkedList<CoordType>::~BinLinkedList() {
    if (this->rcut != 0) {
        free(this->heads_lst);
        free(this->nexts_lst);

        this->rcut = 0;
    }
}


/**
 * @brief Build the `BinLinkedList`, populate `this->heads_lst` and `this->nexts_lst`.
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
void BinLinkedList<CoordType>::_build() {
    // Step 1. Allocate memory for `this->heads_lst`, `this->nexts_lst` and assign all elements as `-1`.
    this->heads_lst = (int*)malloc(sizeof(int) * this->num_bin_xyz[0] * this->num_bin_xyz[1] * this->num_bin_xyz[2]);
    this->nexts_lst = (int*)malloc(sizeof(int) * this->supercell.get_num_atoms());
    for (int ii=0; ii<this->num_bin_xyz[0]*this->num_bin_xyz[1]*this->num_bin_xyz[2]; ii++) {
        this->heads_lst[ii] = -1;
    }
    for (int ii=0; ii<this->supercell.get_num_atoms(); ii++) {
        this->nexts_lst[ii] = -1;
    }

    // Step 2. Populate the `this->heads_lst`, `this->nexts_lst`
    int tmp_bin_idx_xyz[3];
    int tmp_bin_idx;
    for (int ii=0; ii<this->supercell.get_num_atoms(); ii++) {
        const CoordType* atom_coord = this->supercell.get_structure().get_cart_coords()[ii];
        tmp_bin_idx_xyz[0] = std::floor( (atom_coord[0] - this->min_limit_xyz[0]) / this->bin_size_xyz[0] );
        tmp_bin_idx_xyz[1] = std::floor( (atom_coord[1] - this->min_limit_xyz[1]) / this->bin_size_xyz[1] );
        tmp_bin_idx_xyz[2] = std::floor( (atom_coord[2] - this->min_limit_xyz[2]) / this->bin_size_xyz[2] );
        tmp_bin_idx = (
            tmp_bin_idx_xyz[0] + 
            tmp_bin_idx_xyz[1] * this->num_bin_xyz[0] + 
            tmp_bin_idx_xyz[2] * this->num_bin_xyz[0] * this->num_bin_xyz[1]
        );
        this->nexts_lst[ii] = this->heads_lst[tmp_bin_idx];
        this->heads_lst[tmp_bin_idx] = ii;
    }
}


/**
 * @brief 
 *          S1. 找到 prim_atom_idx 在 `center_cell` 中对应的 `atom_idx`
 *          S2. 利用 `atom_cart_coord` 计算 `atom` 所在的 `bin_idx`，并返回
 * 
 * @tparam CoordType 
 * @param prim_atom_idx 
 * @return int 
 */
template <typename CoordType>
int BinLinkedList<CoordType>::get_bin_idx(int prim_atom_idx) const {
    // Step 1. 获取 `prim_atom_idx` 在 supercell 中对应的 `atom_idx`，并获取其坐标 `atom_cart_coord`    
    int atom_idx = prim_atom_idx + (this->supercell.prim_cell_idx * this->supercell.get_prim_num_atoms());
    const CoordType* atom_cart_coord = this->supercell.structure.get_cart_coords()[atom_idx];
    
    // Step 2. 根据 `atom_cart_coord` 计算原子所属的 bin_idx
    int bin_idx_xyz[3];
    for (int ii=0; ii<3; ii++) {
        bin_idx_xyz[ii] = std::floor( (atom_cart_coord[ii] - this->min_limit_xyz[ii]) / this->bin_size_xyz[ii] );
    }

    return (
        bin_idx_xyz[0] +
        bin_idx_xyz[1] * this->num_bin_xyz[0] + 
        bin_idx_xyz[2] * this->num_bin_xyz[0] * this->num_bin_xyz[1]
    );
}



/**
 * @brief 得到 neigh_bins 的数量
 * 
 * @tparam CoordType 
 * @return int 
 */
template <typename CoordType>
int BinLinkedList<CoordType>::get_num_neigh_bins() const {
    int num_extend_neigh_bins[3]; // 沿 x,y,z 的正方向有多少个 neigh_bins
    for (int ii=0; ii<3; ii++) {
        num_extend_neigh_bins[ii] = std::ceil(this->rcut / this->bin_size_xyz[ii]);
    }
    int num_neigh_bins = (
            ( 2 * num_extend_neigh_bins[0] + 1 ) *
            ( 2 * num_extend_neigh_bins[1] + 1 ) *
            ( 2 * num_extend_neigh_bins[2] + 1 )
    );

    return num_neigh_bins;
}



/**
 * @brief Get neighbor bin's index(int) given atom_idx
 * 
 * @tparam CoordType 
 * @param atom_idx 
 * @return int* 
 */
template <typename CoordType>
int* BinLinkedList<CoordType>::get_neigh_bins(int prim_atom_idx) const {
    // Step 1. 得到 atom_idx 所在的 bin_idx
    // Step 1.1.
    assert(prim_atom_idx < this->supercell.get_prim_num_atoms());
    int atom_idx = prim_atom_idx + this->supercell.get_prim_cell_idx() * this->supercell.get_prim_num_atoms();
    // Step 1.2. 得到 `atom_cart_coord`, `min_limit_xyz`
    const CoordType* atom_cart_coord = this->supercell.get_structure().get_cart_coords()[atom_idx];
    CoordType** limit_xyz = this->supercell.get_structure().get_limit_xyz();
    CoordType min_limit_xyz[3];
    min_limit_xyz[0] = limit_xyz[0][0];
    min_limit_xyz[1] = limit_xyz[1][0];
    min_limit_xyz[2] = limit_xyz[2][0];

    // Step 1.3. 计算 atom_idx 的 bin_idx_xyz[3]
    int bin_idx_xyz[3];
    for (int ii=0; ii<3; ii++)
        bin_idx_xyz[ii] = std::floor( (atom_cart_coord[ii] - min_limit_xyz[ii]) / this->bin_size_xyz[ii] );

    // Step 2. 由 `atom_idx` 的 `bin_idx_xyz[3]` 计算 `neigh_bin_idxs_xyz[num_neigh_bins][3]`
    int num_extened_neigh_bins[3];  // 沿 x,y,z 的正方向有多少个 neigh_bins
    for (int ii=0; ii<3; ii++)
        num_extened_neigh_bins[ii] = std::ceil(this->rcut / this->bin_size_xyz[ii]);
    int num_neigh_bins = (
                ( 2 * num_extened_neigh_bins[0] + 1 ) *
                ( 2 * num_extened_neigh_bins[1] + 1 ) * 
                ( 2 * num_extened_neigh_bins[2] + 1 )
    );
    int neigh_bin_idxs_xyz[num_neigh_bins][3];
    int candidate_neigh_bin_idxs_xyz[num_neigh_bins][3];    // 未处理周期性
    int neigh_bin_idx4loop = 0;     // 用于在循环中赋值，并不是实际的 `bin index`
    for (int ii=-num_extened_neigh_bins[0]; ii<=num_extened_neigh_bins[0]; ii++) {
        for (int jj=-num_extened_neigh_bins[1]; jj<=num_extened_neigh_bins[1]; jj++) {
            for (int kk=-num_extened_neigh_bins[2]; kk<=num_extened_neigh_bins[2]; kk++) {
                // Step 2.1. 未处理 pbc
                candidate_neigh_bin_idxs_xyz[neigh_bin_idx4loop][0] = bin_idx_xyz[0] + ii;
                candidate_neigh_bin_idxs_xyz[neigh_bin_idx4loop][1] = bin_idx_xyz[1] + jj;
                candidate_neigh_bin_idxs_xyz[neigh_bin_idx4loop][2] = bin_idx_xyz[2] + kk;
                
                //printf("shift_index_%d = [%d, %d, %d]\n", neigh_bin_idx4loop, ii, jj, kk);

                // Step 2.2. 处理 pbc (wrap around)
                for (int pp=0; pp<3; pp++) {
                    if (this->pbc_xyz[pp] == false) {   // 无周期性
                        neigh_bin_idxs_xyz[neigh_bin_idx4loop][pp] = candidate_neigh_bin_idxs_xyz[neigh_bin_idx4loop][pp];
                    } else {
                        neigh_bin_idxs_xyz[neigh_bin_idx4loop][pp] = (candidate_neigh_bin_idxs_xyz[neigh_bin_idx4loop][pp] + this->num_bin_xyz[pp] ) % this->num_bin_xyz[pp];                        
                    }
                }
                //printf("this->num_bins_xyz[%d][3] = [%d, %d, %d]\n", this->num_bin_xyz[0], this->num_bin_xyz[1], this->num_bin_xyz[2]);                
                //printf("neigh_bin_idxs_xyz[%d][3] = [%d, %d, %d]\n", neigh_bin_idxs_xyz[neigh_bin_idx4loop][0], neigh_bin_idxs_xyz[neigh_bin_idx4loop][1], neigh_bin_idxs_xyz[neigh_bin_idx4loop][2]);

                neigh_bin_idx4loop++;
            }
        }
    }
    
    // Step 3. 由 `neigh_bin_idxs_xyz[num_neigh_bins][3]` 计算 `neigh_bin_idxs[num_neigh_bins]`
    int* neigh_bin_idxs = (int*)malloc(sizeof(int) * num_neigh_bins);
    for (int ii=0; ii<num_neigh_bins; ii++) {
        if (
            (neigh_bin_idxs_xyz[ii][0] >= 0) && (neigh_bin_idxs_xyz[ii][0] < this->num_bin_xyz[0]) &&
            (neigh_bin_idxs_xyz[ii][1] >= 0) && (neigh_bin_idxs_xyz[ii][1] < this->num_bin_xyz[1]) &&
            (neigh_bin_idxs_xyz[ii][2] >= 0) && (neigh_bin_idxs_xyz[ii][2] < this->num_bin_xyz[2]) 
        ) {     // non-pbc时，neigh_bin_idx_xyz 有可能超出边界。
            //printf("%d, %d, %d, %d, %d, %d\n", neigh_bin_idxs_xyz[ii][0] > 0, neigh_bin_idxs_xyz[ii][0] < this->num_bin_xyz[0], neigh_bin_idxs_xyz[ii][1] > 0, neigh_bin_idxs_xyz[ii][1] < this->num_bin_xyz[1], neigh_bin_idxs_xyz[ii][2] > 0, neigh_bin_idxs_xyz[ii][2] < this->num_bin_xyz[2]);
            neigh_bin_idxs[ii] = (
                neigh_bin_idxs_xyz[ii][0] + 
                neigh_bin_idxs_xyz[ii][1] * this->num_bin_xyz[0] + 
                neigh_bin_idxs_xyz[ii][2] * this->num_bin_xyz[0] * this->num_bin_xyz[1]
            );
        } else {
            neigh_bin_idxs[ii] = -1;
        }
    }

    // Step . Free memory
    for (int ii=0; ii<3; ii++) {
        free(limit_xyz[ii]);
    }
    free(limit_xyz);

    return neigh_bin_idxs;
}


/**
 * @brief 返回 prim_atom_idx 的近邻原子的index (index 是在 supercell 中的 index)。后续可以用这个构建完整的 neighbor list
 * 
 * @tparam CoordType 
 * @param prim_atom_idx 
 * @return std::vector<int> 
 */
template <typename CoordType>
std::vector<int> BinLinkedList<CoordType>::get_neigh_atoms(int prim_atom_idx) const {    
    // Step 1. 
    // Step 1.1. 初始化需要返回的 `neigh_atom_idxs` -- `neigh_atom_idxs` 存储了原子在 supercell 中的 index
    std::vector<int> neigh_atom_idxs;
    // Step 1.2. 获取 prim_atom_idx 的
    //              1. 在 supercell 中对应原子的索引        -- `atom_idx`
    //              2. 在 supercell 中对应原子的笛卡尔坐标   -- `atom_cart_coord`
    int atom_idx = prim_atom_idx + this->supercell.get_prim_num_atoms() * this->supercell.get_prim_cell_idx();
    const CoordType* atom_cart_coord = this->supercell.get_structure().get_cart_coords()[atom_idx];
    // Step 1.3. 
    int *neigh_bin_idxs = this->get_neigh_bins(prim_atom_idx);  // 近邻 bin 的 index
    int num_neigh_bins = this->get_num_neigh_bins();            // 近邻 bin 的数目
    int neigh_bin_idx;      // 用于遍历近邻 bin
    int neigh_atom_idx;     // 用于遍历近邻 atom
    const CoordType* neigh_atom_cart_coord; 
    CoordType relative_distance2;
    const CoordType** supercell_atom_cart_coords = this->supercell.get_structure().get_cart_coords();

    // Step 2. 构建 neighbor list
    for (int ii=0; ii<num_neigh_bins; ii++) {
        neigh_bin_idx = neigh_bin_idxs[ii];     // 获取近邻bin的 index
        if (neigh_bin_idx != -1) {              // `-1` 代表 `无效的近邻bin`
            neigh_atom_idx = this->heads_lst[neigh_bin_idx];
            while (neigh_atom_idx != -1) {      // 遍历 bin 里的所有 atoms
                neigh_atom_cart_coord = supercell_atom_cart_coords[neigh_atom_idx];
                relative_distance2 = (
                    std::pow(neigh_atom_cart_coord[0] - atom_cart_coord[0], 2) + 
                    std::pow(neigh_atom_cart_coord[1] - atom_cart_coord[1], 2) + 
                    std::pow(neigh_atom_cart_coord[2] - atom_cart_coord[2], 2)
                );
                
                if (
                    ( relative_distance2 < std::pow(this->rcut, 2) ) &&
                    ( relative_distance2 > EPSILON ) 
                ) { // 将在截断半径(rcut)内、排除中心原子自身 的近邻原子加入 neighbor list
                    //printf("%d,\t %f\n", this->supercell.get_structure().get_atomic_numbers()[neigh_atom_idx], std::sqrt(relative_distance2));
                    neigh_atom_idxs.push_back(neigh_atom_idx);
                }

                neigh_atom_idx = this->nexts_lst[neigh_atom_idx];
            }
        }
    }

    // Step . Free memory
    free(neigh_bin_idxs);

    return neigh_atom_idxs;
}


template <typename CoordType>
void BinLinkedList<CoordType>::show() const {
    if (this->rcut != 0) { 
        printf("rcut = %15.3f\n", this->rcut);
        printf("pbc_xyz = [%d, %d, %d]\n", this->pbc_xyz[0], this->pbc_xyz[1], this->pbc_xyz[2]);
        printf("bin_size_xyz = [%9.3f, %9.3f, %9.3f]\n", this->bin_size_xyz[0], this->bin_size_xyz[1], this->bin_size_xyz[2]);
        printf("num_bin_xyz = [%d, %d, %d]\n", this->num_bin_xyz[0], this->num_bin_xyz[1], this->num_bin_xyz[2]);
        printf("min_limit_xyz = [%9.6f, %9.6f, %9.6f]\n", this->min_limit_xyz[0], this->min_limit_xyz[1], this->min_limit_xyz[2]);
    
        int atom_idx = 0;
        for (int ii=0; ii<this->num_bin_xyz[0]*this->num_bin_xyz[1]*this->num_bin_xyz[2]; ii++) {
            atom_idx = this->heads_lst[ii];
            printf("bin_%d : \t", ii);
            while (atom_idx != -1) {
                printf("%d, ", atom_idx);
                atom_idx = this->nexts_lst[atom_idx];
            }
            printf("\n");
        }
    } else {
        printf("This is a NULL BinLinkedList.\n");
    }
}


template <typename CoordType>
const Supercell<CoordType>& BinLinkedList<CoordType>::get_supercell() const {
    return this->supercell;
}


template <typename CoordType>
const CoordType* BinLinkedList<CoordType>::get_bin_size_xyz() const {
    return (const CoordType*)(this->bin_size_xyz);
}


template <typename CoordType>
const int* BinLinkedList<CoordType>::get_num_bin_xyz() const {
    return (const int*)(this->num_bin_xyz);
}


template <typename CoordType>
const CoordType* BinLinkedList<CoordType>::get_min_limit_xyz() const {
    return (const CoordType*)(this->min_limit_xyz);
}



} // namespace: matersdk

#endif