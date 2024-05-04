#ifndef MATERSDK_SE_H
#define MATERSDK_SE_H


#include <stdlib.h>

#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../../../core/include/vec3Operation.h"
#include "../../../core/include/arrayUtils.h"


namespace matersdk {
namespace deepPotSE{


/**
 * @brief Switching Function in DeepPot-SE
 *          0. uu = \frac{r - r_s}{r_c - r_s}
 *          1. switchFunc(uu) = 
 *              1. 1
 *              2. uu^3(-6uu^2+15uu-10) + 1
 *              3. 0
 *          2. the gradient of switchFunc(uu) with respect to uu:
 *              1. 0
 *              2. -30 uu^4 + 60 uu^3 - 30 uu^2
 *              3. 0
 *          3. the gradient of switchFunc(uu) with respect to r:
 *              1. 0
 *              2. (-30 uu^4 _ 60 uu^3 - 30 uu^2) * 1/(r_c-r_s)
 *              3. 0
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
class SwitchFunc {
public:
    SwitchFunc(CoordType rcut, CoordType rcut_smooth);

    CoordType get_result(CoordType distance_ji) const;

    CoordType get_deriv2r(CoordType distance_ji) const;

    void show() const;

private:
    CoordType rcut = 0;
    CoordType rcut_smooth = 0;
}; // class : SwitchFunc


template <typename CoordType>
SwitchFunc<CoordType>::SwitchFunc(CoordType rcut, CoordType rcut_smooth) {
    this->rcut = rcut;
    this->rcut_smooth = rcut_smooth;
}


template <typename CoordType>
CoordType SwitchFunc<CoordType>::get_result(CoordType distance_ji) const {
    CoordType result;
    CoordType uu = (distance_ji - this->rcut_smooth) / (this->rcut - this->rcut_smooth);

    if (distance_ji < this->rcut_smooth)
        result = 1;
    else if ((distance_ji>=this->rcut_smooth) && (distance_ji<this->rcut))
        result = std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1;
    else
        result = 0;

    return result;
}


template <typename CoordType>
CoordType SwitchFunc<CoordType>::get_deriv2r(CoordType distance_ji) const {
    CoordType derive2r;
    CoordType uu = (distance_ji - this->rcut_smooth) / (this->rcut - this->rcut_smooth);

    if (distance_ji < this->rcut_smooth)
        derive2r = 0;
    else if ((distance_ji>=this->rcut_smooth) && (distance_ji<this->rcut))
        derive2r = 1/(this->rcut - this->rcut_smooth) * ( -30*std::pow(uu, 4) + 60*std::pow(uu, 3) - 30*std::pow(uu, 2) );
    else
        derive2r = 0;
    
    return derive2r;
}


template <typename CoordType>
void SwitchFunc<CoordType>::show() const {
    printf("Inner SwitchFunc:\n");
    printf("\tthis->rcut = %f\n", this->rcut);
    printf("\tthis->rcut_smooth = %f\n", this->rcut_smooth);
}



/**
 * @brief Smooth function(`s(r)`) is DeepPot-SE
 * 
 * @tparam CoordType 
 * @param distance_ji 
 * @param rcut 
 * @param rcut_smooth 
 * @return CoordType 
 */
template <typename CoordType>
CoordType smooth_func(const CoordType& distance_ji, const CoordType& rcut, const CoordType& rcut_smooth) {
    CoordType smooth_value;     // return value
    // Step 1. calculate `r_recip`
    CoordType r_recip = 0;
    if (distance_ji == 0)
        r_recip = 0;
    else
        r_recip = 1 / distance_ji;
    
    // Step 2. uu = (r - r_s) / (r_c - r_s)
    CoordType uu = (distance_ji - rcut_smooth) / (rcut - rcut_smooth);
    
    // Step 3. Calculate `smooth_value`
    if (distance_ji < rcut_smooth)
        smooth_value = r_recip;
    else if ( (distance_ji >= rcut_smooth) && (distance_ji < rcut) )
        smooth_value = r_recip * ( std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1);
    else
        smooth_value = 0;

    return smooth_value;
}


/**
 * @brief Calculate the reciprocal value.
 * 
 * @tparam CoordType 
 * @param value 
 * @return CoordType 
 */
template <typename CoordType>
CoordType recip(const CoordType& value) {
    CoordType value_recip;
    if (value == 0) {
        value_recip = 0;
    } else {
        value_recip = 1.0 / value;
    }
    return value_recip;
}


/**
 * @brief 指定 `center_atomic_number` 和 `neigh_atomic_number`，计算 DeepPot-SE 的 feature
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
class PairTildeR {
public:
    PairTildeR();

    PairTildeR(
                NeighborList<CoordType>& neighbor_list, 
                int center_atomic_number, 
                int neigh_atomic_number, 
                int num_neigh_atoms,
                CoordType rcut_smmoth);
    
    // This constructor just for `TildeR`
    PairTildeR(
                const NeighborList<CoordType>& neighbor_list,
                int center_atomic_number,
                int neigh_atomic_number,
                int num_center_atoms,
                int num_neigh_atoms,
                CoordType rcut_smooth);

    PairTildeR(
                NeighborList<CoordType>& neighbor_list, 
                int center_atomic_number, 
                int neigh_atomic_number,
                CoordType rcut_smooth);

    PairTildeR(
                Structure<CoordType>& structure,
                CoordType rcut,
                bool* pbc_xyz,
                bool sort,
                int center_atomic_number,
                int neigh_atomic_number,
                int num_neigh_atoms,
                CoordType rcut_smooth);
    
    PairTildeR(
                Structure<CoordType>& structure,
                CoordType rcut,
                bool* pbc_xyz,
                bool sort,
                int center_atomic_number,
                int neigh_atomic_number,
                CoordType rcut_smooth);

    PairTildeR(const PairTildeR& rhs);

    PairTildeR& operator=(const PairTildeR& rhs);

    // ~PairTildeR();

    void calc_num_center_atoms();

    const int get_num_center_atoms() const;     // 得到中心原子的数目

    const int get_num_neigh_atoms() const;      // 得到指定的近邻原子数目（`this->num_neigh_atoms`相当于指定了 zero-padding 的尺寸）

    void show() const;

    void show_in_value() const;

    void show_in_deriv() const;

    const int get_max_num_neigh_atoms() const;  // 得到最大近邻原子数目 （指定 `center_atomic_number`, `neigh_atomic_number`, 计算最大近邻原子数）

    CoordType*** generate() const;              // 计算 $\tilde{R^i}$ 特征 .shape = (num_center_atoms, num_neigh_atoms, 4)

    // For lammps
    static CoordType*** generate(
                int inum,                       // 中心原子数目
                int* ilist,                     // 存储中心原子的index -- `ilist[ii]`
                int* numneigh,                  // 每个中心原子的近邻原子数目
                int** firstneigh,               // firstneigh[ii][jj]: 第 ii 个中心原子的第 jj 个近邻原子
                CoordType** x,                  // supercell 中所有原子的坐标
                int* types,                     // supercell 中所有原子的原子序数
                int center_atomic_number,       // 中心原子的原子序数
                int neigh_atomic_number,        // 近邻原子的原子序数
                int num_neigh_atoms,            // 决定了 zero-padding 的尺寸
                CoordType rcut,
                CoordType rcut_smooth);

    CoordType**** deriv() const;                // 计算 $\tilde{R^i}$ 特征的导数 .shape = (num_center_atoms, num_neigh_atoms, 4, 3)

    // For lammps
    static CoordType**** deriv(
                int inum,                       // 中心原子的数目
                int* ilist,                     // 存储中心原子的index -- `ilist[ii]`
                int* numneigh,                  // 每个中心原子的近邻原子数目
                int** firstneigh,               // firstneigh[ii][jj]: 第 ii 个中心原子的第 jj 个近邻原子
                CoordType** x,                  // supercell 中所有原子的坐标
                int* types,                     // supercell 中所有原子的原子序数
                int center_atomic_number,       // 中心原子的原子序数
                int neigh_atomic_numebr,        // 近邻原子的原子序数
                int num_neigh_atoms,            // 决定了 zero-padding 的尺寸
                CoordType rcut,
                CoordType rcut_smooth);

private:
    NeighborList<CoordType> neighbor_list;
    int center_atomic_number = 0;
    int neigh_atomic_number = 0;
    int num_center_atoms = 0;
    int num_neigh_atoms = 0;                    // 相当于指定了 zero-padding 的尺寸
    CoordType rcut = 0;
    CoordType rcut_smooth = 0;
};  // class PairTildeR



/**
 * @brief TildeR : Stack PairTildeRs
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
class TildeR {
public:
    TildeR();

    TildeR(
        NeighborList<CoordType>& neighbor_list,
        int num_center_atomic_numbers,
        int* center_atomic_numbers_lst,
        int num_neigh_atomic_numbers,
        int* neigh_atomic_numbers_lst,
        int* num_neigh_atoms_lst,
        CoordType rcut_smooth);
    
    TildeR(
        NeighborList<CoordType>& neighbor_list,
        int num_center_atomic_numbers,
        int* center_atomic_numbers_lst,
        int num_neigh_atomic_numbers,
        int* neigh_atomic_numbers_lst,
        CoordType rcut_smooth);

    TildeR(
        Structure<CoordType>& structure,
        CoordType rcut,
        bool* pbc_xyz,
        bool sort,
        int num_center_atomic_numbers,
        int* center_atomic_numbers_lst,
        int num_neigh_atomic_numbers,
        int* neigh_atomic_numbers_lst,
        int* num_neigh_atoms_lst,
        CoordType rcut_smooth);

    TildeR(
        Structure<CoordType>& structure,
        CoordType rcut, 
        bool* pbc_xyz,
        bool sort,
        int num_center_atomic_numbers,
        int* center_atomic_numbers_lst,
        int num_neigh_atomic_numbers,
        int* neigh_atomic_numbers_lst,
        CoordType rcut_smooth);

    TildeR(const TildeR& rhs);

    TildeR& operator=(const TildeR& rhs);

    ~TildeR();

    void calc_num_center_atoms_lst();

    void calc_num_neigh_atoms_lst();

    const int get_num_center_atoms() const;   // $\tilde{R}$ 的第一维 = sum(this->num_center_atoms_lst)

    const int get_num_neigh_atoms() const;    // $\tilde{R}$ 的第一维 = sum(this->num_neigh_atoms_lst)

    void show() const;

    void show_in_value() const;

    void show_in_deriv() const;

    CoordType*** generate() const;

    static CoordType*** generate(
                int inum,           // 中心原子数目
                int* ilist,         // 中心原子在 supercell 中的 index
                int* numneigh,      // 每个中心原子的近邻原子数目
                int** firstneigh,   // 近邻原子在 supercell 中的 index -- `firstneigh[ii][jj]`
                CoordType** x,      // supercell 中所有原子的笛卡尔坐标
                int* types,         // supercell 中所有原子的原子序数
                int num_center_atomic_numbers,  // 中心原子的种类数 e.g. MoS2 -- 2
                int* center_atomic_numbers_lst, // 中心原子的原子序数 e.g. MoS2 -- [42, 16]
                int num_neigh_atomic_numbers,   // 近邻原子的种类数 e.g. MoS2 -- 2
                int* neigh_atomic_numbers_lst,  // 近邻原子的原子序数 e.g. MoS2 -- [42, 16]
                int* num_neigh_atoms_lst,   // 近邻原子的数目 : 决定了 zero-padding 的数目 e.g. MoS2 -- [100, 80]
                CoordType rcut,
                CoordType rcut_smooth);
    
    static void generate(
                CoordType*** tilde_r,
                int inum,
                int* ilist,
                int* numneigh,
                int** firstneigh,
                CoordType** x,
                int* types,
                int num_center_atomic_numbers,
                int* center_atomic_numbers_lst,
                int num_neigh_atomic_numbers,
                int* neigh_atomic_numbers,
                int* num_neigh_atoms_lst,
                CoordType rcut, 
                CoordType rcut_smooth);

    CoordType**** deriv() const;

    static CoordType**** deriv(
                int inum,           // 中心原子数目
                int* ilist,         // 中心原子在 supercell 中的 index
                int* numneigh,      // 每个中心原子的近邻原子数目
                int** firstneigh,   // 近邻原子在 supercell 中的 index -- `firstneigh[ii][jj]`
                CoordType** x,      // supercell 中所有原子的笛卡尔坐标
                int* types,         // supercell 中所有原子的原子序数
                int num_center_atomic_numbers,  // 中心原子的种类数 e.g. MoS2 -- 2
                int* center_atomic_numbers_lst,  // 中心原子的原子序数 e.g. MoS2 -- [42, 16]
                int num_neigh_atomic_numbers,   // 近邻原子的种类数 e.g. MoS2 -- 2
                int* neigh_atomic_numbers_lst,  // 近邻原子的原子序数 e.g. MoS2 -- [42, 16]
                int* num_neigh_atoms_lst,   // 近邻原子的数目：决定了 zero-padding 的数目
                CoordType rcut,
                CoordType rcut_smooth);


    static void deriv(
                CoordType**** tilde_r_deriv,
                int inum,
                int* ilist,
                int* numneigh,
                int** firstneigh,
                CoordType** x,
                int* types,
                int num_center_atomic_numbers,
                int* center_atomic_numbers_lst,
                int num_neigh_atomic_numbers,
                int* neigh_atomic_numbers_lst,
                int* num_neigh_atoms_lst,
                CoordType rcut,
                CoordType rcut_smooth);


private:
    NeighborList<CoordType> neighbor_list;
    int num_center_atomic_numbers = 0;
    int* center_atomic_numbers_lst = nullptr;
    int num_neigh_atomic_numbers = 0;
    int* neigh_atomic_numbers_lst = nullptr;
    int* num_center_atoms_lst = nullptr;
    int* num_neigh_atoms_lst = nullptr;
    CoordType rcut = 0;
    CoordType rcut_smooth = 0;
};  // class : TildeR







/**
 * @brief Construct a new Pair Tilde R< Coord Type>:: Pair Tilde R object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
PairTildeR<CoordType>::PairTildeR() {
    this->neighbor_list = NeighborList<CoordType>();
    this->center_atomic_number = 0;
    this->neigh_atomic_number = 0;
    this->num_center_atoms = 0;
    this->num_neigh_atoms = 0;
    this->rcut = 0;
    this->rcut_smooth = 0;
}


/**
 * @brief Construct a new Pair Tilde R< Coord Type>:: Pair Tilde R object
 * 
 * @tparam CoordType 
 * @param neighbor_list 
 * @param center_atomic_number 
 * @param neigh_atomic_number 
 * @param num_neigh_atoms 
 */
template <typename CoordType>
PairTildeR<CoordType>::PairTildeR(
                            NeighborList<CoordType>& neighbor_list, 
                            int center_atomic_number, 
                            int neigh_atomic_number, 
                            int num_neigh_atoms,
                            CoordType rcut_smooth) {
    this->neighbor_list = neighbor_list;
    this->center_atomic_number = center_atomic_number;
    this->neigh_atomic_number = neigh_atomic_number;
    this->calc_num_center_atoms();  // 计算 `this->num_center_atoms`
    this->num_neigh_atoms = num_neigh_atoms;
    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


template <typename CoordType>
PairTildeR<CoordType>::PairTildeR(
                            const NeighborList<CoordType>& neighbor_list,
                            int center_atomic_number,
                            int neigh_atomic_number,
                            int num_center_atoms,
                            int num_neigh_atoms,
                            CoordType rcut_smooth)
{
    this->neighbor_list = neighbor_list;
    this->center_atomic_number = center_atomic_number;
    this->neigh_atomic_number = neigh_atomic_number;
    this->num_center_atoms = num_center_atoms;
    this->num_neigh_atoms = num_neigh_atoms;
    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


/**
 * @brief Construct a new Pair Tilde R< Coord Type>:: Pair Tilde R object
 * 
 * @tparam CoordType 
 * @param neighbor_list 
 * @param center_atomic_number 
 * @param neigh_atomic_number 
 */
template <typename CoordType>
PairTildeR<CoordType>::PairTildeR(NeighborList<CoordType>& neighbor_list, int center_atomic_number, int neigh_atomic_number, CoordType rcut_smooth) {
    this->neighbor_list = neighbor_list;
    this->center_atomic_number = center_atomic_number;
    this->neigh_atomic_number = neigh_atomic_number;
    this->calc_num_center_atoms();  // 计算 `this->num_center_atoms`
    this->num_neigh_atoms = this->get_max_num_neigh_atoms();
    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


/**
 * @brief Construct a new Pair Tilde R< Coord Type>:: Pair Tilde R object
 * 
 * @tparam CoordType 
 * @param structure 
 * @param rcut 
 * @param pbc_xyz 
 * @param sort 
 * @param center_atomic_number 
 * @param neigh_atomic_number 
 * @param num_neigh_atoms 
 * @param rcut_smooth 
 */
template <typename CoordType>
PairTildeR<CoordType>::PairTildeR(
                        Structure<CoordType>& structure,
                        CoordType rcut,
                        bool* pbc_xyz,
                        bool sort,
                        int center_atomic_number,
                        int neigh_atomic_number,
                        int num_neigh_atoms,
                        CoordType rcut_smooth)
{
    this->neighbor_list = NeighborList<CoordType>(structure, rcut, pbc_xyz, sort);
    this->center_atomic_number = center_atomic_number;
    this->neigh_atomic_number = neigh_atomic_number;
    this->calc_num_center_atoms();
    this->num_neigh_atoms = num_neigh_atoms;
    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


/**
 * @brief Construct a new Pair Tilde R< Coord Type>:: Pair Tilde R object
 * 
 * @tparam CoordType 
 * @param structure 
 * @param rcut 
 * @param pbc_xyz 
 * @param sort 
 * @param center_atomic_number 
 * @param neigh_atomic_number 
 * @param rcut_smooth 
 */
template <typename CoordType>
PairTildeR<CoordType>::PairTildeR(
                        Structure<CoordType>& structure,
                        CoordType rcut,
                        bool* pbc_xyz,
                        bool sort,
                        int center_atomic_number,
                        int neigh_atomic_number,
                        CoordType rcut_smooth)
{
    this->neighbor_list = NeighborList<CoordType>(structure, rcut, pbc_xyz, sort);
    this->center_atomic_number = center_atomic_number;
    this->neigh_atomic_number = neigh_atomic_number;
    this->calc_num_center_atoms();
    this->num_neigh_atoms = this->get_max_num_neigh_atoms();
    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


template <typename CoordType>
PairTildeR<CoordType>::PairTildeR(const PairTildeR<CoordType>& rhs) {
    this->neighbor_list = rhs.neighbor_list;
    this->center_atomic_number = rhs.center_atomic_number;
    this->neigh_atomic_number = rhs.neigh_atomic_number;
    this->num_center_atoms = rhs.num_center_atoms;
    this->num_neigh_atoms = rhs.num_neigh_atoms;
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;
}


template <typename CoordType>
PairTildeR<CoordType>& PairTildeR<CoordType>::operator=(const PairTildeR<CoordType>& rhs) {
    this->neighbor_list = rhs.neighbor_list;
    this->center_atomic_number = rhs.center_atomic_number;
    this->neigh_atomic_number = rhs.neigh_atomic_number;
    this->num_center_atoms = rhs.num_center_atoms;
    this->num_neigh_atoms = rhs.num_neigh_atoms;
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;

    return *this;
}



/**
 * @brief 计算 `this->num_center_atoms`
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
void PairTildeR<CoordType>::calc_num_center_atoms() {
    int num_center_atoms = 0;
    int prim_num_atoms = this->neighbor_list.get_binLinkedList().get_supercell().get_prim_num_atoms();
    const int* supercell_atomic_numbers = this->neighbor_list.get_binLinkedList().get_supercell().get_structure().get_atomic_numbers();

    for (int ii=0; ii<prim_num_atoms; ii++) {
        if (supercell_atomic_numbers[ii] == this->center_atomic_number)
            num_center_atoms++;
    }

    this->num_center_atoms = num_center_atoms;
}


/**
 * @brief 得到 `PairTildeR` 中心原子的数目 (原子序数==`this->center_atomic_number`)
 * 
 * @tparam CoordType 
 * @return const int 
 */
template <typename CoordType>
const int PairTildeR<CoordType>::get_num_center_atoms() const {
    return this->num_center_atoms;
}


/**
 * @brief 得到 `PairTildeR` 近邻原子的数目 (原子序数==`this->neigh_atomic_number`)
 * 
 * @tparam CoordType 
 * @return const int 
 */
template <typename CoordType>
const int PairTildeR<CoordType>::get_num_neigh_atoms() const {
    return this->num_neigh_atoms;
}


template <typename CoordType>
void PairTildeR<CoordType>::show() const {
    if (this->center_atomic_number == 0)
        printf("This is a NULL PairTildeR.\n");
    else {
        int max_num_neigh_atoms = this->neighbor_list.get_max_num_neigh_atoms_ssss(this->center_atomic_number, this->neigh_atomic_number);
        printf("center_atomic_number = %d\n", this->center_atomic_number);
        printf("neigh_atomic_number = %d\n", this->neigh_atomic_number);
        printf("num_center_atoms = %d\n", this->num_center_atoms);
        printf("num_neigh_atoms = %d\n", this->num_neigh_atoms);
        printf("max_num_neigh_atoms_ss = %d\n", this->get_max_num_neigh_atoms());
        printf("rcut = %f\n", this->rcut);
        printf("rcut_smooth = %f\n", this->rcut_smooth);
    }
}


template <typename CoordType>
void PairTildeR<CoordType>::show_in_value() const {
    if (this->center_atomic_number == 0)
        printf("This is a NULL PairTildeR.\n");
    else {
        CoordType*** pair_tilde_r = this->generate();

        for (int ii=0; ii<this->num_center_atoms; ii++) {
            for (int jj=0; jj<this->num_neigh_atoms; jj++) {
                printf("[%4d, %4d] -- [%10f, %10f, %10f, %10f]\n", 
                        ii, jj,
                        pair_tilde_r[ii][jj][0],
                        pair_tilde_r[ii][jj][1],
                        pair_tilde_r[ii][jj][2],
                        pair_tilde_r[ii][jj][3]
                );
            }
        }

        // Step . Free memory
        for (int ii=0; ii<this->num_center_atoms; ii++) {
            for (int jj=0; jj<this->num_neigh_atoms; jj++) {
                free(pair_tilde_r[ii][jj]);
            }
            free(pair_tilde_r[ii]);
        }
        free(pair_tilde_r);
    }

}


template <typename CoordType>
void PairTildeR<CoordType>::show_in_deriv() const {
    if (this->center_atomic_number == 0)
        printf("This is a NULL PairTildeR.\n");
    else {
        CoordType**** pair_tilde_r_deriv = this->deriv();

        for (int ii=0; ii<this->num_center_atoms; ii++) {
            for (int jj=0; jj<this->num_neigh_atoms; jj++) {
                printf("[%4d, %4d] -- [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f]\n", 
                        ii, jj,
                        pair_tilde_r_deriv[ii][jj][0][0],
                        pair_tilde_r_deriv[ii][jj][0][1],
                        pair_tilde_r_deriv[ii][jj][0][2],
                        pair_tilde_r_deriv[ii][jj][1][0],
                        pair_tilde_r_deriv[ii][jj][1][1],
                        pair_tilde_r_deriv[ii][jj][1][2],
                        pair_tilde_r_deriv[ii][jj][2][0],
                        pair_tilde_r_deriv[ii][jj][2][1],
                        pair_tilde_r_deriv[ii][jj][2][2],
                        pair_tilde_r_deriv[ii][jj][3][0],
                        pair_tilde_r_deriv[ii][jj][3][1],
                        pair_tilde_r_deriv[ii][jj][3][2]                
                );
            }
        }

        // Step . Free memory
        for (int ii=0; ii<this->num_center_atoms; ii++) {
            for (int jj=0; jj<this->num_neigh_atoms; jj++) {
                for (int kk=0; kk<4; kk++) {
                    free(pair_tilde_r_deriv[ii][jj][kk]);
                }
                free(pair_tilde_r_deriv[ii][jj]);
            }
            free(pair_tilde_r_deriv[ii]);
        }
        free(pair_tilde_r_deriv);
    }
}


/**
 * @brief 指定 `center_atomic_number`, `neigh_atomic_number`, 计算最大近邻原子数
 * 
 * @tparam CoordType 
 * @return const int 
 */
template <typename CoordType>
const int PairTildeR<CoordType>::get_max_num_neigh_atoms() const {
    return this->neighbor_list.get_max_num_neigh_atoms_ssss(this->center_atomic_number, this->neigh_atomic_number);
}



/**
 * @brief 
 * 
 * @tparam CoordType 
 * @return CoordType***     : shape = (num_center_atoms, num_neigh_atoms, 4)
 */
template <typename CoordType>
CoordType*** PairTildeR<CoordType>::generate() const {
    // Step 1. 
    // Step 1.1. $\tilde{R}$ = (s(r_{ji}), x_{ji}, y_{ji}, z_{ji})
    //  = (tilde_s_value, tilde_x_value, tilde_y_value, tilde_z_value)
    CoordType tilde_s_value;
    CoordType tilde_x_value;
    CoordType tilde_y_value;
    CoordType tilde_z_value;

    int prim_num_atoms = this->neighbor_list.get_binLinkedList().get_supercell().get_prim_num_atoms();
    int prim_cell_idx = this->neighbor_list.get_binLinkedList().get_supercell().get_prim_cell_idx();

    int center_atom_idx;            // 中心原子在 supercell 中的索引
    int neigh_atom_idx;             // 近邻原子在 supercell 中的索引
    const CoordType* center_cart_coord;   // 中心原子的坐标
    const CoordType* neigh_cart_coord;    // 近邻原子的坐标
    CoordType* diff_cart_coord = (CoordType*)malloc(sizeof(CoordType) * 3);     // 近邻原子的坐标 - 中心原子的坐标
    CoordType distance_ji;          // 两原子间的距离
    CoordType distance_ji_recip;

    // Step 1.2. Allocate memory for $\tilde{R}$
    CoordType*** pair_tilde_r = (CoordType***)malloc(sizeof(CoordType**) * this->num_center_atoms);
    for (int ii=0; ii<this->num_center_atoms; ii++) {
        pair_tilde_r[ii] = (CoordType**)malloc(sizeof(CoordType*) * this->num_neigh_atoms);
        for (int jj=0; jj<this->num_neigh_atoms; jj++) {
            pair_tilde_r[ii][jj] = (CoordType*)malloc(sizeof(CoordType) * 4);
        }
    }

    for (int ii=0; ii<this->num_center_atoms; ii++) {
        for (int jj=0; jj<this->num_neigh_atoms; jj++) {
            for (int kk=0; kk<4; kk++) {
                pair_tilde_r[ii][jj][kk] = 0;
            }
        }
    }

    // Step 2. 
    const CoordType** supercell_cart_coords = this->neighbor_list.get_binLinkedList().get_supercell().get_structure().get_cart_coords();
    const int* supercell_atomic_numbers = this->neighbor_list.get_binLinkedList().get_supercell().get_structure().get_atomic_numbers();

    // Step 3. Populate `tilde_s/x/y/z_value`
    int tmp_cidx;   // 中心原子 for loop
    int tmp_nidx;   // 近邻原子 for loop

    tmp_cidx = 0;
    for (int ii=0; ii<this->neighbor_list.get_num_center_atoms(); ii++) {
        center_atom_idx = ii + prim_cell_idx * prim_num_atoms;
        center_cart_coord = supercell_cart_coords[center_atom_idx];
        // 若中心原子 不等于 `this->center_atomic_number`，直接跳过
        if (supercell_atomic_numbers[center_atom_idx] != this->center_atomic_number)
            continue;
        
        tmp_nidx = 0;
        for (int jj=0; jj<this->neighbor_list.get_neighbor_lists()[ii].size(); jj++) {
            neigh_atom_idx = this->neighbor_list.get_neighbor_lists()[ii][jj];
            
            // 若近邻原子 不等于 `this->neigh_atomic_number`，直接跳过
            if (supercell_atomic_numbers[neigh_atom_idx] != this->neigh_atomic_number)
                continue;

            // Step 3.1. 计算 1/r (`distance_ji_recip`), s(r_ji) (`tilde_s_value`)
            neigh_cart_coord = supercell_cart_coords[neigh_atom_idx];
            for (int kk=0; kk<3; kk++)
                diff_cart_coord[kk] = neigh_cart_coord[kk] - center_cart_coord[kk];
            distance_ji = vec3Operation::norm(diff_cart_coord);
            distance_ji_recip = recip(distance_ji);
            tilde_s_value = smooth_func(distance_ji, this->rcut, this->rcut_smooth);

            // Step 3.2. 计算 `x_ji_s`, `y_ji_s`, `z_ji_s` 
            tilde_x_value = tilde_s_value * distance_ji_recip * diff_cart_coord[0];
            tilde_y_value = tilde_s_value * distance_ji_recip * diff_cart_coord[1];
            tilde_z_value = tilde_s_value * distance_ji_recip * diff_cart_coord[2];

            // Step 3.3. Assignment
            pair_tilde_r[tmp_cidx][tmp_nidx][0] = tilde_s_value;
            pair_tilde_r[tmp_cidx][tmp_nidx][1] = tilde_x_value;
            pair_tilde_r[tmp_cidx][tmp_nidx][2] = tilde_y_value;
            pair_tilde_r[tmp_cidx][tmp_nidx][3] = tilde_z_value;
            //printf("[%d, %d] -- [%f, %f, %f, %f]\n", tmp_cidx, tmp_nidx, pair_tilde_r[tmp_cidx][tmp_nidx][0], pair_tilde_r[tmp_cidx][tmp_nidx][1], pair_tilde_r[tmp_cidx][tmp_nidx][2], pair_tilde_r[tmp_cidx][tmp_nidx][3]);

            tmp_nidx++;
        }

        tmp_cidx++;
    }

    assert(tmp_cidx == this->num_center_atoms);
    //printf("%d, %d\n", tmp_nidx, this->num_neigh_atoms);
    //assert(tmp_nidx == this->num_neigh_atoms);

    // Step . Free memory
    free(diff_cart_coord);

    return pair_tilde_r;
}


// For lammps
template <typename CoordType>
CoordType*** PairTildeR<CoordType>::generate(
                        int inum,
                        int* ilist,
                        int* numneigh,
                        int** firstneigh,
                        CoordType** x,
                        int* types,
                        int center_atomic_number,
                        int neigh_atomic_number,
                        int num_neigh_atoms,
                        CoordType rcut,
                        CoordType rcut_smooth)
{
    // Step 1.
    // Step 1.1. $\tilde{R}$ = (s(r_{ji}), x_{ji}, y_{ji}, z_{ji})
    //  = (tilde_s_value, tilde_x_value, tilde_y_value, tilde_z_value)
    CoordType tilde_s_value;
    CoordType tilde_x_value;
    CoordType tilde_y_value;
    CoordType tilde_z_value;

    int center_atom_idx;    // 中心原子在 supercell 中的索引
    int neigh_atom_idx;     // 近邻原子在 supercell 中的索引
    CoordType* center_cart_coord;   // 中心原子的坐标
    CoordType* neigh_cart_coord;    // 近邻原子的坐标
    CoordType* diff_cart_coord = (CoordType*)malloc(sizeof(CoordType) * 3);
    CoordType distance_ji;  // 两原子间的距离
    CoordType distance_ji_recip;

    // Step 1.2. Allocate memory for $\tilde{R}$ and assign it as 0
    int num_center_atoms = 0;
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        if (types[center_atom_idx] == center_atomic_number)
            num_center_atoms += 1;
    }
    CoordType*** pair_tilde_r = arrayUtils::allocate3dArray<CoordType>(num_center_atoms, num_neigh_atoms, 4, true);

    // Step 1.3. 计算最大近邻原子数 (可以注释掉提高速度)
    int max_num_neigh_atoms = 0;
    int tmp_max_num_neigh_atoms = 0;
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        if (types[center_atom_idx] == center_atomic_number) {
            tmp_max_num_neigh_atoms = 0;
            for (int jj=0; jj<numneigh[ii]; jj++) {
                neigh_atom_idx = firstneigh[ii][jj];
                if (types[neigh_atom_idx] == neigh_atomic_number) 
                    tmp_max_num_neigh_atoms++;
            }
            if (tmp_max_num_neigh_atoms > max_num_neigh_atoms)
                max_num_neigh_atoms = tmp_max_num_neigh_atoms;
        }
    }


    // Step 2. 获取 supercell 中所有原子的`坐标`和`原子序数
    // 坐标 : x
    // 原子序数 : types

    // Step 3. Populate `tilde_s/x/y/z_value`
    int tmp_cidx;   // 中心原子 for loop
    int tmp_nidx;   // 近邻原子 for loop

    tmp_cidx = 0;
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        center_cart_coord = x[center_atom_idx];
        // 若中心原子 不等于 `center_atomic_number`, 直接跳过
        if (types[center_atom_idx] != center_atomic_number)
            continue;

        assert(num_neigh_atoms >= max_num_neigh_atoms);    // 防止设置的zero-padding尺寸太小
        
        tmp_nidx = 0;
        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_atom_idx = firstneigh[ii][jj];

            // 若近邻原子 不等于 `neigh_atomic_number`，直接跳过
            if (types[neigh_atom_idx] != neigh_atomic_number)
                continue;

            // Step 3.1. 计算 1/r (`distance_ji_recip`), s(r_ji) (`tilde_s_value`)
            neigh_cart_coord = x[neigh_atom_idx];
            for (int kk=0; kk<3; kk++) 
                diff_cart_coord[kk] = neigh_cart_coord[kk] - center_cart_coord[kk];
            distance_ji = vec3Operation::norm(diff_cart_coord);
            distance_ji_recip = recip(distance_ji);
            tilde_s_value = smooth_func(distance_ji, rcut, rcut_smooth);

            // Step 3.2. 计算 `x_ji_s`, `y_ji_s`, `z_ji_s` 
            tilde_x_value = tilde_s_value * distance_ji_recip * diff_cart_coord[0];
            tilde_y_value = tilde_s_value * distance_ji_recip * diff_cart_coord[1];
            tilde_z_value = tilde_s_value * distance_ji_recip * diff_cart_coord[2];

            // Step 3.3. Assignment
            pair_tilde_r[tmp_cidx][tmp_nidx][0] = tilde_s_value;
            pair_tilde_r[tmp_cidx][tmp_nidx][1] = tilde_x_value;
            pair_tilde_r[tmp_cidx][tmp_nidx][2] = tilde_y_value;
            pair_tilde_r[tmp_cidx][tmp_nidx][3] = tilde_z_value;

            tmp_nidx++;
        }
        //printf("***%d, %d\n", tmp_nidx, numneigh[ii]);
        //assert(tmp_nidx == numneigh[ii]);
        tmp_cidx++;
    }
    //printf("***%d, %d\n", tmp_cidx, inum);
    //assert(tmp_cidx == inum);

    // Step . Free memory
    free(diff_cart_coord);

    return pair_tilde_r;
}


/**
 * @brief Calculate the gradient of with $\tilde{R}$ respect to x, y and z
 * 
 * @tparam CoordType 
 * @return CoordType****    shape = (num_center_atoms, num_neigh_atoms, 4, 3)
 */
template <typename CoordType>
CoordType**** PairTildeR<CoordType>::deriv() const {
    // Step 1. 初始化一些必要的变量
    // Step 1.1. 
    int prim_cell_idx = this->neighbor_list.get_binLinkedList().get_supercell().get_prim_cell_idx();
    int prim_num_atoms = this->neighbor_list.get_binLinkedList().get_supercell().get_prim_num_atoms();

    int center_atom_idx;        // 中心原子在 supercell 中的 index
    int neigh_atom_idx;         // 近邻原子在 supercell 中的 index
    const CoordType* center_cart_coord; // 中心原子在 supercell 中的笛卡尔坐标
    const CoordType* neigh_cart_coord;  // 近邻原子在 supercell 中的笛卡尔坐标
    CoordType* diff_cart_coord = (CoordType*)malloc(sizeof(CoordType) * 3); // 近邻原子 - 中心原子 的相对坐标
    CoordType distance_ji;          // 中心原子与近邻原子的距离
    CoordType distance_ji_recip;    // 中心原子与近邻原子的距离的倒数

    // Step 1.2. Allocate memory for `pair_tilde_r_deriv` and assign 0
    CoordType**** pair_tilde_r_deriv = (CoordType****)malloc(sizeof(CoordType***) * this->num_center_atoms);
    for (int ii=0; ii<this->num_center_atoms; ii++) {
        pair_tilde_r_deriv[ii] = (CoordType***)malloc(sizeof(CoordType**) * this->num_neigh_atoms);
        for (int jj=0; jj<this->num_neigh_atoms; jj++) {
            pair_tilde_r_deriv[ii][jj] = (CoordType**)malloc(sizeof(CoordType*) * 4);
            for (int kk=0; kk<4; kk++) {
                pair_tilde_r_deriv[ii][jj][kk] = (CoordType*)malloc(sizeof(CoordType) * 3);
            }
        }
    }
    // Assign 0.
    for (int ii=0; ii<this->num_center_atoms; ii++) {
        for (int jj=0; jj<this->num_neigh_atoms; jj++) {
            for (int kk=0; kk<4; kk++) {
                for (int ll=0; ll<3; ll++) 
                    pair_tilde_r_deriv[ii][jj][kk][ll] = 0;
            }
        }
    }

    // Step 1.3. 
    SwitchFunc<CoordType> switch_func(this->rcut, this->rcut_smooth);

    // Step 2. 存储 `supercell_cart_coords` && `atomic_numbers`
    const CoordType** supercell_cart_coords = this->neighbor_list.get_binLinkedList().get_supercell().get_structure().get_cart_coords();
    const int* supercell_atomic_numbers = this->neighbor_list.get_binLinkedList().get_supercell().get_structure().get_atomic_numbers();

    // Step 3. Populate `pair_tilde_r_deriv`
    int tmp_cidx = 0;
    for (int ii=0; ii<this->neighbor_list.get_num_center_atoms(); ii++) {  // 遍历中心原子
        center_atom_idx = ii + prim_cell_idx * prim_num_atoms;
        if (supercell_atomic_numbers[center_atom_idx] != this->center_atomic_number)
            continue;
        center_cart_coord = supercell_cart_coords[center_atom_idx];

        int tmp_nidx = 0;
        for (int jj=0; jj<this->neighbor_list.get_neighbor_lists()[ii].size(); jj++) {    // 遍历近邻原子
            neigh_atom_idx = this->neighbor_list.get_neighbor_lists()[ii][jj];
            if (supercell_atomic_numbers[neigh_atom_idx] != this->neigh_atomic_number)
                continue;
            neigh_cart_coord = supercell_cart_coords[neigh_atom_idx];

            // Step 3.1. 给一些常用于计算导数的变量赋值
            for (int kk=0; kk<3; kk++)
                diff_cart_coord[kk] = neigh_cart_coord[kk] - center_cart_coord[kk];
            distance_ji = vec3Operation::norm(diff_cart_coord);
            distance_ji_recip = recip(distance_ji);
        
            /*
                1. smooth func = s(r) = \frac{1}{r} \cdot switch_func
                2. s(r) = \frac{1}{r} \cdot switch_func -- 需要分步求导
            */
            // Step 3.2.1. s(r) = switchFunc(r) * $\frac{1}{r}$
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][0][0] = (
                switch_func.get_result(distance_ji) * diff_cart_coord[0] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coord[0] * std::pow(distance_ji_recip, 2)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][0][1] = (
                switch_func.get_result(distance_ji) * diff_cart_coord[1] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coord[1] * std::pow(distance_ji_recip, 2)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][0][2] = (
                switch_func.get_result(distance_ji) * diff_cart_coord[2] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coord[2] * std::pow(distance_ji_recip, 2)
            );
        
            // Step 3.2.2. s(r)x/r = switchFunc(r) * $\frac{x}{r^2}$
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][1][0] = (
                2 * std::pow(diff_cart_coord[0], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coord[0], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][1][1] = (
                2 * diff_cart_coord[0] * diff_cart_coord[1] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[0] * diff_cart_coord[1] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][1][2] = (
                2 * diff_cart_coord[0] * diff_cart_coord[2] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[0] * diff_cart_coord[2] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );

            // Step 3.2.3. s(r)y/r = switchFunc(r) * $\frac{y}{r^2}$
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][2][0] = (
                2 * diff_cart_coord[1] * diff_cart_coord[0] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[1] * diff_cart_coord[0] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][2][1] = (
                2 * std::pow(diff_cart_coord[1], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coord[1], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][2][2] = (
                2 * diff_cart_coord[1] * diff_cart_coord[2] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[1] * diff_cart_coord[2] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );

            // Step 3.2.4. s(r)z/r = switchFunc(r) * $\frac{z}{r^2}$
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][3][0] = (
                2 * diff_cart_coord[2] * diff_cart_coord[0] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[2] * diff_cart_coord[0] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][3][1] = (
                2 * diff_cart_coord[2] * diff_cart_coord[1] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[2] * diff_cart_coord[1] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][3][2] = (
                2 * std::pow(diff_cart_coord[2], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coord[2], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            tmp_nidx++;
        }
        tmp_cidx++;
    }

    // Step . Free memory
    free(diff_cart_coord);

    return pair_tilde_r_deriv;
}



template <typename CoordType>
CoordType**** PairTildeR<CoordType>::deriv(
                    int inum, 
                    int* ilist,
                    int* numneigh,
                    int** firstneigh,
                    CoordType** x,
                    int* types,
                    int center_atomic_number,
                    int neigh_atomic_number,
                    int num_neigh_atoms,
                    CoordType rcut,
                    CoordType rcut_smooth)
{
    // Step 1. 初始化一些必要的变量
    // Step 1.1. 
    int center_atom_idx;            // 中心原子在 supercell 中的 index
    int neigh_atom_idx;             // 近邻原子在 supercell 中的 index
    CoordType* center_cart_coord;   // 中心原子在 supercell 中的坐标
    CoordType* neigh_cart_coord;    // 近邻原子在 supercell 中的坐标
    CoordType* diff_cart_coord = (CoordType*)malloc(sizeof(CoordType) * 3); // 近邻原子 - 中心原子 的相对坐标
    CoordType distance_ji;          // 中心原子与近邻原子的距离
    CoordType distance_ji_recip;    // 中心原子与近邻原子距离的倒数

    // Step 1.2. Allocate memory for `pair_tilde_r_deriv` and assign it as 0
    int num_center_atoms = 0;       // 满足 `center_atomic_number` 的中心原子数
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        if (types[center_atom_idx] == center_atomic_number)
            num_center_atoms++;
    }
    CoordType**** pair_tilde_r_deriv = arrayUtils::allocate4dArray<CoordType>(num_center_atoms, num_neigh_atoms, 4, 3, true);

    // Step 1.3. 
    SwitchFunc<CoordType> switch_func(rcut, rcut_smooth);

    // Step 2. 存储 `supercell_cart_coords` && `atomic_numbers`
    // x
    // types

    // Step 2.2. 计算得到最大的近邻原子数
    int max_num_neigh_atoms = 0;
    int tmp_max_num_neigh_atoms = 0;
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        tmp_max_num_neigh_atoms = 0;
        if (types[center_atom_idx] == center_atomic_number) {
            tmp_max_num_neigh_atoms = 0;
            for (int jj=0; jj<numneigh[ii]; jj++) {
                neigh_atom_idx = firstneigh[ii][jj];
                if (types[neigh_atom_idx] == neigh_atomic_number)
                    tmp_max_num_neigh_atoms++;
            }

            if (tmp_max_num_neigh_atoms > max_num_neigh_atoms)
                max_num_neigh_atoms = tmp_max_num_neigh_atoms;
        }
    }

    // Step 3. Populate `pair_tilde_r_deriv`
    int tmp_cidx = 0;
    int tmp_nidx = 0;
    for (int ii=0; ii<inum; ii++) {
        center_atom_idx = ilist[ii];
        if (types[center_atom_idx] != center_atomic_number)
            continue;
        center_cart_coord = x[center_atom_idx];

        assert(num_neigh_atoms >= max_num_neigh_atoms);

        tmp_nidx = 0;
        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_atom_idx = firstneigh[ii][jj];
            if (types[neigh_atom_idx] != neigh_atomic_number)
                continue;
            neigh_cart_coord = x[neigh_atom_idx];
            
            // Step 3.1. 给一些冲用于计算导数的变量赋值
            for (int kk=0; kk<3; kk++) 
                diff_cart_coord[kk] = neigh_cart_coord[kk] - center_cart_coord[kk];
            distance_ji = vec3Operation::norm(diff_cart_coord);
            distance_ji_recip = recip(distance_ji);
            
            /*
                1. smooth func = s(r) = \frac{1}{r} \cdot switch_func
                2. s(r) = \frac{1}{r} \cdot switch_func -- 需要分步求导
            */
            // Step 3.2.1. s(r) = switchFunc(r) * $\frac{1}{r}$
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][0][0] = (
                switch_func.get_result(distance_ji) * diff_cart_coord[0] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coord[0] * std::pow(distance_ji_recip, 2)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][0][1] = (
                switch_func.get_result(distance_ji) * diff_cart_coord[1] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coord[1] * std::pow(distance_ji_recip, 2)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][0][2] = (
                switch_func.get_result(distance_ji) * diff_cart_coord[2] * std::pow(distance_ji_recip, 3) - \
                switch_func.get_deriv2r(distance_ji) * diff_cart_coord[2] * std::pow(distance_ji_recip, 2)
            );
        
            // Step 3.2.2. s(r)x/r = switchFunc(r) * $\frac{x}{r^2}$
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][1][0] = (
                2 * std::pow(diff_cart_coord[0], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coord[0], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][1][1] = (
                2 * diff_cart_coord[0] * diff_cart_coord[1] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[0] * diff_cart_coord[1] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][1][2] = (
                2 * diff_cart_coord[0] * diff_cart_coord[2] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[0] * diff_cart_coord[2] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );

            // Step 3.2.3. s(r)y/r = switchFunc(r) * $\frac{y}{r^2}$
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][2][0] = (
                2 * diff_cart_coord[1] * diff_cart_coord[0] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[1] * diff_cart_coord[0] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][2][1] = (
                2 * std::pow(diff_cart_coord[1], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coord[1], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][2][2] = (
                2 * diff_cart_coord[1] * diff_cart_coord[2] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[1] * diff_cart_coord[2] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );

            // Step 3.2.4. s(r)z/r = switchFunc(r) * $\frac{z}{r^2}$
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][3][0] = (
                2 * diff_cart_coord[2] * diff_cart_coord[0] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[2] * diff_cart_coord[0] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][3][1] = (
                2 * diff_cart_coord[2] * diff_cart_coord[1] * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                diff_cart_coord[2] * diff_cart_coord[1] * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            pair_tilde_r_deriv[tmp_cidx][tmp_nidx][3][2] = (
                2 * std::pow(diff_cart_coord[2], 2) * switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 4) - \
                switch_func.get_result(distance_ji) * std::pow(distance_ji_recip, 2) - \
                std::pow(diff_cart_coord[2], 2) * std::pow(distance_ji_recip, 3) * switch_func.get_deriv2r(distance_ji)
            );
            tmp_nidx++;
        }
        tmp_cidx++;
    }
    assert(tmp_cidx == num_center_atoms);

    // Step . Free memory
    free(diff_cart_coord);

    return pair_tilde_r_deriv;
} 



/**
 * @brief Construct a new Tilde R< Coord Type>:: Tilde R object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
TildeR<CoordType>::TildeR() {
    this->neighbor_list = NeighborList<CoordType>();
    this->num_center_atomic_numbers = 0;
    this->center_atomic_numbers_lst = nullptr;
    this->num_neigh_atomic_numbers = 0;
    this->neigh_atomic_numbers_lst = nullptr;
    this->num_center_atoms_lst = nullptr;
    this->num_neigh_atoms_lst = nullptr;
    this->rcut = 0;
    this->rcut_smooth = 0;
}


/**
 * @brief Construct a new Tilde R< Coord Type>:: Tilde R object
 * 
 * @tparam CoordType 
 * @param neighbor_list matersdk::NeighborList<CoordType> object
 * @param num_center_atomic_numbers 中心原子的种类数
 * @param center_atomic_numbers_lst 中心原子的种类
 * @param num_neigh_atomic_numbers 近邻原子的种类数
 * @param neigh_atomic_numbers_lst 近邻原子的种类
 * @param num_neigh_atoms_lst 
 * @param rcut_smooth 
 */
template <typename CoordType>
TildeR<CoordType>::TildeR(
                    NeighborList<CoordType>& neighbor_list,
                    int num_center_atomic_numbers,
                    int* center_atomic_numbers_lst, 
                    int num_neigh_atomic_numbers,
                    int* neigh_atomic_numbers_lst, 
                    int* num_neigh_atoms_lst,
                    CoordType rcut_smooth)
{
    // Step 1. Init `TildeR` member variable
    this->neighbor_list = neighbor_list;

    this->num_center_atomic_numbers = num_center_atomic_numbers;
    this->center_atomic_numbers_lst = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        this->center_atomic_numbers_lst[ii] = center_atomic_numbers_lst[ii];

    this->num_neigh_atomic_numbers = num_neigh_atomic_numbers;
    this->neigh_atomic_numbers_lst = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        this->neigh_atomic_numbers_lst[ii] = neigh_atomic_numbers_lst[ii];
    
    this->calc_num_center_atoms_lst();  // 计算 `this->num_center_atoms_lst`

    this->num_neigh_atoms_lst = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        this->num_neigh_atoms_lst[ii] = num_neigh_atoms_lst[ii];

    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


/**
 * @brief Construct a new Tilde R< Coord Type>:: Tilde R object
 * 
 * @tparam CoordType 
 * @param neighbor_list 
 * @param num_center_atomic_numbers 
 * @param center_atomic_numbers_lst 
 * @param num_neigh_atomic_numbers 
 * @param neigh_atomic_numbers 
 * @param rcut_smooth 
 */
template <typename CoordType>
TildeR<CoordType>::TildeR(
                    NeighborList<CoordType>& neighbor_list,
                    int num_center_atomic_numbers,
                    int* center_atomic_numbers_lst,
                    int num_neigh_atomic_numbers,
                    int* neigh_atomic_numbers,
                    CoordType rcut_smooth)
{
    this->neighbor_list = neighbor_list;

    this->num_center_atomic_numbers = num_center_atomic_numbers;
    this->center_atomic_numbers_lst = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        this->center_atomic_numbers_lst[ii] = center_atomic_numbers_lst[ii];
    
    this->num_neigh_atomic_numbers = num_neigh_atomic_numbers;
    this->neigh_atomic_numbers_lst = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        this->neigh_atomic_numbers_lst[ii] = neigh_atomic_numbers[ii];

    this->calc_num_center_atoms_lst();  // 计算 `this->num_center_atoms_lst`
    this->calc_num_neigh_atoms_lst();   // 计算 `this->num_neigh_atoms_lst`
    
    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


/**
 * @brief Construct a new Tilde R< Coord Type>:: Tilde R object
 * 
 * @tparam CoordType 
 * @param structure matersdk::Structure object
 * @param rcut 截断半径
 * @param pbc_xyz 周期性边界条件
 * @param sort 原子是否按照距中心原子距离排序
 * @param num_center_atomic_numbers 
 * @param center_atomic_numbers_lst 
 * @param num_neigh_atomic_numbers 
 * @param neigh_atomic_numbers_lst 
 * @param num_neigh_atoms_lst 
 * @param rcut_smooth 
 */
template <typename CoordType>
TildeR<CoordType>::TildeR(
                    Structure<CoordType>& structure,
                    CoordType rcut,
                    bool* pbc_xyz,
                    bool sort,
                    int num_center_atomic_numbers,
                    int* center_atomic_numbers_lst,
                    int num_neigh_atomic_numbers,
                    int* neigh_atomic_numbers_lst,
                    int* num_neigh_atoms_lst,
                    CoordType rcut_smooth)
{
    this->neighbor_list = NeighborList<CoordType>(structure, rcut, pbc_xyz, sort);
    
    this->num_center_atomic_numbers = num_center_atomic_numbers;
    this->center_atomic_numbers_lst = (int*)malloc(sizeof(int) * this->num_center_atomic_numbers);
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        this->center_atomic_numbers_lst[ii] = center_atomic_numbers_lst[ii];
    
    this->num_neigh_atomic_numbers = num_neigh_atomic_numbers;
    this->neigh_atomic_numbers_lst = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++) 
        this->neigh_atomic_numbers_lst[ii] = neigh_atomic_numbers_lst[ii];
    
    this->calc_num_center_atoms_lst();  // 计算 `this->num_center_atoms_lst`

    this->num_neigh_atoms_lst = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        this->num_neigh_atoms_lst[ii] = num_neigh_atoms_lst[ii];

    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


/**
 * @brief Construct a new Tilde R< Coord Type>:: Tilde R object
 * 
 * @tparam CoordType 
 * @param structure 
 * @param rcut 
 * @param pbc_xyz 
 * @param sort 
 * @param num_center_atomic_numbers 
 * @param center_atomic_numbers_lst 
 * @param num_neigh_atomic_numbers 
 * @param neigh_atomic_numbers_lst 
 * @param rcut_smooth 
 */
template <typename CoordType>
TildeR<CoordType>::TildeR(
                    Structure<CoordType>& structure,
                    CoordType rcut, 
                    bool* pbc_xyz,
                    bool sort,
                    int num_center_atomic_numbers,
                    int* center_atomic_numbers_lst,
                    int num_neigh_atomic_numbers,
                    int* neigh_atomic_numbers_lst,
                    CoordType rcut_smooth)
{
    this->neighbor_list = NeighborList<CoordType>(structure, rcut, pbc_xyz, sort);
    
    this->num_center_atomic_numbers = num_center_atomic_numbers;
    this->center_atomic_numbers_lst = (int*)malloc(sizeof(int) * this->num_center_atomic_numbers);
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        this->center_atomic_numbers_lst[ii] = center_atomic_numbers_lst[ii];

    this->num_neigh_atomic_numbers = num_neigh_atomic_numbers;
    this->neigh_atomic_numbers_lst = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        this->neigh_atomic_numbers_lst[ii] = neigh_atomic_numbers_lst[ii];
    
    this->calc_num_center_atoms_lst();  // 计算 `this->num_center_atoms_lst`
    this->calc_num_neigh_atoms_lst();   // 计算 `this->num_neigh_atoms_lst`

    this->rcut = this->neighbor_list.get_rcut();
    this->rcut_smooth = rcut_smooth;
}


/**
 * @brief Copy Constructor
 * 
 * @tparam CoordType 
 * @param rhs 
 */
template <typename CoordType>
TildeR<CoordType>::TildeR(const TildeR<CoordType>& rhs) {
    this->neighbor_list = rhs.neighbor_list;
    
    this->num_center_atomic_numbers = rhs.num_center_atomic_numbers;
    this->center_atomic_numbers_lst = (int*)malloc(sizeof(this->num_center_atomic_numbers));
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        this->center_atomic_numbers_lst[ii] = rhs.center_atomic_numbers_lst[ii];
    
    this->num_neigh_atomic_numbers = rhs.num_neigh_atomic_numbers;
    this->neigh_atomic_numbers_lst = (int*)malloc(sizeof(this->num_neigh_atomic_numbers));
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        this->neigh_atomic_numbers_lst[ii] = rhs.neigh_atomic_numbers_lst[ii];

    this->num_center_atoms_lst = (int*)malloc(sizeof(int) * this->num_center_atomic_numbers);
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        this->num_center_atoms_lst[ii] = rhs.num_center_atoms_lst[ii];

    this->num_neigh_atoms_lst = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        this->num_neigh_atoms_lst[ii] = rhs.num_neigh_atoms_lst[ii];

    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;
}


/**
 * @brief Overloading the assignment operator
 * 
 * @tparam CoordType 
 * @param rhs 
 * @return TildeR<CoordType>& 
 */
template <typename CoordType>
TildeR<CoordType>& TildeR<CoordType>::operator=(const TildeR<CoordType>& rhs) {
    if (this->num_center_atomic_numbers != 0) {
        free(this->center_atomic_numbers_lst);
        free(this->num_center_atoms_lst);
        free(this->neigh_atomic_numbers_lst);
        free(this->num_neigh_atoms_lst);

        this->num_center_atomic_numbers = 0;
        this->num_neigh_atomic_numbers = 0;
    }

    this->neighbor_list = rhs.neighbor_list;
    this->num_center_atomic_numbers = rhs.num_center_atomic_numbers;
    this->num_neigh_atomic_numbers = rhs.num_neigh_atomic_numbers;
    
    if (this->num_center_atomic_numbers != 0) {
        this->center_atomic_numbers_lst = (int*)malloc(sizeof(int) * this->num_center_atomic_numbers);
        this->num_center_atoms_lst = (int*)malloc(sizeof(int) * this->num_center_atomic_numbers);
        for (int ii=0; ii<this->num_center_atomic_numbers; ii++) {
            this->center_atomic_numbers_lst[ii] = rhs.center_atomic_numbers_lst[ii];
            this->num_center_atoms_lst[ii] = rhs.num_center_atoms_lst[ii];
        }
    }

    if (this->num_neigh_atomic_numbers != 0) {
        this->neigh_atomic_numbers_lst = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
        this->num_neigh_atoms_lst = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
        for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++) {
            this->neigh_atomic_numbers_lst[ii] = rhs.neigh_atomic_numbers_lst[ii];
            this->num_neigh_atoms_lst[ii] = rhs.num_neigh_atoms_lst[ii];
        }
    }
    
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;

    return *this;
}


template <typename CoordType>
TildeR<CoordType>::~TildeR() {
    if (this->num_center_atomic_numbers != 0) {
        free(this->center_atomic_numbers_lst);
        free(this->num_center_atoms_lst);
    }

    if (this->num_neigh_atomic_numbers != 0) {
        free(this->neigh_atomic_numbers_lst);
        free(this->num_neigh_atoms_lst);
    }

    this->num_center_atomic_numbers = 0;
    this->num_neigh_atomic_numbers = 0;
}


template <typename CoordType>
void TildeR<CoordType>::calc_num_center_atoms_lst() {
    int tmp_num_center_atoms;
    const int* supercell_atomic_numbers = this->neighbor_list.get_binLinkedList().get_supercell().get_structure().get_atomic_numbers();
    this->num_center_atoms_lst = (int*)malloc(sizeof(int) * this->num_center_atomic_numbers);
    
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++) {
        tmp_num_center_atoms = 0;
        for (int jj=0; jj<this->neighbor_list.get_num_center_atoms(); jj++) {
            if (supercell_atomic_numbers[jj] == this->center_atomic_numbers_lst[ii])
                tmp_num_center_atoms++;
        }
        this->num_center_atoms_lst[ii] = tmp_num_center_atoms;
    }
}


template <typename CoordType>
void TildeR<CoordType>::calc_num_neigh_atoms_lst() {
    int tmp_num_neigh_atoms;
    const int* supercell_atomic_numbers = this->neighbor_list.get_binLinkedList().get_supercell().get_structure().get_atomic_numbers();
    this->num_neigh_atoms_lst = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        this->num_neigh_atoms_lst[ii] = 0;

    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++) {
        for (int jj=0; jj<this->neighbor_list.get_num_center_atoms(); jj++) {
            tmp_num_neigh_atoms = 0;
            for (int kk=0; kk<this->neighbor_list.get_neighbor_lists()[jj].size(); kk++) {
                if (supercell_atomic_numbers[this->neighbor_list.get_neighbor_lists()[jj][kk]] == this->neigh_atomic_numbers_lst[ii])
                    tmp_num_neigh_atoms++;
            }
            if (tmp_num_neigh_atoms > this->num_neigh_atoms_lst[ii])
                this->num_neigh_atoms_lst[ii] = tmp_num_neigh_atoms;
        }
    }
}


/**
 * @brief $\tilde{R}$ 的第一维 = sum(this->num_center_atoms_lst)
 * 
 * @tparam CoordType 
 * @return const int 
 */
template <typename CoordType>
const int TildeR<CoordType>::get_num_center_atoms() const {
    int tot_num_center_atoms = 0;
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        tot_num_center_atoms += this->num_center_atoms_lst[ii];
    return tot_num_center_atoms;
}


/**
 * @brief $\tilde{R}$ 的第一维 = sum(this->num_neigh_atoms_lst)
 * 
 * @tparam CoordType 
 * @return const int 
 */
template <typename CoordType>
const int TildeR<CoordType>::get_num_neigh_atoms() const {
    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        tot_num_neigh_atoms += this->num_neigh_atoms_lst[ii];
    return tot_num_neigh_atoms;
}


template <typename CoordType>
void TildeR<CoordType>::show() const {
    printf("*** TildeR Summary ***\n");

    if (this->num_center_atomic_numbers == 0) {
        printf("This is a NULL TildeR object.\n");
    } else {
        printf("center_atomic_numbers_lst: ");
        printf("[");
        for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
            printf("%4d, ", this->center_atomic_numbers_lst[ii]);
        printf("]\n");

        printf("neigh_atomic_numbers_lst: ");
        printf("[");
        for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
            printf("%4d, ", this->neigh_atomic_numbers_lst[ii]);
        printf("]\n");

        printf("rcut = %f\n", this->rcut);
        printf("rcut_smooth = %f\n", this->rcut_smooth);

        printf("num_center_atoms_lst: ");
        printf("[");
        for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
            printf("%5d, ", this->num_center_atoms_lst[ii]);
        printf("]\n");

        printf("num_neigh_atoms_lst: ");
        printf("[");
        for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
            printf("%5d, ", this->num_neigh_atoms_lst[ii]);
        printf("]\n");

        printf("Shape of Tilde_R : [%d, %d, 4]\n", this->get_num_center_atoms(), this->get_num_neigh_atoms());
        printf("Shape of Tilde_R_derivative : [%d, %d, 4, 3]\n", this->get_num_center_atoms(), this->get_num_neigh_atoms());
    }
}


template <typename CoordType>
void TildeR<CoordType>::show_in_value() const {
    if (this->num_center_atomic_numbers == 0) {
        printf("This is a NULL TildeR object. So no value!\n");
    } else {
        // Step 1.
        int tot_num_center_atoms = 0;
        for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
            tot_num_center_atoms += this->num_center_atoms_lst[ii];
        
        int tot_num_neigh_atoms = 0;
        for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++) 
            tot_num_neigh_atoms += this->num_neigh_atoms_lst[ii];

        CoordType*** tilde_r = this->generate();

        // Step 2.
        for (int ii=0; ii<tot_num_center_atoms; ii++) {
            for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
                printf("[%4d, %4d] -- [%10f, %10f, %10f, %10f]\n", ii, jj, tilde_r[ii][jj][0], tilde_r[ii][jj][1], tilde_r[ii][jj][2], tilde_r[ii][jj][3]);
            }
        }

        // Step . Free memory
        arrayUtils::free3dArray(tilde_r, tot_num_center_atoms, tot_num_neigh_atoms);
    }
}


template <typename CoordType>
void TildeR<CoordType>::show_in_deriv() const {
    if (this->num_center_atomic_numbers == 0) {
        printf("This is a NULL TildeR object. So no derivative!\n");
    } else {
        // Step 1.
        int tot_num_center_atoms = 0;
        for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
            tot_num_center_atoms += this->num_center_atoms_lst[ii];
        
        int tot_num_neigh_atoms = 0;
        for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
            tot_num_neigh_atoms += this->num_neigh_atoms_lst[ii];
        
        CoordType**** tilde_r_deriv = this->deriv();

        for (int ii=0; ii<tot_num_center_atoms; ii++) {
            for (int jj=0; jj<tot_num_neigh_atoms; jj++) {
                printf("[%4d, %4d] -- [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f], [%10f, %10f, %10f]\n",
                    ii, jj,
                    tilde_r_deriv[ii][jj][0][0], tilde_r_deriv[ii][jj][0][1], tilde_r_deriv[ii][jj][0][2],
                    tilde_r_deriv[ii][jj][1][0], tilde_r_deriv[ii][jj][1][1], tilde_r_deriv[ii][jj][1][2],
                    tilde_r_deriv[ii][jj][2][0], tilde_r_deriv[ii][jj][2][1], tilde_r_deriv[ii][jj][2][2],
                    tilde_r_deriv[ii][jj][3][0], tilde_r_deriv[ii][jj][3][1], tilde_r_deriv[ii][jj][3][2]
                );
            }
        }

        // Step . Free memory
        arrayUtils::free4dArray(tilde_r_deriv, tot_num_center_atoms, tot_num_neigh_atoms, 4);
    }
}


/**
 * @brief Generate the $\tilde{R}$ for deepPot-SE. (组合 `PairTildeR`)
 * 
 * @tparam CoordType 
 * @return CoordType*** 
 */
template <typename CoordType>
CoordType*** TildeR<CoordType>::generate() const {
    // Step 1. Allocate memory for `tilde_r`. `tilde_r.shape = (num_center_atoms, num_neigh_atoms, 3)`
    int tot_num_center_atoms = 0;
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        tot_num_center_atoms += this->num_center_atoms_lst[ii];

    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        tot_num_neigh_atoms += this->num_neigh_atoms_lst[ii];

    CoordType*** tilde_r = arrayUtils::allocate3dArray<CoordType>(tot_num_center_atoms, tot_num_neigh_atoms, 4);


    // Step 2. Populate `tilde_r`
    // Step 2.1. 
    int tmp_center_atomic_number;           // 存储 `PairTildeR.center_atomic_number`
    int tmp_neigh_atomic_number;            // 存储 `PairTildeR.neigh_atomic_number`
    PairTildeR<CoordType> tmp_pair_tilde_r; // 存储 `PairTildeR`
    CoordType*** tmp_pair_tilde_r_value;    // 存储 `PairTildeR.generate()`
    int tmp_cidx;
    int tmp_nidx;

    // Step 2.2. 计算中心原子对应tilde_r[m, n, q] 的 m 索引起始；计算近邻原子对应的 tilde_r[m, n, q] 的 n 索引起始
    /*
    e.g.
    ----
        1. 12 原子的 MoS2:
            1. (4, 80, 4)       -- Mo-Mo
            2. (4, 100, 4)      -- Mo-S
            3. (8, 80, 4)       -- S-Mo
            4. (8, 1000, 4)     -- S-S
    */
    int* cstart_idxs = (int*)malloc(sizeof(int) * this->num_center_atomic_numbers);
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        cstart_idxs[ii] = 0;
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            cstart_idxs[ii] += this->num_center_atoms_lst[jj];
    }

    int* nstart_idxs = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        nstart_idxs[ii] = 0;
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            nstart_idxs[ii] += this->num_neigh_atoms_lst[jj];
    }
    /*
    test
    ----
        printf("cstart_idxs: ");
        for (int ii=0; ii<this->num_center_atomic_numbers; ii++) {
            printf("%d, ", cstart_idxs[ii]);
        }
        printf("\n");
        printf("nstart_idxs: ");
        for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++) {
            printf("%d, ", nstart_idxs[ii]);
        }
        printf("\n");
    */
    
    // Step 2.3. 
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++) {
        tmp_center_atomic_number = this->center_atomic_numbers_lst[ii];
        for (int jj=0; jj<this->num_neigh_atomic_numbers; jj++) {
            tmp_neigh_atomic_number = this->neigh_atomic_numbers_lst[jj];
            tmp_pair_tilde_r = PairTildeR<CoordType>(
                                        this->neighbor_list, 
                                        this->center_atomic_numbers_lst[ii], 
                                        this->neigh_atomic_numbers_lst[jj], 
                                        this->num_center_atoms_lst[ii], 
                                        this->num_neigh_atoms_lst[jj], 
                                        this->rcut_smooth
                                );
            tmp_pair_tilde_r_value = tmp_pair_tilde_r.generate();

            tmp_cidx = cstart_idxs[ii];         // 0, 4
            for (int kk=0; kk<this->num_center_atoms_lst[ii]; kk++) {
                tmp_nidx = nstart_idxs[jj];     // 0, 80
                for (int ll=0; ll<this->num_neigh_atoms_lst[jj]; ll++) {
                    tilde_r[tmp_cidx][tmp_nidx][0] = tmp_pair_tilde_r_value[kk][ll][0];
                    tilde_r[tmp_cidx][tmp_nidx][1] = tmp_pair_tilde_r_value[kk][ll][1];
                    tilde_r[tmp_cidx][tmp_nidx][2] = tmp_pair_tilde_r_value[kk][ll][2];
                    tilde_r[tmp_cidx][tmp_nidx][3] = tmp_pair_tilde_r_value[kk][ll][3];
                    tmp_nidx++;
                }
                tmp_cidx++;
            }

            // Step . Free memory
            arrayUtils::free3dArray<CoordType>(
                            tmp_pair_tilde_r_value, 
                            this->num_center_atoms_lst[ii], 
                            this->num_neigh_atoms_lst[jj]);
        }
    }

    // Step . Free memory
    free(cstart_idxs);
    free(nstart_idxs);
    
    return tilde_r;
}



template <typename CoordType>
CoordType*** TildeR<CoordType>::generate(
                    int inum,
                    int* ilist,
                    int* numneigh,
                    int** firstneigh,
                    CoordType** x,
                    int* types,
                    int num_center_atomic_numbers,
                    int* center_atomic_numbers_lst,
                    int num_neigh_atomic_numbers,
                    int* neigh_atomic_numbers_lst,
                    int* num_neigh_atoms_lst,
                    CoordType rcut,
                    CoordType rcut_smooth)
{
    // Step 1. Allocate memory for `tilde_r`
    // Step 1.1. 计算 `num_center_atoms_lst`
    int* num_center_atoms_lst = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<num_center_atomic_numbers; ii++)
        num_center_atoms_lst[ii] = 0;
    int center_atom_idx;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {    // 中心元素种类
        for (int jj=0; jj<inum; jj++) { // 中心原子
            center_atom_idx = ilist[jj];    // 中心原子在 supercell 中的索引
            if (types[center_atom_idx] == center_atomic_numbers_lst[ii])
                num_center_atoms_lst[ii]++;
        }
    }

    // Step 1.2. Allocate memory for `tilde_r` `tilde_r.shape = (num_center_atoms, num_neigh_atoms, 3)`
    int tot_num_center_atoms = 0;
    for (int ii=0; ii<num_center_atomic_numbers; ii++)
        tot_num_center_atoms += num_center_atoms_lst[ii];
    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++) 
        tot_num_neigh_atoms += num_neigh_atoms_lst[ii];
    CoordType*** tilde_r = arrayUtils::allocate3dArray<CoordType>(tot_num_center_atoms, tot_num_neigh_atoms, 4);


    // Step 2. Populate `tilde_r`
    // Step 2.1. 
    int tmp_center_atomic_number;               // 
    int tmp_neigh_atomic_number;                // 
    CoordType*** tmp_pair_tilde_r;        // 存储 `PairTildeR.generate()`
    int tmp_cidx;
    int tmp_nidx;

    // Step 2.2. 计算中心原子对应tilde_r[m, n, q] 的 m 索引起始；计算近邻原子对应的 tilde_r[m, n, q] 的 n 索引起始
    /*
    e.g.
    ----
        1. 12 原子的 MoS2:
            1. (4, 80, 4)       -- Mo-Mo
            2. (4, 100, 4)      -- Mo-S
            3. (8, 80, 4)       -- S-Mo
            4. (8, 1000, 4)     -- S-S
     */
    int* cstart_idxs = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<num_center_atomic_numbers; ii++)
        cstart_idxs[ii] = 0;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++) 
            cstart_idxs[ii] += num_center_atoms_lst[jj];
    }
    
    int* nstart_idxs = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        nstart_idxs[ii] = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++) 
            nstart_idxs[ii] += num_neigh_atoms_lst[jj];
    }


    // Step 2.3. 
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        tmp_center_atomic_number = center_atomic_numbers_lst[ii];
        for (int jj=0; jj<num_neigh_atomic_numbers; jj++) {
            tmp_neigh_atomic_number = neigh_atomic_numbers_lst[jj];
            tmp_pair_tilde_r = PairTildeR<CoordType>::generate(
                                    inum,
                                    ilist,
                                    numneigh,
                                    firstneigh,
                                    x,
                                    types,
                                    tmp_center_atomic_number,
                                    tmp_neigh_atomic_number,
                                    num_neigh_atoms_lst[jj],
                                    rcut, 
                                    rcut_smooth);
            
            tmp_cidx = cstart_idxs[ii];         // 0, 4
            for (int kk=0; kk<num_center_atoms_lst[ii]; kk++) {
                tmp_nidx = nstart_idxs[jj];     // 0, 80
                for (int ll=0; ll<num_neigh_atoms_lst[jj]; ll++) {
                    tilde_r[tmp_cidx][tmp_nidx][0] = tmp_pair_tilde_r[kk][ll][0];
                    tilde_r[tmp_cidx][tmp_nidx][1] = tmp_pair_tilde_r[kk][ll][1];
                    tilde_r[tmp_cidx][tmp_nidx][2] = tmp_pair_tilde_r[kk][ll][2];
                    tilde_r[tmp_cidx][tmp_nidx][3] = tmp_pair_tilde_r[kk][ll][3];
                    tmp_nidx++;
                }
                tmp_cidx++;
            }

            // Step . Free memory
            arrayUtils::free3dArray<CoordType>(
                        tmp_pair_tilde_r,
                        num_center_atoms_lst[ii],
                        num_neigh_atoms_lst[jj]);
        }
    }

    // Step . Free memory
    free(num_center_atoms_lst);
    free(cstart_idxs);
    free(nstart_idxs);

    return tilde_r;
}


template <typename CoordType>
void TildeR<CoordType>::generate(
                CoordType*** tilde_r,
                int inum,
                int* ilist,
                int* numneigh,
                int** firstneigh,
                CoordType** x,
                int* types,
                int num_center_atomic_numbers,
                int* center_atomic_numbers_lst,
                int num_neigh_atomic_numbers,
                int* neigh_atomic_numbers_lst,
                int* num_neigh_atoms_lst,
                CoordType rcut,
                CoordType rcut_smooth)
{
    // Step 1. Allocate memory for `tilde_r`
    // Step 1.1. 计算 `num_center_atoms_lst`
    int* num_center_atoms_lst = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<num_center_atomic_numbers; ii++)
        num_center_atoms_lst[ii] = 0;
    int center_atom_idx;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<inum; jj++) {
            center_atom_idx = ilist[jj];
            if (types[center_atom_idx] == center_atomic_numbers_lst[ii]) 
                num_center_atoms_lst[ii]++;
        }
    }


    // Step 2. Populate `tilde_r`
    // Step 2.1. 
    int tmp_center_atomic_number;
    int tmp_neigh_atomic_number;
    CoordType*** tmp_pair_tilde_r;
    int tmp_cidx;
    int tmp_nidx;

    // Step 2.2. 计算中心原子对应tilde_r[m, n, q] 的 m 索引起始；计算近邻原子对应的 tilde_r[m, n, q] 的 n 索引起始
    /*
    e.g.
    ----
        1. 12 原子的 MoS2:
            1. (4, 80, 4)       -- Mo-Mo
            2. (4, 100, 4)      -- Mo-S
            3. (8, 80, 4)       -- S-Mo
            4. (8, 1000, 4)     -- S-S
     */
    int* cstart_idxs = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<num_center_atomic_numbers; ii++)
        cstart_idxs[ii] = 0;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            cstart_idxs[ii] += num_center_atoms_lst[jj];
    }

    int* nstart_idxs = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        nstart_idxs[ii] = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            nstart_idxs[ii] += num_neigh_atoms_lst[jj];
    }

    
    // Step 2.3. 
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        tmp_center_atomic_number = center_atomic_numbers_lst[ii];
        for (int jj=0; jj<num_neigh_atomic_numbers; jj++) {
            tmp_neigh_atomic_number = neigh_atomic_numbers_lst[jj];
            tmp_pair_tilde_r = PairTildeR<CoordType>::generate(
                                    inum,
                                    ilist,
                                    numneigh,
                                    firstneigh,
                                    x,
                                    types,
                                    tmp_center_atomic_number,
                                    tmp_neigh_atomic_number,
                                    num_neigh_atoms_lst[jj],
                                    rcut,
                                    rcut_smooth);


            tmp_cidx = cstart_idxs[ii];
            for (int kk=0; kk<num_center_atoms_lst[ii]; kk++) {
                tmp_nidx = nstart_idxs[jj];
                for (int ll=0; ll<num_neigh_atoms_lst[jj]; ll++) {
                    tilde_r[tmp_cidx][tmp_nidx][0] = tmp_pair_tilde_r[kk][ll][0];
                    tilde_r[tmp_cidx][tmp_nidx][1] = tmp_pair_tilde_r[kk][ll][1];
                    tilde_r[tmp_cidx][tmp_nidx][2] = tmp_pair_tilde_r[kk][ll][2];
                    tilde_r[tmp_cidx][tmp_nidx][3] = tmp_pair_tilde_r[kk][ll][3];
                    tmp_nidx++;
                }
                tmp_cidx++;
            }

            // Step . Free memory
            arrayUtils::free3dArray<CoordType>(
                            tmp_pair_tilde_r,
                            num_center_atoms_lst[ii],
                            num_neigh_atoms_lst[jj]);

        }
    }

    // Step . Free memory
    free(num_center_atoms_lst);
    free(cstart_idxs);
    free(nstart_idxs);
}


/**
 * @brief Combination of `PairTildeR`
 * 
 * @tparam CoordType 
 * @return CoordType**** 
 */
template <typename CoordType>
CoordType**** TildeR<CoordType>::deriv() const {
    // Step 1. Allocate memory for `tidle_r_deriv`, `tilde_r_deriv.shape = (num_center_atoms, num_neigh_atoms, 4, 3)`
    int tot_num_center_atoms = 0;
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        tot_num_center_atoms += this->num_center_atoms_lst[ii];
    
    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++) 
        tot_num_neigh_atoms += this->num_neigh_atoms_lst[ii];

    CoordType**** tilde_r_deriv = arrayUtils::allocate4dArray<CoordType>(
                                                tot_num_center_atoms, 
                                                tot_num_neigh_atoms,
                                                4, 
                                                3,
                                                true);
    
    // Step 2. Populate `tilde_r_deriv`
    // Step 2.1. 
    int tmp_center_atomic_number;               // 存储 `PairTildeR.center_atomic_number`
    int tmp_neigh_atomic_number;                // 存储 `PairTildeR.neigh_atomic_number`
    PairTildeR<CoordType> tmp_pair_tilde_r;     // 存储 `PairTildeR`
    CoordType**** tmp_pair_tilde_r_deriv;       // 存储 `PairTildeR.generate()`
    int tmp_cidx;
    int tmp_nidx;

    // Step 2.2. 计算中心原子对应tilde_r[m, n, q] 的 m 索引起始；计算近邻原子对应的 tilde_r[m, n, q] 的 n 索引起始
    /*
    e.g.
    ----
        1. 12 原子的 MoS2:
            1. (4, 80, 4, 3)       -- Mo-Mo
            2. (4, 100, 4, 3)      -- Mo-S
            3. (8, 80, 4, )       -- S-Mo
            4. (8, 1000, 4, 3)     -- S-S
    */
    int* cstart_idxs = (int*)malloc(sizeof(int) * this->num_center_atomic_numbers);
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++)
        cstart_idxs[ii] = 0;
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            cstart_idxs[ii] += this->num_center_atoms_lst[jj];
    }

    int* nstart_idxs = (int*)malloc(sizeof(int) * this->num_neigh_atomic_numbers);
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++)
        nstart_idxs[ii] = 0;
    for (int ii=0; ii<this->num_neigh_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            nstart_idxs[ii] += this->num_neigh_atoms_lst[jj];
    }

    // Step 2.3. 
    for (int ii=0; ii<this->num_center_atomic_numbers; ii++) {
        tmp_center_atomic_number = this->center_atomic_numbers_lst[ii];
        for (int jj=0; jj<this->num_neigh_atomic_numbers; jj++) {
            tmp_neigh_atomic_number = this->neigh_atomic_numbers_lst[jj];
            tmp_pair_tilde_r = PairTildeR<CoordType>(
                                    this->neighbor_list,
                                    this->center_atomic_numbers_lst[ii],
                                    this->neigh_atomic_numbers_lst[jj],
                                    this->num_center_atoms_lst[ii],
                                    this->num_neigh_atoms_lst[jj],
                                    this->rcut_smooth
                                );
            tmp_pair_tilde_r_deriv = tmp_pair_tilde_r.deriv();

            tmp_cidx = cstart_idxs[ii];
            for (int kk=0; kk<this->num_center_atoms_lst[ii]; kk++) {
                tmp_nidx = nstart_idxs[jj];
                for (int ll=0; ll<this->num_neigh_atoms_lst[jj]; ll++) {
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][0] = tmp_pair_tilde_r_deriv[kk][ll][0][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][1] = tmp_pair_tilde_r_deriv[kk][ll][0][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][2] = tmp_pair_tilde_r_deriv[kk][ll][0][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][0] = tmp_pair_tilde_r_deriv[kk][ll][1][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][1] = tmp_pair_tilde_r_deriv[kk][ll][1][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][2] = tmp_pair_tilde_r_deriv[kk][ll][1][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][0] = tmp_pair_tilde_r_deriv[kk][ll][2][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][1] = tmp_pair_tilde_r_deriv[kk][ll][2][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][2] = tmp_pair_tilde_r_deriv[kk][ll][2][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][0] = tmp_pair_tilde_r_deriv[kk][ll][3][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][1] = tmp_pair_tilde_r_deriv[kk][ll][3][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][2] = tmp_pair_tilde_r_deriv[kk][ll][3][2];

                    tmp_nidx++;
                }
                tmp_cidx++;
            }

            // Step . Free memory
            arrayUtils::free4dArray<CoordType>(tmp_pair_tilde_r_deriv, this->num_center_atoms_lst[ii], this->num_neigh_atoms_lst[jj], 4);
        }
    }

    // Step . Free memory
    free(cstart_idxs);
    free(nstart_idxs);

    return tilde_r_deriv;
}


template <typename CoordType>
CoordType**** TildeR<CoordType>::deriv(
            int inum,
            int* ilist,
            int* numneigh,
            int** firstneigh,
            CoordType** x,
            int* types,
            int num_center_atomic_numbers,
            int* center_atomic_numbers_lst,
            int num_neigh_atomic_numbers,
            int* neigh_atomic_numbers_lst,
            int* num_neigh_atoms_lst,
            CoordType rcut,
            CoordType rcut_smooth)
{
    // Step 1. 计算中心原子的数目 -- `num_center_atoms_lst`
    int center_atom_idx;
    int* num_center_atoms_lst = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<num_center_atomic_numbers; ii++) 
        num_center_atoms_lst[ii] = 0;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<inum; jj++) {
            center_atom_idx = ilist[jj];
            if (types[center_atom_idx] == center_atomic_numbers_lst[ii])
                num_center_atoms_lst[ii]++;
        }
    }

    // Step 2. Allocate memory for `tilde_r_deriv` -- `tilde_r_deriv.shape = (原子总数，近邻原子数，4, 3)`
    int tot_num_center_atoms = 0;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) 
        tot_num_center_atoms += num_center_atoms_lst[ii];

    int tot_num_neigh_atoms = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        tot_num_neigh_atoms += num_neigh_atoms_lst[ii];

    CoordType**** tilde_r_deriv = arrayUtils::allocate4dArray<CoordType>(
                                        tot_num_center_atoms,
                                        tot_num_neigh_atoms,
                                        4,
                                        3,
                                        true);
    

    // Step 3. Populate the `tilde_r_deriv`
    // Step 3.1. 
    int tmp_center_atomic_number;
    int tmp_neigh_atomic_number;
    CoordType**** tmp_pair_tilde_r_deriv;
    int tmp_cidx;
    int tmp_nidx;
    
    // Step 3.2. 计算中心原子对应tilde_r[m, n, q] 的 m 索引起始；计算近邻原子对应的 tilde_r[m, n, q] 的 n 索引起始
    /*
    e.g.
    ----
        1. 12 原子的 MoS2:
            1. (4, 80, 4, 3)       -- Mo-Mo
            2. (4, 100, 4, 3)      -- Mo-S
            3. (8, 80, 4, )       -- S-Mo
            4. (8, 1000, 4, 3)     -- S-S
     */
    int* cstart_idxs = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<num_center_atomic_numbers; ii++)
        cstart_idxs[ii] = 0;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            cstart_idxs[ii] += num_center_atoms_lst[jj];
    }

    int* nstart_idxs = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        nstart_idxs[ii] = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            nstart_idxs[ii] += num_neigh_atoms_lst[jj];
    }


    // Step 3.3. 
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        tmp_center_atomic_number = center_atomic_numbers_lst[ii];
        for (int jj=0; jj<num_neigh_atomic_numbers; jj++) {
            tmp_neigh_atomic_number = neigh_atomic_numbers_lst[jj];
            
            tmp_pair_tilde_r_deriv = PairTildeR<CoordType>::deriv(
                                        inum,
                                        ilist,
                                        numneigh,
                                        firstneigh,
                                        x,
                                        types,
                                        tmp_center_atomic_number,
                                        tmp_neigh_atomic_number,
                                        num_neigh_atoms_lst[jj],
                                        rcut,
                                        rcut_smooth);
            
            tmp_cidx = cstart_idxs[ii];
            for (int kk=0; kk<num_center_atoms_lst[ii]; kk++) {
                tmp_nidx = nstart_idxs[jj];
                for (int ll=0; ll<num_neigh_atoms_lst[jj]; ll++) {
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][0] = tmp_pair_tilde_r_deriv[kk][ll][0][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][1] = tmp_pair_tilde_r_deriv[kk][ll][0][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][2] = tmp_pair_tilde_r_deriv[kk][ll][0][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][0] = tmp_pair_tilde_r_deriv[kk][ll][1][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][1] = tmp_pair_tilde_r_deriv[kk][ll][1][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][2] = tmp_pair_tilde_r_deriv[kk][ll][1][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][0] = tmp_pair_tilde_r_deriv[kk][ll][2][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][1] = tmp_pair_tilde_r_deriv[kk][ll][2][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][2] = tmp_pair_tilde_r_deriv[kk][ll][2][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][0] = tmp_pair_tilde_r_deriv[kk][ll][3][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][1] = tmp_pair_tilde_r_deriv[kk][ll][3][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][2] = tmp_pair_tilde_r_deriv[kk][ll][3][2];

                    tmp_nidx++;
                }
                tmp_cidx++;
            }

            // Step . Free memory
            arrayUtils::free4dArray<CoordType>(tmp_pair_tilde_r_deriv, num_center_atoms_lst[ii], num_neigh_atoms_lst[jj], 4);
        }
    }

    // Step . Free memory
    free(num_center_atoms_lst);
    free(cstart_idxs);
    free(nstart_idxs);

    return tilde_r_deriv;
}


template <typename CoordType>
void TildeR<CoordType>::deriv(
            CoordType**** tilde_r_deriv,
            int inum,
            int* ilist,
            int* numneigh,
            int** firstneigh,
            CoordType** x,
            int* types,
            int num_center_atomic_numbers,
            int* center_atomic_numbers_lst,
            int num_neigh_atomic_numbers,
            int* neigh_atomic_numbers_lst,
            int* num_neigh_atoms_lst,
            CoordType rcut, 
            CoordType rcut_smooth)
{
    // Step 1. 计算中心原子的数目 -- `num_center_atoms_lst`
    int center_atom_idx;
    int* num_center_atoms_lst = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<num_center_atomic_numbers; ii++)
        num_center_atoms_lst[ii] = 0;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<inum; jj++) {
            center_atom_idx = ilist[jj];
            if (types[center_atom_idx] == center_atomic_numbers_lst[ii])
                num_center_atoms_lst[ii]++;
        }
    }

    // Step 2. Populate the `tilde_r_deriv`
    // Step 2.1. 
    int tmp_center_atomic_number;
    int tmp_neigh_atomic_number;
    CoordType**** tmp_pair_tilde_r_deriv;
    int tmp_cidx;
    int tmp_nidx;

    // Step 2.2. 计算中心原子对应tilde_r[m, n, q] 的 m 索引起始；计算近邻原子对应的 tilde_r[m, n, q] 的 n 索引起始
    /*
    e.g.
    ----
        1. 12 原子的 MoS2:
            1. (4, 80, 4, 3)       -- Mo-Mo
            2. (4, 100, 4, 3)      -- Mo-S
            3. (8, 80, 4, )       -- S-Mo
            4. (8, 1000, 4, 3)     -- S-S
     */
    int* cstart_idxs = (int*)malloc(sizeof(int) * num_center_atomic_numbers);
    for (int ii=0; ii<num_center_atomic_numbers; ii++)
        cstart_idxs[ii] = 0;
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            cstart_idxs[ii] += num_center_atoms_lst[jj];
    }

    int* nstart_idxs = (int*)malloc(sizeof(int) * num_neigh_atomic_numbers);
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++)
        nstart_idxs[ii] = 0;
    for (int ii=0; ii<num_neigh_atomic_numbers; ii++) {
        for (int jj=0; jj<ii; jj++)
            nstart_idxs[ii] += num_neigh_atoms_lst[jj];
    }

    // Step 2.3. 
    for (int ii=0; ii<num_center_atomic_numbers; ii++) {
        tmp_center_atomic_number = center_atomic_numbers_lst[ii];
        for (int jj=0; jj<num_neigh_atomic_numbers; jj++) {
            tmp_neigh_atomic_number = neigh_atomic_numbers_lst[jj];

            tmp_pair_tilde_r_deriv = PairTildeR<CoordType>::deriv(
                                        inum,
                                        ilist,
                                        numneigh,
                                        firstneigh,
                                        x,
                                        types,
                                        tmp_center_atomic_number,
                                        tmp_neigh_atomic_number,
                                        num_neigh_atoms_lst[jj],
                                        rcut,
                                        rcut_smooth);
            
            tmp_cidx = cstart_idxs[ii];
            for (int kk=0; kk<num_center_atoms_lst[ii]; kk++) {
                tmp_nidx = nstart_idxs[jj];
                for (int ll=0; ll<num_neigh_atoms_lst[jj]; ll++) {
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][0] = tmp_pair_tilde_r_deriv[kk][ll][0][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][1] = tmp_pair_tilde_r_deriv[kk][ll][0][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][0][2] = tmp_pair_tilde_r_deriv[kk][ll][0][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][0] = tmp_pair_tilde_r_deriv[kk][ll][1][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][1] = tmp_pair_tilde_r_deriv[kk][ll][1][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][1][2] = tmp_pair_tilde_r_deriv[kk][ll][1][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][0] = tmp_pair_tilde_r_deriv[kk][ll][2][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][1] = tmp_pair_tilde_r_deriv[kk][ll][2][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][2][2] = tmp_pair_tilde_r_deriv[kk][ll][2][2];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][0] = tmp_pair_tilde_r_deriv[kk][ll][3][0];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][1] = tmp_pair_tilde_r_deriv[kk][ll][3][1];
                    tilde_r_deriv[tmp_cidx][tmp_nidx][3][2] = tmp_pair_tilde_r_deriv[kk][ll][3][2];          

                    tmp_nidx++;
                }
                tmp_cidx++;
            }

            // Step . Free memory
            arrayUtils::free4dArray<CoordType>(tmp_pair_tilde_r_deriv, num_center_atoms_lst[ii], num_neigh_atoms_lst[jj], 4);
        }
    }


    // Step . Free memory
    free(num_center_atoms_lst);
}


}   // namespace : deepPotSE
}   // namespace : matersdk


#endif