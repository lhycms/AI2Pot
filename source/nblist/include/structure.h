#ifndef MATERSDK_STRUCTURE_H
#define MATERSDK_STRUCTURE_H

#include <stdio.h>
#include <stdlib.h>
#include "../../core/include/vec3Operation.h"


namespace matersdk {

// Forward declaration of class B for `friend class`
template <typename CoordType>
class Supercell;

template <typename CoordType>
class BasicStructureInfo;



template <typename CoordType>
class Structure {
public:
    Structure();

    Structure(int num_atoms);
    
    Structure(int num_atoms, CoordType **basis_vectors, int *atomic_numbers, CoordType **coords, bool is_cart_coords=true);
    
    Structure(int num_atoms, CoordType basis_vectors[3][3], int atomic_number[], CoordType coords[][3], bool is_cart_coords=true);

    Structure(const Structure &rhs);

    Structure& operator=(const Structure &rhs);
    
    ~Structure();

    void calc_cart_coords(CoordType **frac_coords);

    void calc_cart_coords(CoordType frac_coords[][3]);

    // Note: `0~this->num_atoms` are owned atoms; others are ghost atoms.
    void make_supercell(const int *scaling_matix);   // Note: You can use `int[3]` to init it.

    // void make_supercell(const int scaling_matrix[3]);

    void show() const;

    const int get_num_atoms() const;

    const CoordType** get_basis_vectors() const; // Returns a pointer to a pointer to a constant double value.

    const int* get_atomic_numbers() const;      // Returns a pointer to a constant double value.

    const CoordType** get_cart_coords() const;  // Returns a pointer to a pointer to a constant double value.

    CoordType* get_projected_lengths() const; //

    CoordType* get_interplanar_distances() const;

    const CoordType* get_pseudo_origin() const;

    CoordType** get_vertexes() const;

    CoordType** get_limit_xyz() const;      // [3][2]

    friend class Supercell<CoordType>;

    friend class BasicStructureInfo<CoordType>;

private:
    int num_atoms = 0;  // Note: 初始化为0，防止 `matersdk::Structure<double> structure;` 后，拷贝赋值函数无法得到正确的 `this->num_atoms`
    CoordType **basis_vectors;
    CoordType *pseudo_orgin;            // 仅当 `make_supercell` 后改变
    int *atomic_numbers;
    CoordType **cart_coords;
}; // class: Structure



} // namespace: matersdk








// Definition of Structure member function
namespace matersdk {


/**
 * @brief Construct a new Structure< Coord Type>:: Structure object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
Structure<CoordType>::Structure() {
    // this->num_atoms = 0;
}


/**
 * @brief Construct a new Structure< Coord Type>:: Structure object
 * 
 * @tparam CoordType 
 * @param num_atoms 
 */
template <typename CoordType>
Structure<CoordType>::Structure(int num_atoms) {
    this->num_atoms = num_atoms;

    if (this->num_atoms != 0) {
        // Step 1. Allocate memory for `this->basis_vectors`
        this->basis_vectors = (CoordType**)malloc(sizeof(CoordType*) * 3);
        for (int ii=0; ii<3; ii++) {
            this->basis_vectors[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }

        // Step 2. Allocate memory for `this->pseudo_origin`
        this->pseudo_orgin = (CoordType*)malloc(sizeof(CoordType) * 3);

        // Step 3. Allocate memory for `this->atomic_numbers`
        this->atomic_numbers = (int*)malloc(sizeof(int) * num_atoms);

        // Step 4. Allocate memory for `this->cart_coords`
        this->cart_coords = (CoordType**)malloc(sizeof(CoordType*) * num_atoms);
        for (int ii=0; ii<this->num_atoms; ii++) {
            this->cart_coords[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
    }
}


/**
 * @brief Construct a new Structure< Coord Type>:: Structure object
 * 
 * @tparam CoordType 
 * @param num_atoms 
 * @param basis_vectors 
 * @param atomic_numbers 
 * @param cart_coords 
 */
template <typename CoordType>
Structure<CoordType>::Structure(
        int num_atoms,
        CoordType **basis_vectors, int *atomic_numbers, CoordType **coords,
        bool is_cart_coords)
{
    this->num_atoms = num_atoms;

    if (this->num_atoms != 0) {
        // Step 1. Allocate memory for `this->basis_vectors` and assign
        this->basis_vectors = (CoordType**)malloc(sizeof(CoordType*) * 3);
        for (int ii=0; ii<3; ii++) {
            this->basis_vectors[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
        // Step 1.1. Assign `this->basis_vectors`
        for (int ii=0; ii<3; ii++) {
            for (int jj=0; jj<3; jj++) {
                this->basis_vectors[ii][jj] = basis_vectors[ii][jj];
            }
        }

        // Step 2. Allocate memory for `this->pseudo_origin` and assigin
        this->pseudo_orgin = (CoordType*)malloc(sizeof(CoordType) * 3);
        // Step 2.1. Assign `this->pseudo_origin`.
        this->pseudo_orgin[0] = 0;
        this->pseudo_orgin[1] = 0;
        this->pseudo_orgin[2] = 0;


        // Step 3. Allocate memory for `this->atomic_numbers` and assign
        this->atomic_numbers = (int*)malloc(sizeof(int) * num_atoms);
        // Step 4.1. Assign `this->atomic_numbers`
        for (int ii=0; ii<num_atoms; ii++) {
            this->atomic_numbers[ii] = atomic_numbers[ii];
        }

        // Step 4. Allocate memory for `this->cart_coords` and assign
        this->cart_coords = (CoordType**)malloc(sizeof(CoordType*) * num_atoms);
        for (int ii=0; ii<num_atoms ; ii++) {
            this->cart_coords[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
        // Step 4.1. Assign
        if (is_cart_coords) {
            for (int ii=0; ii<num_atoms; ii++) {
                for (int jj=0; jj<3; jj++) {
                    this->cart_coords[ii][jj] = coords[ii][jj];
                }
            }
        } else {
            this->calc_cart_coords(coords);
        }
    }
}



/**
 * @brief Construct a new Structure< Coord Type>:: Structure object
 * 
 * @tparam CoordType 
 * @param num_atoms 
 * @param basis_vectors 
 * @param atomic_numbers 
 * @param coords 
 * @param is_cart_coords 
 */
template <typename CoordType>
Structure<CoordType>::Structure(int num_atoms,
        CoordType basis_vectors[3][3], int atomic_numbers[], CoordType coords[][3],
        bool is_cart_coords)
{
    this->num_atoms = num_atoms;

    if (this->num_atoms != 0) {
        // Step 1. Allocate memory for `this->basis_vectors` and assign
        this->basis_vectors = (CoordType**)malloc(sizeof(CoordType*) * 3);
        for (int ii=0; ii<3; ii++) {
            this->basis_vectors[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
        for (int ii=0; ii<3; ii++) {
            for (int jj=0; jj<3; jj++) {
                this->basis_vectors[ii][jj] = basis_vectors[ii][jj];
            }
        }

        // Step 2. Allocate memory for `this->pseudo_origin`
        this->pseudo_orgin = (CoordType*)malloc(sizeof(CoordType) * 3);
        this->pseudo_orgin[0] = 0;
        this->pseudo_orgin[1] = 0;
        this->pseudo_orgin[2] = 0;

        // Step 3. Allocate memory for `this->atomic_numbers`
        this->atomic_numbers = (int*)malloc(sizeof(int) * this->num_atoms);
        for (int ii=0; ii<this->num_atoms; ii++) {
            this->atomic_numbers[ii] = atomic_numbers[ii];
        }

        // Step 4. Allocate memory for `this->cart_coords`
        this->cart_coords = (CoordType**)malloc(sizeof(CoordType*) * this->num_atoms);
        for (int ii=0; ii<this->num_atoms; ii++) {
            this->cart_coords[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
        if (is_cart_coords) {   // 如果 `coords` 是笛卡尔坐标
            for (int ii=0; ii<this->num_atoms; ii++) {
                for (int jj=0; jj<3; jj++) {
                    this->cart_coords[ii][jj] = coords[ii][jj];
                }
            }
        } else {    // 若如果不是笛卡尔坐标
            this->calc_cart_coords(coords);
        }
    }
}


/**
 * @brief Construct a new Structure< Coord Type>:: Structure object
 * 
 * @tparam CoordType 
 * @param rhs 
 */
template <typename CoordType>
Structure<CoordType>::Structure(const Structure &rhs)
{  
    // Step 1. Allocate and Assign
    this->num_atoms = rhs.num_atoms;

    if (rhs.num_atoms != 0) {
        // Step 1.1. Allocate memory for `this->basis_vectors` and assign
        this->basis_vectors = (CoordType**)malloc(sizeof(CoordType*) * 3);
        for (int ii=0; ii<3; ii++) {
            this->basis_vectors[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
        for (int ii=0; ii<3; ii++) {
            for (int jj=0; jj<3; jj++) {
                this->basis_vectors[ii][jj] = rhs.basis_vectors[ii][jj];
            }
        }

        // Step 1.2. Allocate memory for `this->pseudo_origin` and assign it
        this->pseudo_orgin = (CoordType*)malloc(sizeof(CoordType) * 3);
        this->pseudo_orgin[0] = rhs.pseudo_orgin[0];
        this->pseudo_orgin[1] = rhs.pseudo_orgin[1];        
        this->pseudo_orgin[2] = rhs.pseudo_orgin[2];

        // Step 1.3. Allocate memory for `this->atomic_numbers` and assign
        this->atomic_numbers = (int*)malloc(sizeof(int) * this->num_atoms);
        for (int ii=0; ii<this->num_atoms; ii++) {
            this->atomic_numbers[ii] = rhs.atomic_numbers[ii];
        }

        // Step 1.4. Allocate memory for `this->cart_coords` and assign
        this->cart_coords = (CoordType**)malloc(sizeof(CoordType*) * this->num_atoms);
        for (int ii=0; ii<this->num_atoms; ii++) {
            this->cart_coords[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
        for (int ii=0; ii<this->num_atoms; ii++) {
            for (int jj=0; jj<3; jj++) {
                this->cart_coords[ii][jj] = rhs.cart_coords[ii][jj];
            }
        }
    }

}


/**
 * @brief Copy assignment operator
 * 
 * @tparam CoordType 
 * @param rhs 
 * @return Structure<CoordType>& 
 */
template <typename CoordType>
Structure<CoordType>& Structure<CoordType>::operator=(const Structure &rhs) {
    // Step 1. Free memory 
    if (this->num_atoms != 0) {
        // Step 1.1. `this->basis_vectors`
        for (int ii=0; ii<3; ii++) {
            free(this->basis_vectors[ii]);
        }
        free(this->basis_vectors);

        // Step 1.2. 
        free(this->pseudo_orgin);

        // Step 1.3. `this->atomic_numbers`
        free(this->atomic_numbers);

        // Step 1.4. `this->cart_coords`
        for (int ii=0; ii<this->num_atoms; ii++) {
            free(this->cart_coords[ii]);
        }
        free(this->cart_coords);

        // Step 1.5. `this->num_atoms = 0`
        this->num_atoms = 0;
    }

    // Step 2. Reallocate and reassign
    this->num_atoms = rhs.num_atoms;

    if (rhs.num_atoms != 0) {
        // Step 2.1. Allocate memory for `this->basis_vectors` and assign it
        this->basis_vectors = (CoordType**)malloc(sizeof(CoordType*) * 3);
        for (int ii=0; ii<3; ii++) {
            this->basis_vectors[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
        for (int ii=0; ii<3; ii++) {
            this->basis_vectors[ii][0] = rhs.basis_vectors[ii][0];
            this->basis_vectors[ii][1] = rhs.basis_vectors[ii][1];
            this->basis_vectors[ii][2] = rhs.basis_vectors[ii][2];
        }

        // Step 2.2. Allocate memory for `this->pseudo_origin`
        this->pseudo_orgin = (CoordType*)malloc(sizeof(CoordType) * 3);
        this->pseudo_orgin[0] = rhs.pseudo_orgin[0];
        this->pseudo_orgin[1] = rhs.pseudo_orgin[1];
        this->pseudo_orgin[2] = rhs.pseudo_orgin[2];
        
        // Step 2.3. Allocate memory for `this->atomic_numbers` and assign it
        this->atomic_numbers = (int*)malloc(sizeof(int) * this->num_atoms);
        for (int ii=0; ii<this->num_atoms; ii++) {
            this->atomic_numbers[ii] = rhs.atomic_numbers[ii];
        }

        // Step 2.4. Allocate memory for `this->cart_coords` and assign it 
        this->cart_coords = (CoordType**)malloc(sizeof(CoordType*) * this->num_atoms);
        for (int ii=0; ii<this->num_atoms; ii++) {
            this->cart_coords[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
        }
        for (int ii=0; ii<this->num_atoms; ii++) {
            this->cart_coords[ii][0] = rhs.cart_coords[ii][0];
            this->cart_coords[ii][1] = rhs.cart_coords[ii][1];
            this->cart_coords[ii][2] = rhs.cart_coords[ii][2];
        }
    }

    return *this;
}


/**
 * @brief Destroy the Structure< Coord Type>:: Structure object
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
Structure<CoordType>::~Structure() {
    if (this->num_atoms != 0) {
        // Step 1. Deallocate `this->basis_vectors`
        for (int ii=0; ii<3; ii++) {
            free(this->basis_vectors[ii]);
        }
        free(this->basis_vectors);

        // Step 2. Deallocate `this->pseudo_origin`
        free(this->pseudo_orgin);

        // Step 3. Deallocate `this->atomic_numbers`
        free(this->atomic_numbers);

        // Step 4. Deallocate `this->cart_coords`
        for (int ii=0; ii<this->num_atoms; ii++) {
            free(this->cart_coords[ii]);
        }
        free(this->cart_coords);

        // Step 5. `this->num_atoms = 0`
        this->num_atoms = 0;
    }
}


/**
 * @brief Convert the `fractional coordinates` to `cartesian coordinates`
 * 
 * @tparam CoordType 
 * @param frac_coords 
 */
template <typename CoordType>
void Structure<CoordType>::calc_cart_coords(CoordType **frac_coords) {
    for (int ii=0; ii<this->num_atoms; ii++) {
        this->cart_coords[ii][0] = (
            frac_coords[ii][0] * this->basis_vectors[0][0] + 
            frac_coords[ii][1] * this->basis_vectors[1][0] + 
            frac_coords[ii][2] * this->basis_vectors[2][0]
        );
        this->cart_coords[ii][1] = (
            frac_coords[ii][0] * this->basis_vectors[0][1] + 
            frac_coords[ii][1] * this->basis_vectors[1][1] + 
            frac_coords[ii][2] * this->basis_vectors[2][1]
        );
        this->cart_coords[ii][2] = (
            frac_coords[ii][0] * this->basis_vectors[0][2] + 
            frac_coords[ii][1] * this->basis_vectors[1][2] +
            frac_coords[ii][2] * this->basis_vectors[2][2]
        );
    }
}


/**
 * @brief Convert the `fractional coordinates` to `cartesian coordinates`
 * 
 * @tparam CoordType 
 * @param frac_coords 
 */
template <typename CoordType>
void Structure<CoordType>::calc_cart_coords(CoordType frac_coords[][3]) {
    for (int ii=0; ii<this->num_atoms; ii++) {
        this->cart_coords[ii][0] = (
            frac_coords[ii][0] * this->basis_vectors[0][0] + 
            frac_coords[ii][1] * this->basis_vectors[1][0] + 
            frac_coords[ii][2] * this->basis_vectors[2][0]
        );
        this->cart_coords[ii][1] = (
            frac_coords[ii][0] * this->basis_vectors[0][1] +
            frac_coords[ii][1] * this->basis_vectors[1][1] + 
            frac_coords[ii][2] * this->basis_vectors[2][1]
        );
        this->cart_coords[ii][2] = (
            frac_coords[ii][0] * this->basis_vectors[0][2] + 
            frac_coords[ii][1] * this->basis_vectors[1][2] +
            frac_coords[ii][2] * this->basis_vectors[2][2]
        );
    }
}


/**
 * @brief make supercell
 * 
 * @tparam CoordType 
 * @param scaling_matrix 
 * 
 */
template <typename CoordType>
void Structure<CoordType>::make_supercell(const int *scaling_matrix) {
    /*
        1. 奇数: 
            ( -\frac{num-1}{2}, \frac{num-1}{2})
        2. 偶数: 
            ( -(\frac{num}{2}+1), \frac{num}{2} )
    */
    int range[3][2];
    for (int ii=0; ii<3; ii++) {
        if (scaling_matrix[ii] % 2 == 0){   // 偶数
            range[ii][0] = -scaling_matrix[ii]/2 + 1;
            range[ii][1] = scaling_matrix[ii]/2;
        } else {    // 奇数
            range[ii][0] = -(scaling_matrix[ii]-1)/2;
            range[ii][1] = (scaling_matrix[ii]-1)/2;
        }
    }

    // Step 2. Allocate memory for `primitive_cell` and Reallocate memory for `supercell (this)`
    // Step 2.1. 利用 `num_atoms_prim`, `atomic_numbers_prim`, `cart_coords_prim` 存储原胞的信息
    int num_atoms_prim = this->num_atoms;
    // `atomic_numbers`
    int *atomic_numbers_prim = (int*)malloc(sizeof(int) * num_atoms_prim);
    for (int ii=0; ii<num_atoms_prim; ii++) {
        atomic_numbers_prim[ii] = this->atomic_numbers[ii];
    }
    // `cart_coords`
    CoordType **cart_coords_prim = (CoordType**)malloc(sizeof(CoordType*) * num_atoms_prim);
    for (int ii=0; ii<num_atoms_prim; ii++) {
        cart_coords_prim[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
    }
    for (int ii=0; ii<num_atoms_prim; ii++) {
        for (int jj=0; jj<3; jj++) {
            cart_coords_prim[ii][jj] = this->cart_coords[ii][jj];
        }
    }

    // Step 2.2. Reallocate memory for `this`
    this->num_atoms = num_atoms_prim * scaling_matrix[0] * scaling_matrix[1] * scaling_matrix[2];
    
    // `atomic_numbers`
    free(this->atomic_numbers);
    this->atomic_numbers = (int*)malloc(sizeof(int) * this->num_atoms);

    // `cart_coords`
    for (int ii=0; ii<num_atoms_prim; ii++) {
        free(this->cart_coords[ii]);
    }
    free(this->cart_coords);
    this->cart_coords = (CoordType**)malloc(sizeof(CoordType*) * this->num_atoms);
    for (int ii=0; ii<this->num_atoms; ii++) {
        this->cart_coords[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
    }

    // Step 3. Reassign `basis_vectors`, `atomic_numbers`, `cart_coords`
    // Step 3.1. Calculate `atomic_numbers` and assign it to `this->atomic_numbers`
    for (int num_copies=0; num_copies<scaling_matrix[0]*scaling_matrix[1]*scaling_matrix[2]; num_copies++) {
        for (int atom_idx=0; atom_idx<num_atoms_prim; atom_idx++) {
            this->atomic_numbers[num_copies*num_atoms_prim + atom_idx] = atomic_numbers_prim[atom_idx];
        }
    }


    // Step 3.2. Calculate `cart_coords` and assign it to `this->cart_coords`
    int atom_idx = 0;
    for (int ii=range[0][0]; ii <= range[0][1]; ii++) {
        for (int jj=range[1][0]; jj <= range[1][1]; jj++) {
            for (int kk=range[2][0]; kk <= range[2][1]; kk++) {
                for (int prim_atom_idx=0; prim_atom_idx<num_atoms_prim; prim_atom_idx++) {
                    //std::cout << ii << ", " << jj << ", " << kk << std::endl;
                    this->cart_coords[atom_idx][0] = (
                        cart_coords_prim[prim_atom_idx][0] + 
                        this->basis_vectors[0][0] * ii + 
                        this->basis_vectors[1][0] * jj + 
                        this->basis_vectors[2][0] * kk
                    );
                    this->cart_coords[atom_idx][1] = (
                        cart_coords_prim[prim_atom_idx][1] + 
                        this->basis_vectors[0][1] * ii + 
                        this->basis_vectors[1][1] * jj +
                        this->basis_vectors[2][1] * kk
                    );
                    this->cart_coords[atom_idx][2] = (
                        cart_coords_prim[prim_atom_idx][2] + 
                        this->basis_vectors[0][2] * ii + 
                        this->basis_vectors[1][2] * jj + 
                        this->basis_vectors[2][2] * kk
                    );

                    atom_idx++;
                }
            }
        }
    }

    // Step 3.3. Calculate `pseudo_origin` and assign it to `this->pseudo_origin`
    for (int ii=0; ii<3; ii++) {
        this->pseudo_orgin[ii] = (
            range[0][0] * this->basis_vectors[0][ii] + 
            range[1][0] * this->basis_vectors[1][ii] + 
            range[2][0] * this->basis_vectors[2][ii]
        );
    }

    // Step 3.4. Calculate `basis_vectors` and assign it to `this->basis_vectors`
    for (int ii=0; ii<3; ii++) { // 三个基矢方向
        // Step 3.2.1. 缩放：扩大 scaling_factor 倍
        this->basis_vectors[ii][0] *= scaling_matrix[ii];
        this->basis_vectors[ii][1] *= scaling_matrix[ii];
        this->basis_vectors[ii][2] *= scaling_matrix[ii];
    }


    // Step 4. Free memory
    free(atomic_numbers_prim);
    for (int ii=0; ii<num_atoms_prim; ii++)
        free(cart_coords_prim[ii]);
    free(cart_coords_prim);
}


/**
 * @brief Output the information of `Sturcture`
 * 
 * @tparam CoordType 
 */
template <typename CoordType>
void Structure<CoordType>::show() const {
    if (this->num_atoms != 0) {
        printf("Lattice (Origin = [0, 0, 0])\n");
        printf("-------------------------------------------------------\n");
        printf(" %-15.6f %-15.6f %-15.6f\n", this->basis_vectors[0][0], this->basis_vectors[0][1], this->basis_vectors[0][2]);
        printf(" %-15.6f %-15.6f %-15.6f\n", this->basis_vectors[1][0], this->basis_vectors[1][1], this->basis_vectors[1][2]);
        printf(" %-15.6f %-15.6f %-15.6f\n", this->basis_vectors[2][0], this->basis_vectors[2][1], this->basis_vectors[2][2]);
        printf("\nPseudo Origin\n");
        printf("-------------------------------------------------------\n");
        printf(" %-15.6f %-15.6f %-15.6f\n", this->pseudo_orgin[0], this->pseudo_orgin[1], this->pseudo_orgin[2]);
        printf("\nSite (Cartesian Coordinate)\n");
        printf("-------------------------------------------------------\n");
        for (int ii=0; ii<this->num_atoms; ii++)
            printf(" %-4d %-4d  %-15.6f %-15.6f %-15.6f\n", ii, this->atomic_numbers[ii], this->cart_coords[ii][0], this->cart_coords[ii][1], this->cart_coords[ii][2]);
    } else {
        printf("This is a NULL matersdk::Sturcture\n");
    }
}


template <typename CoordType>
const int Structure<CoordType>::get_num_atoms() const {
    return (const int)this->num_atoms;
}


/**
 * @brief Convert `this->basis_vectors` to const value, and return it
 * 
 * @tparam CoordType 
 * @return const CoordType** 
 * 
 * @note You can't return `this->vectors` directly, because 
 *          - error: invalid implicit conversion from `double**` to `const double**`
 */
template <typename CoordType>
const CoordType** Structure<CoordType>::get_basis_vectors() const {
    if (this->num_atoms != 0)
        return (const CoordType**)this->basis_vectors;  // Note: You'd better not use implicit conversion
    else
        return nullptr;
}


template <typename CoordType>
const int* Structure<CoordType>::get_atomic_numbers() const {
    if (this->num_atoms != 0)
        return this->atomic_numbers;
    else
        return nullptr;
}


template <typename CoordType>
const CoordType** Structure<CoordType>::get_cart_coords() const {
    if (this->num_atoms != 0)
        return (const CoordType**)this->cart_coords;
    else
        return nullptr;
}



/**
 * @brief Get the length of sum basis vectors projected on x, y, z axis.
 * 
 * @tparam CoordType 
 * @return const CoordType* 
 */
template <typename CoordType>
CoordType* Structure<CoordType>::get_projected_lengths() const {
    if (this->num_atoms == 0) 
        return nullptr;

    CoordType* projected_lengths = (CoordType*)malloc(sizeof(CoordType) * 3);
    projected_lengths[0] = (
            std::abs(this->basis_vectors[0][0]) + 
            std::abs(this->basis_vectors[1][0]) + 
            std::abs(this->basis_vectors[2][0])
    );
    projected_lengths[1] = (
            std::abs(this->basis_vectors[0][1]) + 
            std::abs(this->basis_vectors[1][1]) + 
            std::abs(this->basis_vectors[2][1])
    );
    projected_lengths[2] = (
            std::abs(this->basis_vectors[0][2]) + 
            std::abs(this->basis_vectors[1][2]) + 
            std::abs(this->basis_vectors[2][2])
    );

    return projected_lengths;
}


/**
 * @brief Get inter-planar distances for structure.
 * 
 * @tparam CoordType 
 * @return const CoordType* 
 */
template <typename CoordType>
CoordType* Structure<CoordType>::get_interplanar_distances() const {
    if (this->num_atoms == 0) 
        return nullptr;

    CoordType* vec_vertical_yz = vec3Operation::cross(this->basis_vectors[1], this->basis_vectors[2]);
    CoordType* vec_vertical_xz = vec3Operation::cross(this->basis_vectors[0], this->basis_vectors[2]);
    CoordType* vec_vertical_xy = vec3Operation::cross(this->basis_vectors[0], this->basis_vectors[1]);
    CoordType* unit_vec_vertical_yz = vec3Operation::normalize(vec_vertical_yz);
    CoordType* unit_vec_vertical_xz = vec3Operation::normalize(vec_vertical_xz);
    CoordType* unit_vec_vertical_xy = vec3Operation::normalize(vec_vertical_xy);

    CoordType* interplanar_distances = (CoordType*)malloc(sizeof(CoordType) * 3);
    interplanar_distances[0] = std::abs( vec3Operation::dot(this->basis_vectors[0], unit_vec_vertical_yz) );
    interplanar_distances[1] = std::abs( vec3Operation::dot(this->basis_vectors[1], unit_vec_vertical_xz) );
    interplanar_distances[2] = std::abs( vec3Operation::dot(this->basis_vectors[2], unit_vec_vertical_xy) );
    
    free(vec_vertical_yz);
    free(vec_vertical_xz);
    free(vec_vertical_xy);
    free(unit_vec_vertical_yz);
    free(unit_vec_vertical_xz);
    free(unit_vec_vertical_xy);

    return interplanar_distances;
}


template <typename CoordType>
const CoordType* Structure<CoordType>::get_pseudo_origin() const {
    return (const CoordType*)this->pseudo_orgin;
}


/**
 * @brief Get 8 vertexes for orthgonal or triclinic system.
 * 
 * @tparam CoordType 
 * @return CoordType** 
 */
template <typename CoordType>
CoordType** Structure<CoordType>::get_vertexes() const {
    CoordType** vertexes = (CoordType**)malloc(sizeof(CoordType*) * 8);
    for (int ii=0; ii<8; ii++) {
        vertexes[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
    }

    int vertex_idx = 0;
    for (int ii=0; ii<=1; ii++) {
        for (int jj=0; jj<=1; jj++) {
            for (int kk=0; kk<=1; kk++) {
                CoordType shift[3] = {0, 0, 0};
                shift[0] = (
                    ii * this->basis_vectors[0][0] + 
                    jj * this->basis_vectors[1][0] + 
                    kk * this->basis_vectors[2][0]
                );
                shift[1] = (
                    ii * this->basis_vectors[0][1] + 
                    jj * this->basis_vectors[1][1] +
                    kk * this->basis_vectors[2][1]
                );
                shift[2] = (
                    ii * this->basis_vectors[0][2] + 
                    jj * this->basis_vectors[1][2] + 
                    kk * this->basis_vectors[2][2]
                );

                vertexes[vertex_idx][0] = this->pseudo_orgin[0] + shift[0];
                vertexes[vertex_idx][1] = this->pseudo_orgin[1] + shift[1];
                vertexes[vertex_idx][2] = this->pseudo_orgin[2] + shift[2];

                vertex_idx++;
            }
        }
    }

    return vertexes;
}


template <typename CoordType>
CoordType** Structure<CoordType>::get_limit_xyz() const {
    CoordType** limit_xyz = (CoordType**)malloc(sizeof(CoordType*) * 3);
    for (int ii=0; ii<3; ii++) {
        limit_xyz[ii] = (CoordType*)malloc(sizeof(CoordType) * 2);
    }

    CoordType** vertexes = this->get_vertexes();

    // Step 1. 用 `vertexes[0]` 初始化 `limit_xyz` 
    limit_xyz[0][0] = vertexes[0][0];
    limit_xyz[0][1] = vertexes[0][0];
    limit_xyz[1][0] = vertexes[0][1];
    limit_xyz[1][1] = vertexes[0][1];
    limit_xyz[2][0] = vertexes[0][2];
    limit_xyz[2][1] = vertexes[0][2];

    // Step 2. Populate `limit_xyz`
    for (int ii=1; ii<8; ii++) {
        // Step 2.1. limit_x
        if (vertexes[ii][0] < limit_xyz[0][0])
            limit_xyz[0][0] = vertexes[ii][0];
        if (vertexes[ii][0] > limit_xyz[0][1])
            limit_xyz[0][1] = vertexes[ii][0];

        // Step 2.2. limit_y
        if (vertexes[ii][1] < limit_xyz[1][0])
            limit_xyz[1][0] = vertexes[ii][1];
        if (vertexes[ii][1] > limit_xyz[1][1])
            limit_xyz[1][1] = vertexes[ii][1];

        // Step 2.3. limit_z
        if (vertexes[ii][2] < limit_xyz[2][0]) 
            limit_xyz[2][0] = vertexes[ii][2];
        if (vertexes[ii][2] > limit_xyz[2][1]) 
            limit_xyz[2][1] = vertexes[ii][2];
        
    }

    for (int ii=0; ii<8; ii++) {
        free(vertexes[ii]);
    }
    free(vertexes);

    return limit_xyz;
}


}   // namespace: matersdk


#endif