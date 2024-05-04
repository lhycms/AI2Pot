#ifndef CORE_VEC3_OPERATION_H
#define CORE_VEC3_OPERATION_H


#include <stdlib.h>
#include <cmath>

namespace matersdk {
namespace vec3Operation
{


template <typename CoordType>
CoordType dot(CoordType* vec1, CoordType* vec2);

template <typename CoordType>
CoordType* cross(CoordType* vec1, CoordType* vec2);

template <typename CoordType>
CoordType* normalize(CoordType* vec);

template <typename CoordType>
CoordType norm(CoordType* vec);





template <typename CoordType>
CoordType dot(CoordType* vec1, CoordType* vec2) {
    CoordType inner_product = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
    return inner_product;
}


template <typename CoordType>
CoordType** RecursionOuterProduct(int nju, CoordType* vec) {
    CoordType** result = (CoordType**)malloc(sizeof(CoordType*) * std::pow(3, nju-1));
    for (int ii=0; ii<std::pow(3, nju-1); ii++)
        result[ii] = (CoordType*)malloc(sizeof(CoordType) * 3);
    
    if (nju == 1) {
        result[0][0] = vec[0];
        result[0][1] = vec[1];
        result[0][2] = vec[2];
    } else if (nju == 2) {
        for (int ii=0; ii<3; ii++)
            for (int jj=0; jj<3; jj++)
                result[ii][jj] = vec[ii] * vec[jj];
    } else {
        CoordType** result_lower;
        int nju_lower = nju - 1;
        int nrows_lower = std::pow(3, nju_lower - 1);
        int ncols_lower = 3;
        
        result_lower = RecursionOuterProduct(nju_lower, vec);
        for (int ii=0; ii<std::pow(3, nju_lower-1); ii++) {
            for (int jj=0; jj<3; jj++) {
                for (int kk=0; kk<3; kk++) {
                    int tmp_row = ii * ncols_lower + jj;
                    int tmp_col = kk;
                    result[tmp_row][tmp_col] = result_lower[ii][jj] * vec[kk];
                }
            }
        }
    }

    return result;
}


template <typename CoordType>
CoordType* cross(CoordType* vec1, CoordType* vec2) {
    CoordType* vertical_vec = (CoordType*)malloc(sizeof(CoordType) * 3);
    vertical_vec[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    vertical_vec[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    vertical_vec[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
    return vertical_vec;
}


template <typename CoordType>
CoordType* normalize(CoordType* vec) {
    CoordType* unit_vec = (CoordType*)malloc(sizeof(CoordType) * 3);
    CoordType vec_length = std::sqrt( pow(vec[0], 2) + std::pow(vec[1], 2) + std::pow(vec[2], 2) );
    unit_vec[0] = vec[0] / vec_length;
    unit_vec[1] = vec[1] / vec_length;
    unit_vec[2] = vec[2] / vec_length;

    return unit_vec;
}


template <typename CoordType>
CoordType norm(CoordType* vec) {
    return std::sqrt(
        std::pow(vec[0], 2) + std::pow(vec[1], 2) + std::pow(vec[2], 2) 
    );
}


    
}   // namespace: vecOperation
}   // namespace: matersdk


#endif