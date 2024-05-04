#ifndef MATERSDK_ARRAY_UTILS_H
#define MATERSDK_ARRAY_UTILS_H


#include <stdlib.h>
#include <stdio.h>


namespace matersdk {
namespace arrayUtils {


/**
 * @brief 为三维数组分配内存，并返回其指针
 * 
 * @tparam CoordType 
 * @param element_size_0
 * @param element_size_1
 * @param element_size_2
 * [m, n, q], if return `TildeR`, then
 *                  1. `element_size_0` = num_center_atoms
 *                  2. `element_size_1` = num_neigh_atoms
 *                  3. `element_size_2` = 3
 * @param init_mark 是否将所有元素初始化为 0
 * 
 * @return CoordType*** 
 */
template <typename CoordType>
CoordType*** allocate3dArray(int element_size_0, int element_size_1, int element_size_2, bool init_mark=false) {
    CoordType*** pointer_3dArray = (CoordType***)malloc(sizeof(CoordType**) * element_size_0);
    for (int ii=0; ii<element_size_0; ii++) {
        pointer_3dArray[ii] = (CoordType**)malloc(sizeof(CoordType*) * element_size_1);
        for (int jj=0; jj<element_size_1; jj++) {
            pointer_3dArray[ii][jj] = (CoordType*)malloc(sizeof(CoordType) * element_size_2);
        }
    }

    if (init_mark) {
        for (int ii=0; ii<element_size_0; ii++) {
            for (int jj=0; jj<element_size_1; jj++) {
                for (int kk=0; kk<element_size_2; kk++)
                    pointer_3dArray[ii][jj][kk] = 0;
            }
        }
    }

    return pointer_3dArray;
}


/**
 * @brief 为三维数组分配内存，并返回其指针
 * 
 * @tparam CoordType 
 * @param pointer_3dArray 需要释放的三维数组指针
 * @param element_size_0 
 * @param element_size_1 
 */
template <typename CoordType>
void free3dArray(CoordType*** pointer_3dArray, int element_size_0, int element_size_1) {
    for (int ii=0; ii<element_size_0; ii++) {
        for (int jj=0; jj<element_size_1; jj++)
            free(pointer_3dArray[ii][jj]);
        free(pointer_3dArray[ii]);
    }
    free(pointer_3dArray);
}


template <typename CoordType>
void show3dArray(CoordType*** pointer_3dArray, int element_size_0, int element_size_1, int element_size_2) {
    for (int ii=0; ii<element_size_0; ii++) {
        for (int jj=0; jj<element_size_1; jj++) {
            for (int kk=0; kk<element_size_2; kk++) {
                printf("[%d, %d, %d]: %f\n", ii, jj, kk, pointer_3dArray[ii][jj][kk]);
            }
        }
    }
}




template <typename CoordType>
CoordType**** allocate4dArray(int element_size_0, int element_size_1, int element_size_2, int element_size_3, bool init_mark) {
    CoordType**** pointer_4dArray = (CoordType****)malloc(sizeof(CoordType***) * element_size_0);
    for (int ii=0; ii<element_size_0; ii++) {
        pointer_4dArray[ii] = (CoordType***)malloc(sizeof(CoordType**) * element_size_1);
        for (int jj=0; jj<element_size_1; jj++) {
            pointer_4dArray[ii][jj] = (CoordType**)malloc(sizeof(CoordType*) * element_size_2);
            for (int kk=0; kk<element_size_2; kk++) {
                pointer_4dArray[ii][jj][kk] = (CoordType*)malloc(sizeof(CoordType) * element_size_3);
            }
        }
    }

    if (init_mark) {
        for (int ii=0; ii<element_size_0; ii++) {
            for (int jj=0; jj<element_size_1; jj++) {
                for (int kk=0; kk<element_size_2; kk++) {
                    for (int ll=0; ll<element_size_3; ll++)
                        pointer_4dArray[ii][jj][kk][ll] = 0;
                }
            }
        }
    }


    return pointer_4dArray;
}


template <typename CoordType>
void free4dArray(CoordType**** pointer_4dArray, int element_size_0, int element_size_1, int element_size_2) {
    for (int ii=0; ii<element_size_0; ii++) {
        for (int jj=0; jj<element_size_1; jj++) {
            for (int kk=0; kk<element_size_2; kk++)
                free(pointer_4dArray[ii][jj][kk]);
            free(pointer_4dArray[ii][jj]);
        }
        free(pointer_4dArray[ii]);
    }
    free(pointer_4dArray);
}


template <typename CoordType>
void show4dArray(CoordType**** pointer_4dArray, int element_size_0, int element_size_1, int element_size_2, int element_size_3) {
    for (int ii=0; ii<element_size_0; ii++) {
        for (int jj=0; jj<element_size_1; jj++) {
            for (int kk=0; kk<element_size_2; kk++) {
                for (int ll=0; ll<element_size_3; ll++)
                    printf("[%d, %d, %d, %d] : %f\n", ii, jj, kk, ll, pointer_4dArray[ii][jj][kk][ll]);
            }
        }
    }
}



template <typename CoordType>
CoordType***** allocate5dArray(
                        int element_size_0,
                        int element_size_1,
                        int element_size_2,
                        int element_size_3,
                        int element_size_4,
                        bool init_mark=false)
{
    CoordType***** pointer_5dArray = (CoordType*****)malloc(sizeof(CoordType****) * element_size_0);
    for (int ii=0; ii<element_size_0; ii++) {
        pointer_5dArray[ii] = (CoordType****)malloc(sizeof(CoordType***) * element_size_1);
        for (int jj=0; jj<element_size_1; jj++) {
            pointer_5dArray[ii][jj] = (CoordType***)malloc(sizeof(CoordType**) * element_size_2);
            for (int kk=0; kk<element_size_2; kk++) {
                pointer_5dArray[ii][jj][kk] = (CoordType**)malloc(sizeof(CoordType*) * element_size_3);
                for (int ll=0; ll<element_size_3; ll++) {
                    pointer_5dArray[ii][jj][kk][ll] = (CoordType*)malloc(sizeof(CoordType) * element_size_4);
                }
            }
        }
    }

    if (init_mark) {
        for (int ii=0; ii<element_size_0; ii++) {
            for (int jj=0; jj<element_size_1; jj++) {
                for (int kk=0; kk<element_size_2; kk++) {
                    for (int ll=0; ll<element_size_3; ll++) {
                        for (int mm=0; mm<element_size_4; mm++)
                            pointer_5dArray[ii][jj][kk][ll][mm] = 0;
                    }
                }
            }
        }
    }

    return pointer_5dArray;
}


template <typename CoordType>
void free5dArray(
                CoordType***** pointer_5dArray,
                int element_size_0,
                int element_size_1,
                int element_size_2,
                int element_size_3)
{
    for (int ii=0; ii<element_size_0; ii++) {
        for (int jj=0; jj<element_size_1; jj++) {
            for (int kk=0; kk<element_size_2; kk++) {
                for (int ll=0; ll<element_size_3; ll++) 
                    free(pointer_5dArray[ii][jj][kk][ll]);
                free(pointer_5dArray[ii][jj][kk]);
            }
            free(pointer_5dArray[ii][jj]);
        }
        free(pointer_5dArray[ii]);
    }
    free(pointer_5dArray);
}


template <typename CoordType>
void show5dArray(
                CoordType***** pointer_5dArray,
                int element_size_0,
                int element_size_1,
                int element_size_2, 
                int element_size_3,
                int element_size_4)
{
    for (int ii=0; ii<element_size_0; ii++) {
        for (int jj=0; jj<element_size_1; jj++) {
            for (int kk=0; kk<element_size_2; kk++) {
                for (int ll=0; ll<element_size_3; ll++) {
                    for (int mm=0; mm<element_size_4; mm++) 
                        printf("[%d, %d, %d, %d, %d] : %f\n", ii, jj, kk, ll, mm, pointer_5dArray[ii][jj][kk][ll][mm]);
                }
            }
        }
    }
}


};  // namespace : matersdk
};  // namespace : arrayUtils



#endif