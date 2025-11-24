/*
    Copyright 2025 Hanyu Liu
    This file is part of AI2Pot.
    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef AI2POT_FITUTILS_DESCRIPTOR_STATISTICS_H
#define AI2POT_FITUTILS_DESCRIPTOR_STATISTICS_H

#include <math.h>
#include <stdlib.h>
#include <float.h>


namespace ai2pot {
namespace fitutils {


template <typename CoordType>
class AllTypeDescriptorsStatistics {
public:
    static void find_descriptors_statistics(
        int *natoms_in_batch_ptr,
        CoordType *descriptors_mean,
        CoordType *descriptors_M2,
        int batch_size,
        int natoms_pad,
        int descriptor_dim,
        int *binum,
        CoordType *bdescriptors);
};  // class : AllTypeDescriptorsStatistics


template <typename CoordType>
class EachTypeDescriptorsStatistics {
public:
    static void find_descriptors_statistics(
        int *natoms_in_batch,
        CoordType *descriptors_mean,
        CoordType *descriptors_M2,
        int batch_size,
        int natoms_pad,
        int descriptor_dim,
        int *binum,
        int *bilist,
        int *btypes,
        int ntypes,
        CoordType *bdescriptors);
};  // class : EachTypeDescriptorsStatistics


template <typename CoordType>
class AllTypeDescriptorsMaxmin {
public:
    static void find_descriptors_maxmin(
        CoordType *descriptors_max,
        CoordType *descriptors_min,
        int batch_size,
        int natoms_pad,
        int descriptor_dim,
        int *binum,
        CoordType *bdescriptors);
};  // class : AllTypeDescriptorsStatistics






template <typename CoordType>
void AllTypeDescriptorsStatistics<CoordType>::find_descriptors_statistics(
    int *natoms_in_batch_ptr,
    CoordType *descriptors_mean,
    CoordType *descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors)
{
    int natoms_in_batch = 0;
    memset(descriptors_mean, 0, sizeof(CoordType) * descriptor_dim);
    memset(descriptors_M2, 0, sizeof(CoordType) * descriptor_dim);

    for (int bb=0; bb<batch_size; bb++)
        natoms_in_batch += binum[bb];

    for (int kk=0; kk<descriptor_dim; kk++) {
        for (int bb=0; bb<batch_size; bb++) {
            for (int ii=0; ii<binum[bb]; ii++) {
                descriptors_mean[kk] += bdescriptors[bb*natoms_pad*descriptor_dim + ii*descriptor_dim + kk];
            }
        }
        descriptors_mean[kk] = descriptors_mean[kk] / natoms_in_batch;
    }

    for (int kk=0; kk<descriptor_dim; kk++) {
        for (int bb=0; bb<batch_size; bb++) {
            for (int ii=0; ii<binum[bb]; ii++) {
                CoordType tmp_descriptor = bdescriptors[bb*natoms_pad*descriptor_dim + ii*descriptor_dim + kk];
                descriptors_M2[kk] += std::pow(tmp_descriptor - descriptors_mean[kk], 2);
            }
        }
    }

    // Step . Reassign
    (*natoms_in_batch_ptr) = natoms_in_batch;
}


template <typename CoordType>
void EachTypeDescriptorsStatistics<CoordType>::find_descriptors_statistics(
    int *natoms_in_batch,
    CoordType *descriptors_mean,
    CoordType *descriptors_M2,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    int *bilist,
    int *btypes,
    int ntypes,
    CoordType *bdescriptors)
{
    memset(natoms_in_batch, 0, sizeof(int)*ntypes);
    memset(descriptors_mean, 0, sizeof(CoordType)*ntypes*descriptor_dim);
    memset(descriptors_M2, 0, sizeof(CoordType)*ntypes*descriptor_dim);

    for (int bb=0; bb<batch_size; bb++) {
        for (int ii=0; ii<binum[bb]; ii++) {
            int center_idx = bilist[bb*natoms_pad + ii];
            int type_central = btypes[bb*natoms_pad + center_idx];
            natoms_in_batch[type_central]++;
        }
    }

    for (int kk=0; kk<descriptor_dim; kk++) {
        for (int bb=0; bb<batch_size; bb++) {
            for (int ii=0; ii<binum[bb]; ii++) {
                int center_idx = bilist[bb*natoms_pad + ii];
                int type_central = btypes[bb*natoms_pad + center_idx];

                descriptors_mean[type_central*descriptor_dim + kk] += bdescriptors[bb*natoms_pad*descriptor_dim
                                                                                   + ii*descriptor_dim
                                                                                   + kk];
            }
        }

        for (int tt=0; tt<ntypes; tt++) {
            if (natoms_in_batch[tt] != 0)
                descriptors_mean[tt*descriptor_dim + kk] = descriptors_mean[tt*descriptor_dim + kk] / natoms_in_batch[tt];
        }
    }

    for (int kk=0; kk<descriptor_dim; kk++) {
        for (int bb=0; bb<batch_size; bb++) {
            for (int ii=0; ii<binum[bb]; ii++) {
                int center_idx = bilist[bb*natoms_pad + ii];
                int type_central = btypes[bb*natoms_pad + center_idx];
                
                CoordType tmp_descriptor = bdescriptors[bb*natoms_pad*descriptor_dim
                                                        + ii*descriptor_dim
                                                        + kk];
                descriptors_M2[type_central*descriptor_dim + kk] += std::pow(tmp_descriptor-descriptors_mean[type_central*descriptor_dim + kk], 2);
            }
        }
    }
}


template <typename CoordType>
void AllTypeDescriptorsMaxmin<CoordType>::find_descriptors_maxmin(
    CoordType *descriptors_max,
    CoordType *descriptors_min,
    int batch_size,
    int natoms_pad,
    int descriptor_dim,
    int *binum,
    CoordType *bdescriptors)
{
    for (int ii=0; ii<descriptor_dim; ii++) {
        descriptors_max[ii] = FLT_MIN;
        descriptors_min[ii] = FLT_MAX;
    }

    for (int kk=0; kk<descriptor_dim; kk++) {
        for (int bb=0; bb<batch_size; bb++) {
            for (int ii=0; ii<binum[bb]; ii++) {
                int tmp_descriptor_idx = bb*natoms_pad*descriptor_dim + ii*descriptor_dim + kk;
                descriptors_max[kk] = (descriptors_max[kk] > bdescriptors[tmp_descriptor_idx]) ? descriptors_max[kk] : bdescriptors[tmp_descriptor_idx];
                descriptors_min[kk] = (descriptors_min[kk] < bdescriptors[tmp_descriptor_idx]) ? descriptors_min[kk] : bdescriptors[tmp_descriptor_idx];
            }
        }
    }
}

};  // namespace : fituitls
};  // namespace : ai2pot


#endif