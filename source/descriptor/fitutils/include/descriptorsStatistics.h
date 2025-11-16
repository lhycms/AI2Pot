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
};  // class : AllDescriptorStatistics



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


};  // namespace : fituitls
};  // namespace : ai2pot


#endif