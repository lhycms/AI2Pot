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

#ifndef AI2POT_FITUTILS_TARGET_STATISTICS_H
#define AI2POT_FITUTILS_TARGET_STATISTICS_H

#include <math.h>

namespace ai2pot {
namespace fitutils {


template <typename CoordType>
class TargetStatistics {
public:
    static void find_ef_statistics(
        int *natoms_in_batch_ptr,
        CoordType *ei_mean_ptr,
        CoordType *ei_M2_ptr,
        CoordType *force_M2_ptr,
        int batch_size,
        int natoms_pad,
        int *binum,
        CoordType *betot_dft,
        CoordType (*bforce_dft)[3]);
};  // class: TargetNormalizer


template <typename CoordType>
void TargetStatistics<CoordType>::find_ef_statistics(
    int *natoms_in_batch_ptr,
    CoordType *ei_mean_ptr,
    CoordType *ei_M2_ptr,
    CoordType *force_M2_ptr,
    int batch_size,
    int natoms_pad,
    int *binum,
    CoordType *betot_dft,
    CoordType (*bforce_dft)[3])
{
    int natoms_in_batch = 0;
    CoordType ei_mean = 0.0;
    CoordType ei_M2 = 0.0;
    CoordType force_M2 = 0.0;

    // Step . natoms_in_batch && ei_mean && force_M2
    for (int bb=0; bb<batch_size; bb++) {
        natoms_in_batch += binum[bb];
        ei_mean += betot_dft[bb] / binum[bb];
        
        for (int ii=0; ii<binum[bb]; ii++)
            for (int aa=0; aa<3; aa++)
                force_M2 += std::pow(bforce_dft[bb*natoms_pad + ii][aa], 2);
    }
    ei_mean = ei_mean / batch_size;

    // Step . ei_M2
    for (int bb=0; bb<batch_size; bb++)
        ei_M2 += std::pow((betot_dft[bb]/binum[bb] - ei_mean), 2);

    // Step . Final
    (*natoms_in_batch_ptr) = natoms_in_batch;
    (*ei_mean_ptr) = ei_mean;
    (*ei_M2_ptr) = ei_M2;
    (*force_M2_ptr) = force_M2;
}

};  // namespace: fitutils
};  // namespace: ai2pot

#endif
