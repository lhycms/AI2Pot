/*
    Copyright 2025 Hanyu Liu

    SPDX-License-Identifier: LGPL-3.0-only

    This file is part of AI2Pot.

    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, version 3 of the License.

    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with AI2Pot. If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef AI2POT_GNN_NBLIST_TO_GRAPH_H
#define AI2POT_GNN_NBLIST_TO_GRAPH_H

#include <stdlib.h>

namespace ai2pot {
namespace gnn {

template <typename CoordType>
class GraphBuilder {
public:
    static void find_graph(
        int *slot_offsets,
        int batch_size,
        int natoms_pad,
        int *binum,
        int *bilist,
        int *bnumneigh,
        int *bfirstneigh,
        CoordType (*brcs)[3],
        int *btypes,
        int *bnghost,
        int umax_num_neigh_atoms);
};  // class : GraphBuilder


template <typename CoordType>
void GraphBuilder<CoordType>::find_graph(
    int *slot_offsets,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int *bnghost,
    int umax_num_neigh_atoms)
{   
    // memset(slot_offsets, 0, batch_size*natoms_pad + 1);

    // Step 1.
    #ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int idx=0; idx<(batch_size*natoms_pad); idx++) {
        int bb = idx / natoms_pad;
        int ii = idx - bb * natoms_pad;
        slot_offsets[idx + 1] = (ii < binum[bb]) ? bnumneigh[bb*natoms_pad + bilist[idx]] : 0;
    }

    // Step 2.
    for (int idx=0; idx<(batch_size*natoms_pad); idx++)
        slot_offsets[idx+1] += slot_offsets[idx];
    int num_edges = slot_offsets[batch_size*natoms_pad];
    
    // Step 3.
}

};  // namespace : gnn
};  // namespace : ai2pot

#endif