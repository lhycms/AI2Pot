#ifndef AI2POT_DP_UTILS_H
#define AI2POT_DP_UTILS_H
#include <stdlib.h>
#include <string.h>


namespace ai2pot {
namespace deepPotSE {

template <typename CoordType>
static void find_dp_firstneigh(
    int *dp_firstneigh,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType *relative_coords,
    int *types,
    int ntypes,
    int *umax_num_neigh_atoms_lst);


template <typename CoordType>
void find_dp_firstneigh(
    int *dp_firstneigh,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType *relative_coords,
    int *types,
    int ntypes,
    int *umax_num_neigh_atoms_lst)
{   
    int umax_num_neigh_atoms = 0;
    for (int ii=0; ii<ntypes; ii++)
        umax_num_neigh_atoms += umax_num_neigh_atoms_lst[ii];
    memset(dp_firstneigh, -1, sizeof(int) * inum * umax_num_neigh_atoms);

    int *nloops = (int*)malloc(sizeof(int) * ntypes);

    for (int ii=0; ii<inum; ii++) {
        memset(nloops, 0, sizeof(int) * ntypes);
        for (int jj=0; jj<numneigh[ii]; jj++) {
            int jtype = types[firstneigh[ii*umax_num_neigh_atoms + jj]];
            int jtype_start;
            if (jtype==0)
                jtype_start = 0;
            else
                jtype_start = umax_num_neigh_atoms_lst[jtype-1];

            dp_firstneigh[ii*umax_num_neigh_atoms + jj] = jtype_start + nloops[jtype];
            nloops[jtype]++;
        }
    }

    // Step . Free
    free(nloops);
}

};  // namespace : deepPotSE
};  // namespace : ai2pot

#endif
