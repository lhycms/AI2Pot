#ifndef AI2POT_FORCE_SR_H
#define AI2POT_FORCE_SR_H
#include <cstring>

namespace ai2pot {
namespace fvt {

template <typename CoordType>
class ForceSr {
public:
    static void find_val_der(
        CoordType *force_sr_val,
        CoordType *force_sr_der,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        int nghost,
        int umax_num_neighs,
        CoordType *dei_drij);
};  // class : ForceSr


/**
 * @brief Construct a new ForceSr<CoordType>::find_val_der object
 * 
 * @tparam CoordType 
 * @param force_sr_val Value of force on atom_i
 * @param force_sr_der Derivative w.r.t. dei_drij
 * @param inum Number of atoms in primitive cell
 * @param ilist 
 * @param numneigh 
 * @param firstneigh 
 * @param nghost Number of ghost atoms
 * @param umax_num_neighs Total number of neighbor atoms specify by user.
 * @param dei_drij $\frac{\partial E_i}{\partial r_{ij\alpha}}$
 */
template <typename CoordType>
void ForceSr<CoordType>::find_val_der(
    CoordType *force_sr_val,
    CoordType *force_sr_der,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    int nghost,
    int umax_num_neighs,
    CoordType *dei_drij)
{
    memset(force_sr_val, 0, sizeof(CoordType) * (inum+nghost) * 3);
    memset(force_sr_der, 0, sizeof(CoordType) * (inum+nghost) * 3 * inum * umax_num_neighs);

    for (int ii=0; ii<inum; ii++) {
        int center_idx = ilist[ii];
        for (int jj=0; jj<numneigh[ii]; jj++) {
            int neigh_idx = firstneigh[ii*umax_num_neighs + jj];
            for (int kk=0; kk<3; kk++) {
                int tmp_de_idx = ii*umax_num_neighs*3 + jj*3 + kk;
                force_sr_val[center_idx*3 + kk] += dei_drij[tmp_de_idx];
                force_sr_val[neigh_idx*3 + kk] -= dei_drij[tmp_de_idx];
                force_sr_der[(center_idx*3+kk)*inum*umax_num_neighs 
                             + ii*umax_num_neighs
                             + jj] += 1;
                force_sr_der[(neigh_idx*3+kk)*inum*umax_num_neighs
                             + ii*umax_num_neighs
                             + jj] -= 1;
            }
        }
    }
}

};  // namespace : fvt
};  // namespace : ai2pot

#endif
