#ifndef AI2POT_VIRIAL_SR_H
#define AI2POT_VIRIAL_SR_H
#include <cstring>

namespace ai2pot {
namespace fvt {

template <typename CoordType>
class VirialSr {
public:
    static void find_val_der(
        CoordType *virial_sr_val,
        CoordType *virial_sr_der1,  // Derivative w.r.t. dei_drij
        CoordType *virial_sr_der2,  // Derivative w.r.t. rc
        int inum,
        int *numneigh,
        CoordType *rcs,
        int umax_num_neighs,
        CoordType *dei_drij);
};  // class : VirialSr


template <typename CoordType>
void VirialSr<CoordType>::find_val_der(
    CoordType *virial_sr_val,
    CoordType *virial_sr_der1,   // Derivative w.r.t. dei_drij
    CoordType *virial_sr_der2,   // Derivative w.r.t. rc
    int inum,
    int *numneigh,
    CoordType *rcs,
    int umax_num_neighs,
    CoordType *dei_drij)
{
    memset(virial_sr_val, 0, sizeof(CoordType) * 3 * 3);
    memset(virial_sr_der1, 0, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs);
    memset(virial_sr_der2, 0, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs);

    for (int ii=0; ii<inum; ii++) {
        for (int jj=0; jj<numneigh[ii]; jj++) {
            for (int aa=0; aa<3; aa++) {
                for (int bb=0; bb<3; bb++) {
                    int tmp_de_idx = ii*umax_num_neighs*3 + jj*3 + aa;
                    int tmp_rij_idx = ii*umax_num_neighs*3 + jj*3 + bb;
                    virial_sr_val[aa*3 + bb] -= dei_drij[tmp_de_idx] * rcs[tmp_rij_idx];
                    virial_sr_der1[(aa*3+bb)*inum*umax_num_neighs + ii*umax_num_neighs + jj] -= rcs[tmp_rij_idx];
                    virial_sr_der2[(aa*3+bb)*inum*umax_num_neighs + ii*umax_num_neighs + jj] -= dei_drij[tmp_de_idx];
                }
            }
        }
    }
}

};  // namespace : fvt
};  // namespace : ai2pot

#endif
