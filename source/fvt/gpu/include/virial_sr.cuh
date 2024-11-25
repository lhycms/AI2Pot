#ifndef AI2POT_VIRIAL_SR_CUH
#define AI2POT_VIRIAL_SR_CUH
#include "./fvt_utilities.cuh"


namespace ai2pot {
namespace fvt {

template <typename CoordType>
__global__
void find_virial_sr_val_der(
    CoordType *virial_sr_val,
    CoordType *virial_sr_der1,  // Derivative w.r.t. dei_drij
    CoordType *virial_sr_der2,  // Derivative w.r.t. rij
    int inum,
    int *numneigh,
    CoordType *rcs,
    int umax_num_neighs,
    CoordType *dei_drij);


template <typename CoordType>
__global__
void find_virial_sr_val_der(
    CoordType *virial_sr_val,
    CoordType *virial_sr_der1,  // Derivative w.r.t. dei_drij
    CoordType *virial_sr_der2,  // Derivative w.r.t. rij
    int inum,
    int *numneigh,
    CoordType *rcs,
    int umax_num_neighs,
    CoordType *dei_drij)
{
    
}

};  // namespace : fvt
};  // namespace: ai2pot

#endif