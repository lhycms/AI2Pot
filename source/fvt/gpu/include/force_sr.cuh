#ifndef AI2POT_FORCE_SR_CUH
#define AI2POT_FORCE_SR_CUH
#include "./fvt_utilities.cuh"


namespace ai2pot {
namespace fvt {

template <typename CoordType>
__global__
void find_force_sr_val_der(
    CoordType *force_sr_val,
    CoordType *force_sr_der,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    int nghost,
    int umax_num_neighs,
    CoordType *dei_drij);



template <typename CoordType>
__global__
void find_force_sr_val_der(
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
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if ((nx < inum) && (ny < umax_num_neighs)) {
        int ii = nx;
        int jj = ny;
        int center_idx = ilist[ii];
        int neigh_idx = firstneigh[ii*umax_num_neighs + jj];
        for (int kk=0; kk<3; kk++) {
            int tmp_idx = ii*umax_num_neighs*3 + jj*3 + kk;
            force_sr_val[center_idx*3 + kk] += dei_drij[tmp_idx];
            force_sr_val[neigh_idx*3 + kk] -= dei_drij[tmp_idx];
            force_sr_der[(center_idx*3 + kk)*inum*umax_num_neighs*3 + tmp_idx] += 1;
            force_sr_der[(neigh_idx*3 + kk)*inum*umax_num_neighs*3 + tmp_idx] -= 1;
        }
    }
}


};  // namespace : fvt
};  // namespace : ai2pot

#endif
