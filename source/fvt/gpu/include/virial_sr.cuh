#ifndef AI2POT_VIRIAL_SR_CUH
#define AI2POT_VIRIAL_SR_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "./fvt_utilities.cuh"


namespace ai2pot {
namespace fvt {

template <typename CoordType>
__global__
static void find_virial_sr_val_der(
    CoordType *virial_sr_val,
    CoordType *virial_sr_der1,  // Derivative w.r.t. dei_drij
    CoordType *virial_sr_der2,  // Derivative w.r.t. rij
    int inum,
    int *numneigh,
    CoordType *rcs,
    int umax_num_neighs,
    CoordType *dei_drij);

template <typename CoordType>
static void find_virial_sr_val_der_launcher(
    CoordType *virial_sr_val,
    CoordType *virial_sr_der1,
    CoordType *virial_sr_der2,
    int inum,
    int *numneigh,
    CoordType *rcs,
    int umax_num_neighs,
    CoordType *dei_drij);



template <typename CoordType>
__global__
static void find_virial_sr_val_der(
    CoordType *virial_sr_val,
    CoordType *virial_sr_der1,  // Derivative w.r.t. dei_drij
    CoordType *virial_sr_der2,  // Derivative w.r.t. rij
    int inum,
    int *numneigh,
    CoordType *rcs,
    int umax_num_neighs,
    CoordType *dei_drij)
{
    /* Wrong for CUDA:
    memset(virial_sr_val, sizeof(CoordType) * 3 * 3);
    memset(virial_sr_der1, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs);
    memset(virial_sr_der2, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs);
    */
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    const int jj = nx;
    const int ii = ny;
    if ( (ii<inum) && (jj<numneigh[ii]) ) {
        for (int aa=0; aa<3; aa++) {
            for (int bb=0; bb<3; bb++) {
                int tmp_de_idx = ii*umax_num_neighs*3 + jj*3 + aa;
                int tmp_rij_idx = ii*umax_num_neighs*3 + jj*3 + bb;
                atomicAdd(&virial_sr_val[aa*3 + bb], dei_drij[tmp_de_idx] * rcs[tmp_rij_idx]);
                atomicAdd(&virial_sr_der1[(aa*3+bb)*inum*umax_num_neighs + ii*umax_num_neighs + jj],
                          rcs[tmp_rij_idx]);
                atomicAdd(&virial_sr_der2[(aa*3+bb)*inum*umax_num_neighs + ii*umax_num_neighs + jj],
                          dei_drij[tmp_de_idx]);
            }
        }
    }
}


template <typename CoordType>
static void find_virial_sr_val_der_launcher(
    CoordType *h_virial_sr_val,
    CoordType *h_virial_sr_der1,
    CoordType *h_virial_sr_der2,
    int inum,
    int *h_numneigh,
    CoordType *h_rcs,
    int umax_num_neighs,
    CoordType *h_dei_drij)
{
    int grid_size_x = (umax_num_neighs - 1) / BLOCK_SIZE_X + 1;
    int grid_size_y = (inum - 1) / BLOCK_SIZE_Y + 1;
    CoordType *d_virial_sr_val;
    CoordType *d_virial_sr_der1;
    CoordType *d_virial_sr_der2;
    int *d_numneigh;
    CoordType *d_rcs;
    CoordType *d_dei_drij;

    CHECK( cudaMalloc((void**)&d_virial_sr_val, sizeof(CoordType) * 3 * 3) );
CHECK( cudaMemset(d_virial_sr_val, 0, sizeof(CoordType) * 3 * 3) );
    CHECK( cudaMalloc((void**)&d_virial_sr_der1, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs) );
CHECK( cudaMemset(d_virial_sr_der1, 0, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs) );
    CHECK( cudaMalloc((void**)&d_virial_sr_der2, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs) );
CHECK( cudaMemset(d_virial_sr_der2, 0, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs) );
    CHECK( cudaMalloc((void**)&d_numneigh, sizeof(int) * inum * umax_num_neighs) );
    CHECK( cudaMalloc((void**)&d_rcs, sizeof(CoordType) * inum * umax_num_neighs * 3) );
    CHECK( cudaMalloc((void**)&d_dei_drij, sizeof(CoordType) * inum * umax_num_neighs * 3) );

    CHECK( cudaMemcpy(d_numneigh, h_numneigh, sizeof(int) * inum * umax_num_neighs, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_rcs, h_rcs, sizeof(CoordType) * inum * umax_num_neighs * 3, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_dei_drij, h_dei_drij, sizeof(CoordType) * inum * umax_num_neighs * 3, cudaMemcpyHostToDevice) );

    dim3 grid_size(grid_size_x, grid_size_y);
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    find_virial_sr_val_der KERNEL_ARG2(grid_size, block_size) (
        d_virial_sr_val,
        d_virial_sr_der1,
        d_virial_sr_der2,
        inum,
        d_numneigh,
        d_rcs,
        umax_num_neighs,
        d_dei_drij);
    
    CHECK( cudaMemcpy(h_virial_sr_val, d_virial_sr_val, sizeof(CoordType) * 3 * 3, cudaMemcpyDeviceToHost) );
    CHECK( cudaMemcpy(h_virial_sr_der1, d_virial_sr_der1, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs, cudaMemcpyDeviceToHost) );
    CHECK( cudaMemcpy(h_virial_sr_der2, d_virial_sr_der2, sizeof(CoordType) * 3 * 3 * inum * umax_num_neighs, cudaMemcpyDeviceToHost) );

    CHECK( cudaFree(d_virial_sr_val) );
    CHECK( cudaFree(d_virial_sr_der1) );
    CHECK( cudaFree(d_virial_sr_der2) );
    CHECK( cudaFree(d_numneigh) );
    CHECK( cudaFree(d_rcs) );
    CHECK( cudaFree(d_dei_drij) );
}


};  // namespace : fvt
};  // namespace: ai2pot

#endif