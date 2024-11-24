#ifndef AI2POT_FORCE_SR_CUH
#define AI2POT_FORCE_SR_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include "./fvt_utilities.cuh"


namespace ai2pot {
namespace fvt {

template <typename CoordType>
__global__
static void find_force_sr_val_der(
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
static void find_force_sr_val_der(
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
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    int ii = nx;
    int jj = ny;
    if ( (ii<inum) && (jj<numneigh[ii]) ) {
        const int center_idx = ilist[ii];
        const int neigh_idx = firstneigh[ii*umax_num_neighs + jj];
        for (int kk=0; kk<3; kk++) {
            int tmp_de_idx = ii*umax_num_neighs*3 + jj*3 + kk;
            
            atomicAdd(&force_sr_val[center_idx*3 + kk], dei_drij[tmp_de_idx]);
            atomicAdd(&force_sr_val[neigh_idx*3 + kk], -dei_drij[tmp_de_idx]);
            atomicAdd(&force_sr_der[(center_idx*3 + kk)*inum*umax_num_neighs 
                        + ii*umax_num_neighs 
                        + jj], 1);
            atomicAdd(&force_sr_der[(neigh_idx*3 + kk)*inum*umax_num_neighs
                        + ii*umax_num_neighs
                        + jj], -1);
        }
    }
}


template <typename CoordType>
static void find_force_sr_val_der_launcher(
    CoordType *h_force_sr_val,
    CoordType *h_force_sr_der,
    int inum,
    int *h_ilist,
    int *h_numneigh,
    int *h_firstneigh,
    int nghost,
    int umax_num_neighs,
    CoordType *h_dei_drij)
{
    int grid_size_x = (inum - 1) / BLOCK_SIZE_X + 1;
    int grid_size_y = (umax_num_neighs - 1) / BLOCK_SIZE_Y + 1;
    CoordType *d_force_sr_val;
    CoordType *d_force_sr_der;
    int *d_ilist;
    int *d_numneigh;
    int *d_firstneigh;
    CoordType *d_dei_drij;
    CHECK( cudaMalloc((void**)&d_force_sr_val, sizeof(CoordType) * (inum+nghost) * 3) );
    CHECK( cudaMalloc((void**)&d_force_sr_der, sizeof(CoordType) * (inum+nghost) * 3 * inum * umax_num_neighs) );
    CHECK( cudaMalloc((void**)&d_ilist, sizeof(int) * inum) );
    CHECK( cudaMalloc((void**)&d_numneigh, sizeof(int) * inum) );
    CHECK( cudaMalloc((void**)&d_firstneigh, sizeof(int) * inum * umax_num_neighs) );
    CHECK( cudaMalloc((void**)&d_dei_drij, sizeof(CoordType) * inum * umax_num_neighs * 3) );

    CHECK( cudaMemcpy(d_ilist, h_ilist, sizeof(int) * inum, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_numneigh, h_numneigh, sizeof(int) * inum, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_firstneigh, h_firstneigh, sizeof(int) * inum * umax_num_neighs, cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_dei_drij, h_dei_drij, sizeof(double) * inum * umax_num_neighs * 3, cudaMemcpyHostToDevice) );

    dim3 grid_size(grid_size_x, grid_size_y);
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    find_force_sr_val_der KERNEL_ARG2(grid_size, block_size) (
        d_force_sr_val,
        d_force_sr_der,
        inum,
        d_ilist,
        d_numneigh,
        d_firstneigh,
        nghost,
        umax_num_neighs,
        d_dei_drij);

    CHECK( cudaMemcpy(h_force_sr_val, d_force_sr_val, sizeof(CoordType) * (inum + nghost) * 3, cudaMemcpyDeviceToHost));
    CHECK( cudaMemcpy(h_force_sr_der, d_force_sr_der, sizeof(CoordType) * (inum + nghost) * 3 * inum * umax_num_neighs, cudaMemcpyDeviceToHost));

    CHECK( cudaFree(d_force_sr_val) );
    CHECK( cudaFree(d_force_sr_der) );
    CHECK( cudaFree(d_ilist) );
    CHECK( cudaFree(d_numneigh) );
    CHECK( cudaFree(d_firstneigh) );
    CHECK( cudaFree(d_dei_drij) );
}

};  // fvt
};  // ai2pot


#endif
