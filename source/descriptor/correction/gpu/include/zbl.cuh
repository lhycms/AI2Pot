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

#ifndef AI2POT_ZBL_CUH
#define AI2POT_ZBL_CUH

#include <math.h>
#include <chrono>
#include "./zbl_utilities.cuh"


namespace ai2pot {
namespace correction {

//double K_C_SP = 14.399645; // 1/(4*PI*epsilon_0*e^2)


template <typename CoordType>
class PairZBL {
public:
    static __host__ __device__
    CoordType get_K_C_SP() {
        return 14.399645;   // 1/(4*PI*epsilon_0*e^2)
    }

    static __host__ __device__
    CoordType find_switch_func(CoordType rmax,
                               CoordType rmin,
                               CoordType distance_ij);

    static __host__ __device__
    CoordType find_switch_func_der2rij(CoordType rmax,
                                       CoordType rmin,
                                       CoordType distance_ij);

    static __host__ __device__
    CoordType find_phi_func(int Zi,
                            int Zj,
                            CoordType distance_ij,
                            CoordType *ck,
                            CoordType *dk);

    static __host__ __device__
    CoordType find_phi_func_der2rij(int Zi,
                                    int Zj,
                                    CoordType distance_ij,
                                    CoordType *ck,
                                    CoordType *dk);
    
    static __host__ __device__
    CoordType find_pair_energy(int Zi,
                               int Zj,
                               CoordType rmax,
                               CoordType rmin,
                               CoordType distance_ij,
                               CoordType *ck,
                               CoordType *dk);
    
    static __host__ __device__
    void add_atomic_energy_one(CoordType &atomic_energy,
                               int Zi,
                               int Zj,
                               CoordType rmax,
                               CoordType rmin,
                               CoordType distance_ij,
                               CoordType *ck,
                               CoordType *dk);
    
    static __host__ __device__
    void add_atomic_force_one(CoordType *atomic_force,
                              int Zi,
                              int Zj,
                              CoordType rmax,
                              CoordType rmin,
                              CoordType *neigh_vec,
                              CoordType *ck,
                              CoordType *dk);

    static __host__ __device__
    void add_virial_one(CoordType *virial,
                        int Zi,
                        int Zj,
                        CoordType rmax,
                        CoordType rmin,
                        CoordType *neigh_vec,
                        CoordType *ck,
                        CoordType *dk);
};


template <typename CoordType>
static __device__
void correct_zbl_efv_atom(CoordType &etot,
                          CoordType *force,
                          CoordType *virial,
                          CoordType rmax,
                          CoordType rmin,
                          CoordType *cks,
                          CoordType *dks,
                          int silist,
                          int snumneigh,
                          int *sfirstneigh,
                          CoordType (*srcs)[3],
                          int *types,
                          int ntypes,
                          int *type_map,
                          int umax_num_neigh_atoms);


template <typename CoordType>
static __global__
void correct_zbl_efv_kernel(CoordType &etot,
                            CoordType *force,
                            CoordType *virial,
                            CoordType rmax,
                            CoordType rmin,
                            CoordType *cks,
                            CoordType *dks,
                            int inum,
                            int *ilist,
                            int *numneigh,
                            int *firstneigh,
                            CoordType (*rcs)[3],
                            int *types,
                            int ntypes,
                            int *type_map,
                            int umax_num_neigh_atoms);


template <typename CoordType>
static __host__
void correct_zbl_efv_launcher(CoordType &h_etot,
                              CoordType *h_force,
                              CoordType *h_virial,
                              CoordType rmax,
                              CoordType rmin,
                              CoordType *h_cks,
                              CoordType *h_dks,
                              int inum,
                              int *h_ilist,
                              int *h_numneigh,
                              int *h_firstneigh,
                              CoordType (*h_rcs)[3],
                              int *h_types,
                              int ntypes,
                              int *type_map,
                              int umax_num_neigh_atoms);




template <typename CoordType>
__host__ __device__
CoordType PairZBL<CoordType>::find_switch_func(CoordType rmax,
                                               CoordType rmin,
                                               CoordType distance_ij)
{
    if (distance_ij < rmin)
        return 1.0;
    else if ((distance_ij >= rmin) && (distance_ij <= rmax))
    {
        CoordType uu = (distance_ij - rmin) / (rmax - rmin);
        return std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1;
    }
    else 
        return 0.0;
    
}


template <typename CoordType>
__host__ __device__
CoordType PairZBL<CoordType>::find_switch_func_der2rij(CoordType rmax,
                                                       CoordType rmin,
                                                       CoordType distance_ij)
{
    if (distance_ij < rmin)
        return 0.0;
    else if ((distance_ij >= rmin) && (distance_ij <= rmax))
    {
        CoordType uu = (distance_ij - rmin) / (rmax - rmin);
        return 1.0 / (rmax - rmin) * (-30*std::pow(uu, 4) + 60*std::pow(uu, 3) - 30*std::pow(uu, 2));
    }
    else
        return 0.0;
}


template <typename CoordType>
__host__ __device__
CoordType PairZBL<CoordType>::find_phi_func(int Zi,
                                            int Zj,
                                            CoordType distance_ij,
                                            CoordType *ck,
                                            CoordType *dk)
{
    CoordType phi_val = 0.0;
    CoordType a = 0.46848 / (std::pow(Zi, 0.23) + std::pow(Zj, 0.23));
    CoordType x = distance_ij / a;

    for (int ii=0; ii<4; ii++) {
        phi_val += ck[ii] * std::exp(-dk[ii] * x);
    }
    return phi_val;
}


template <typename CoordType>
__host__ __device__
CoordType PairZBL<CoordType>::find_phi_func_der2rij(int Zi,
                                                    int Zj,
                                                    CoordType distance_ij,
                                                    CoordType *ck,
                                                    CoordType *dk)
{
    CoordType phi_der2rij = 0.0;
    CoordType a = 0.46848 / (std::pow(Zi, 0.23) + std::pow(Zj, 0.23));

    for (int ii=0; ii<4; ii++)
        phi_der2rij += - ck[ii] * dk[ii] / a
                       * std::exp(-dk[ii] * distance_ij / a);
    
    return phi_der2rij;
}


template <typename CoordType>
__host__ __device__
CoordType PairZBL<CoordType>::find_pair_energy(int Zi,
                                               int Zj,
                                               CoordType rmax,
                                               CoordType rmin,
                                               CoordType distance_ij,
                                               CoordType *ck,
                                               CoordType *dk)
{
    CoordType phi_val = find_phi_func(Zi, Zj, distance_ij, ck, dk);
    CoordType switch_func_val = find_switch_func(rmax, rmin, distance_ij);
    CoordType pair_energy = get_K_C_SP() * Zi * Zj / distance_ij
                            * phi_val
                            * switch_func_val;
    return pair_energy;
}


template <typename CoordType>
__host__ __device__
void PairZBL<CoordType>::add_atomic_energy_one(CoordType &atomic_energy,
                                               int Zi,
                                               int Zj,
                                               CoordType rmax,
                                               CoordType rmin,
                                               CoordType distance_ij,
                                               CoordType *ck,
                                               CoordType *dk)
{
    CoordType half_pair_energy = 0.5 * find_pair_energy(Zi, Zj, rmax, rmin, distance_ij, ck, dk);
    //atomic_energy += half_pair_energy;
    atomicAdd(&atomic_energy, half_pair_energy);
}


template <typename CoordType>
__host__ __device__
void PairZBL<CoordType>::add_atomic_force_one(CoordType *atomic_force,
                                              int Zi,
                                              int Zj,
                                              CoordType rmax,
                                              CoordType rmin,
                                              CoordType *neigh_vec,
                                              CoordType *ck,
                                              CoordType *dk)
{
    CoordType distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                       + std::pow(neigh_vec[1], 2)
                                       + std::pow(neigh_vec[2], 2) );
    CoordType A = get_K_C_SP() * Zi * Zj / distance_ij;
    CoordType B = find_phi_func(Zi, Zj, distance_ij, ck, dk);
    CoordType C = find_switch_func(rmax, rmin, distance_ij);

    CoordType A_der = -get_K_C_SP() * Zi * Zj / std::pow(distance_ij, 2);
    CoordType B_der = find_phi_func_der2rij(Zi, Zj, distance_ij, ck, dk);
    CoordType C_der = find_switch_func_der2rij(rmax, rmin, distance_ij);

    for (int aa=0; aa<3; aa++) {
        //atomic_force[aa] += (A_der*B*C 
        //                     + A*B_der*C
        //                     + A*B*C_der) * neigh_vec[aa] / distance_ij;
        atomicAdd(&atomic_force[aa], (A_der*B*C 
                                      + A*B_der*C
                                      + A*B*C_der) * neigh_vec[aa] / distance_ij);

    }
}


template <typename CoordType>
__host__ __device__
void PairZBL<CoordType>::add_virial_one(CoordType *virial,
                                        int Zi,
                                        int Zj,
                                        CoordType rmax,
                                        CoordType rmin,
                                        CoordType *neigh_vec,
                                        CoordType *ck,
                                        CoordType *dk)
{
    CoordType distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                       + std::pow(neigh_vec[1], 2)
                                       + std::pow(neigh_vec[2], 2) );
    CoordType A = get_K_C_SP() * Zi * Zj / distance_ij;
    CoordType B = find_phi_func(Zi, Zj, distance_ij, ck, dk);
    CoordType C = find_switch_func(rmax, rmin, distance_ij);

    CoordType A_der = -get_K_C_SP() * Zi * Zj / std::pow(distance_ij, 2);
    CoordType B_der = find_phi_func_der2rij(Zi, Zj, distance_ij, ck, dk);
    CoordType C_der = find_switch_func_der2rij(rmax, rmin, distance_ij);

    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            CoordType Fijb = (A_der*B*C 
                             + A*B_der*C
                             + A*B*C_der) * neigh_vec[bb] / distance_ij;
            
            //virial[aa*3 + bb] += -0.5 * neigh_vec[aa] * Fijb;
            atomicAdd(&virial[aa*3 + bb], -0.5 * neigh_vec[aa] * Fijb);
        }
    }
}


template <typename CoordType>
__device__
void correct_zbl_efv_atom(CoordType &etot,
                          CoordType *force,
                          CoordType *virial,
                          CoordType rmax,
                          CoordType rmin,
                          CoordType *cks,
                          CoordType *dks,
                          int silist,
                          int snumneigh,
                          int *sfirstneigh,
                          CoordType (*srcs)[3],
                          int *types,
                          int ntypes,
                          int *type_map,
                          int umax_num_neigh_atoms)
{
    int center_idx = silist;
    int type_central = types[center_idx];
    int Zi = type_map[type_central];
    int neigh_idx;
    int type_outer;
    int Zj;
    CoordType neigh_vec[3] = {0.0};
    CoordType distance_ij;
    CoordType distance_ij_inv;
    CoordType *ck;
    CoordType *dk;
    CoordType *atomic_force;

    for (int jj=0; jj<snumneigh; jj++) {
        neigh_idx = sfirstneigh[jj];
        type_outer = types[neigh_idx];
        Zj = type_map[type_outer];
        for (int aa=0; aa<3; aa++)
            neigh_vec[aa] = srcs[jj][aa];
        distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                 + std::pow(neigh_vec[1], 2)
                                 + std::pow(neigh_vec[2], 2) );
        if (distance_ij > rmax)
            continue;
        distance_ij_inv = 1.0 / distance_ij;

        int zbl_idx = type_central*ntypes + type_outer;
        ck = &cks[zbl_idx*4];
        dk = &dks[zbl_idx*4];
        atomic_force = &force[center_idx*3 + 0];

        PairZBL<CoordType>::add_atomic_energy_one(etot,
                                                  Zi,
                                                  Zj,
                                                  rmax,
                                                  rmin,
                                                  distance_ij,
                                                  ck,
                                                  dk);
        PairZBL<CoordType>::add_atomic_force_one(atomic_force,
                                                 Zi,
                                                 Zj,
                                                 rmax,
                                                 rmin,
                                                 neigh_vec,
                                                 ck,
                                                 dk);
        PairZBL<CoordType>::add_virial_one(virial,
                                           Zi,
                                           Zj,
                                           rmax,
                                           rmin,
                                           neigh_vec,
                                           ck,
                                           dk);
    }
}


template <typename CoordType>
__global__
void correct_zbl_efv_kernel(CoordType &etot,
                            CoordType *force,
                            CoordType *virial,
                            CoordType rmax,
                            CoordType rmin,
                            CoordType *cks,
                            CoordType *dks,
                            int inum,
                            int *ilist,
                            int *numneigh,
                            int *firstneigh,
                            CoordType (*rcs)[3],
                            int *types,
                            int ntypes,
                            int *type_map,
                            int umax_num_neigh_atoms)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = nx;

    if (ii < inum) {
        int silist = ilist[ii];
        int snumneigh = numneigh[ii];
        int *sfirstneigh = &firstneigh[ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = (CoordType (*)[3])&rcs[ii*umax_num_neigh_atoms][0];

        correct_zbl_efv_atom<CoordType>(etot,
                                        force,
                                        virial,
                                        rmax,
                                        rmin,
                                        cks,
                                        dks,
                                        silist,
                                        snumneigh,
                                        sfirstneigh,
                                        srcs,
                                        types,
                                        ntypes,
                                        type_map,
                                        umax_num_neigh_atoms);
    }
}


template <typename CoordType>
static __host__
void correct_zbl_efv_launcher(CoordType &h_etot,
                              CoordType *h_force,
                              CoordType *h_virial,
                              CoordType rmax,
                              CoordType rmin,
                              CoordType *h_cks,
                              CoordType *h_dks,
                              int inum,
                              int *h_ilist,
                              int *h_numneigh,
                              int *h_firstneigh,
                              CoordType (*h_rcs)[3],
                              int *h_types,
                              int ntypes,
                              int *h_type_map,
                              int umax_num_neigh_atoms)
{
    int block_size_x = 128;
    int grid_size_x = (inum - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_etot_ptr;
    CoordType *d_force;
    CoordType *d_virial;
    CoordType *d_cks;
    CoordType *d_dks;
    int *d_ilist;
    int *d_numneigh;
    int *d_firstneigh;
    CoordType (*d_rcs)[3];
    int *d_types;
    int *d_type_map;

    CHECK_CUDA_API( cudaMalloc((void**)&d_etot_ptr, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMemset(d_etot_ptr, 0.0, sizeof(CoordType)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_force, sizeof(CoordType)*inum*3) );
    CHECK_CUDA_API( cudaMemset(d_force, 0.0, sizeof(CoordType)*inum*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_virial, sizeof(CoordType)*9) );
    CHECK_CUDA_API( cudaMemset(d_virial, 0.0, sizeof(CoordType)*9) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_cks, sizeof(CoordType)*ntypes*ntypes*4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_dks, sizeof(CoordType)*ntypes*ntypes*4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_ilist, sizeof(int)*inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_numneigh, sizeof(int)*inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_firstneigh, sizeof(int)*inum*umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_rcs, sizeof(CoordType)*inum*umax_num_neigh_atoms*3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_types, sizeof(int)*inum) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int)*ntypes) );

    CHECK_CUDA_API( cudaMemcpy(d_cks, h_cks, sizeof(CoordType)*ntypes*ntypes*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_dks, h_dks, sizeof(CoordType)*ntypes*ntypes*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_ilist, h_ilist, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_numneigh, h_numneigh, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_firstneigh, h_firstneigh, sizeof(int)*inum*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_rcs, h_rcs, sizeof(CoordType)*inum*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_types, h_types, sizeof(int)*inum, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );

    auto t1 = std::chrono::high_resolution_clock::now();

    // Launch kernel
    correct_zbl_efv_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (*d_etot_ptr,
                                                                        d_force,
                                                                        d_virial,
                                                                        rmax,
                                                                        rmin,
                                                                        d_cks,
                                                                        d_dks,
                                                                        inum,
                                                                        d_ilist,
                                                                        d_numneigh,
                                                                        d_firstneigh,
                                                                        d_rcs,
                                                                        d_types,
                                                                        ntypes,
                                                                        d_type_map,
                                                                        umax_num_neigh_atoms);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "cost time: " << duration.count() << " ms.\n";

    CHECK_CUDA_API( cudaMemcpy(&h_etot, d_etot_ptr, sizeof(CoordType), cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_force, d_force, sizeof(CoordType)*inum*3, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_virial, d_virial, sizeof(CoordType)*9, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_etot_ptr) );
    CHECK_CUDA_API( cudaFree(d_force) );
    CHECK_CUDA_API( cudaFree(d_virial) );
    CHECK_CUDA_API( cudaFree(d_cks) );
    CHECK_CUDA_API( cudaFree(d_dks) );
    CHECK_CUDA_API( cudaFree(d_ilist) );
    CHECK_CUDA_API( cudaFree(d_numneigh) );
    CHECK_CUDA_API( cudaFree(d_firstneigh) );
    CHECK_CUDA_API( cudaFree(d_rcs) );
    CHECK_CUDA_API( cudaFree(d_types) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
}




};  // namespace : correction
};  // namespace : ai2pot

#endif