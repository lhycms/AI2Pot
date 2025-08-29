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
#include <iostream>
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
    void add_atomic_energy_one(CoordType *atomic_energy_ptr,
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
                        CoordType *dk,
                        CoordType *s_local_virial);
};


template <typename CoordType>
static __device__
void correct_zbl_efv_atom(
    CoordType *etot_ptr,
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
    int umax_num_neigh_atoms,
    CoordType *s_local_virial);


template <typename CoordType>
static __global__
void correct_zbl_efv_kernel(
    CoordType *betot_ptr,
    CoordType *bforce,
    CoordType *bvirial,
    CoordType rmax,
    CoordType rmin,
    CoordType *cks,
    CoordType *dks,
    int batch_size, 
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost);


template <typename CoordType>
static __host__
void correct_zbl_efv_launcher(
    CoordType *h_betot_ptr,
    CoordType *h_bforce,
    CoordType *h_bvirial,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_cks,
    CoordType *h_dks,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    int *h_bnumneigh,
    int *h_bfirstneigh,
    CoordType (*h_brcs)[3],
    int *h_btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost);

template <typename CoordType>
static __device__
void correct_zbl_ef_atom(CoordType *etot_ptr,
                         CoordType *force,
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
void correct_zbl_ef_kernel(
    CoordType *betot_ptr,
    CoordType *bforce,
    CoordType rmax,
    CoordType rmin,
    CoordType *cks,
    CoordType *dks,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost);


template <typename CoordType>
static __host__
void correct_zbl_ef_launcher(
    CoordType *h_betot_ptr,
    CoordType *h_bforce,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_cks,
    CoordType *h_dks,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    int *h_bnumneigh,
    int *h_bfirstneigh,
    CoordType (*h_brcs)[3],
    int *h_btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost);






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
void PairZBL<CoordType>::add_atomic_energy_one(CoordType *atomic_energy_ptr,
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
    atomicAdd(atomic_energy_ptr, half_pair_energy);
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
                                        CoordType *dk,
                                        CoordType *s_local_virial)
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
            
            //atomicAdd(&virial[aa*3 + bb], -0.5 * neigh_vec[aa] * Fijb);
            s_local_virial[aa*3 + bb] -= 0.5 * neigh_vec[aa] * Fijb;
        }
    }
}


template <typename CoordType>
__device__
void correct_zbl_efv_atom(
    CoordType *etot_ptr,
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
    int umax_num_neigh_atoms,
    CoordType *s_local_virial)
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

        PairZBL<CoordType>::add_atomic_energy_one(etot_ptr,
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
                                           dk,
                                           s_local_virial);
    }
}


template <typename CoordType>
__global__
void correct_zbl_efv_kernel(
    CoordType *betot_ptr,
    CoordType *bforce,
    CoordType *bvirial,
    CoordType rmax,
    CoordType rmin,
    CoordType *cks,
    CoordType *dks,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    int ii = nx % natoms_pad;
    if (istruct >= batch_size)
        return;

    __shared__ CoordType s_local_virial[64][9];
    int tid = threadIdx.x;
    for (int ii=0; ii<9; ii++)
        s_local_virial[tid][ii] = 0.0;

    CoordType *etot_ptr = &betot_ptr[istruct];
    CoordType *force = &bforce[istruct*(natoms_pad+nghost)*3 + 0];
    CoordType *virial = &bvirial[istruct*9 + 0];
    int inum = binum[istruct];
    int *types = &btypes[istruct*(natoms_pad+nghost)];

    if (ii < inum) {
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms];
        CoordType (*srcs)[3] = (CoordType (*)[3])(&brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms][0]);

        correct_zbl_efv_atom<CoordType>(
            etot_ptr,
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
            umax_num_neigh_atoms,
            s_local_virial[tid]);
    }
    __syncthreads();

    if (tid == 0) {
        for (int t=0; t<blockDim.x; t++)
            for (int i=0; i<9; i++)
                atomicAdd(&virial[i], s_local_virial[t][i]);
    }
}


template <typename CoordType>
static __host__
void correct_zbl_efv_launcher(
    CoordType *h_betot_ptr,
    CoordType *h_bforce,
    CoordType *h_bvirial,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_cks,
    CoordType *h_dks,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    int *h_bnumneigh,
    int *h_bfirstneigh,
    CoordType (*h_brcs)[3],
    int *h_btypes,
    int ntypes,
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size*natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_betot_ptr;
    CoordType *d_bforce;
    CoordType *d_bvirial;
    CoordType *d_cks;
    CoordType *d_dks;
    int *d_binum;
    int *d_bilist;
    int *d_bnumneigh;
    int *d_bfirstneigh;
    CoordType (*d_brcs)[3];
    int *d_btypes;
    int *d_type_map;

    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ptr, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMemset(d_betot_ptr, 0.0, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce, sizeof(CoordType) * batch_size * (natoms_pad+nghost) * 3) );
    CHECK_CUDA_API( cudaMemset(d_bforce, 0.0, sizeof(CoordType) * batch_size * (natoms_pad+nghost) * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bvirial, sizeof(CoordType) * batch_size * 9) );
    CHECK_CUDA_API( cudaMemset(d_bvirial, 0.0, sizeof(CoordType) * batch_size * 9) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_cks, sizeof(CoordType) * ntypes * ntypes * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_dks, sizeof(CoordType) * ntypes * ntypes * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(int) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int) * batch_size * (natoms_pad+nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );

    CHECK_CUDA_API( cudaMemcpy(d_cks, h_cks, sizeof(CoordType) * ntypes * ntypes * 4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_dks, h_dks, sizeof(CoordType) * ntypes * ntypes * 4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*(natoms_pad+nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );

    auto t1 = std::chrono::high_resolution_clock::now();

    // Launch kernel
    correct_zbl_efv_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_betot_ptr,
        d_bforce,
        d_bvirial,
        rmax,
        rmin,
        d_cks,
        d_dks,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        d_bnumneigh,
        d_bfirstneigh,
        d_brcs,
        d_btypes,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms,
        nghost);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "cost time: " << duration.count() << " ms.\n";

    CHECK_CUDA_API( cudaMemcpy(h_betot_ptr, d_betot_ptr, sizeof(CoordType)*batch_size, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bforce, d_bforce, sizeof(CoordType)*batch_size*(natoms_pad+nghost)*3, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bvirial, d_bvirial, sizeof(CoordType)*batch_size*9, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_betot_ptr) );
    CHECK_CUDA_API( cudaFree(d_bforce) );
    CHECK_CUDA_API( cudaFree(d_bvirial) );
    CHECK_CUDA_API( cudaFree(d_cks) );
    CHECK_CUDA_API( cudaFree(d_dks) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_bnumneigh) );
    CHECK_CUDA_API( cudaFree(d_bfirstneigh) );
    CHECK_CUDA_API( cudaFree(d_brcs) );
    CHECK_CUDA_API( cudaFree(d_btypes) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
}


template <typename CoordType>
__device__
void correct_zbl_ef_atom(CoordType *etot_ptr,
                         CoordType *force,
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

        PairZBL<CoordType>::add_atomic_energy_one(etot_ptr,
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
    }
}


template <typename CoordType>
__global__
void correct_zbl_ef_kernel(
    CoordType *betot_ptr,
    CoordType *bforce,
    CoordType rmax,
    CoordType rmin,
    CoordType *cks,
    CoordType *dks,
    int batch_size,
    int natoms_pad,
    int *binum,
    int *bilist,
    int *bnumneigh,
    int *bfirstneigh,
    CoordType (*brcs)[3],
    int *btypes,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost)
{
    int nx = blockIdx.x * blockDim.x + threadIdx.x;
    int istruct = nx / natoms_pad;
    int ii = nx % natoms_pad;
    if (istruct >= batch_size)
        return;

    CoordType *etot_ptr = &betot_ptr[istruct];
    CoordType *force = &bforce[istruct*(natoms_pad+nghost)*3 + 0];
    int inum = binum[istruct];
    int *types = &btypes[istruct*(natoms_pad+nghost)];

    if (ii < inum) {
        int silist = bilist[istruct*natoms_pad + ii];
        int snumneigh = bnumneigh[istruct*natoms_pad + ii];
        int *sfirstneigh = &bfirstneigh[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms + 0];
        CoordType (*srcs)[3] = (CoordType (*)[3])&brcs[istruct*natoms_pad*umax_num_neigh_atoms + ii*umax_num_neigh_atoms + 0][0];

        correct_zbl_ef_atom<CoordType>(
            etot_ptr,
            force,
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
void correct_zbl_ef_launcher(
    CoordType *h_betot_ptr,
    CoordType *h_bforce,
    CoordType rmax,
    CoordType rmin,
    CoordType *h_cks,
    CoordType *h_dks,
    int batch_size,
    int natoms_pad,
    int *h_binum,
    int *h_bilist,
    int *h_bnumneigh,
    int *h_bfirstneigh,
    CoordType (*h_brcs)[3],
    int *h_btypes,
    int ntypes,
    int *h_type_map,
    int umax_num_neigh_atoms,
    int nghost)
{
    int block_size_x = 64;
    int grid_size_x = (batch_size * natoms_pad - 1) / block_size_x + 1;
    dim3 grid_size(grid_size_x);
    dim3 block_size(block_size_x);

    CoordType *d_betot_ptr;
    CoordType *d_bforce;
    CoordType *d_cks;
    CoordType *d_dks;
    int *d_binum;
    int *d_bilist;
    int *d_bnumneigh;
    int *d_bfirstneigh;
    CoordType (*d_brcs)[3];
    int *d_btypes;
    int *d_type_map;

    CHECK_CUDA_API( cudaMalloc((void**)&d_betot_ptr, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMemset(d_betot_ptr, 0.0, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bforce, sizeof(CoordType) * batch_size * (natoms_pad+nghost) * 3) );
    CHECK_CUDA_API( cudaMemset(d_bforce, 0.0, sizeof(CoordType) * batch_size * (natoms_pad+nghost) * 3) );

    CHECK_CUDA_API( cudaMalloc((void**)&d_cks, sizeof(CoordType) * ntypes * ntypes * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_dks, sizeof(CoordType) * ntypes * ntypes * 4) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_binum, sizeof(CoordType) * batch_size) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bilist, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bnumneigh, sizeof(int) * batch_size * natoms_pad) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_bfirstneigh, sizeof(int) * batch_size * natoms_pad * umax_num_neigh_atoms) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_brcs, sizeof(CoordType) * batch_size * natoms_pad * umax_num_neigh_atoms * 3) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_btypes, sizeof(int) * batch_size * (natoms_pad+nghost)) );
    CHECK_CUDA_API( cudaMalloc((void**)&d_type_map, sizeof(int) * ntypes) );

    CHECK_CUDA_API( cudaMemcpy(d_cks, h_cks, sizeof(CoordType)*ntypes*ntypes*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_dks, h_dks, sizeof(CoordType)*ntypes*ntypes*4, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_binum, h_binum, sizeof(int)*batch_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bilist, h_bilist, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bnumneigh, h_bnumneigh, sizeof(int)*batch_size*natoms_pad, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_bfirstneigh, h_bfirstneigh, sizeof(int)*batch_size*natoms_pad*umax_num_neigh_atoms, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_brcs, h_brcs, sizeof(CoordType)*batch_size*natoms_pad*umax_num_neigh_atoms*3, cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_btypes, h_btypes, sizeof(int)*batch_size*(natoms_pad+nghost), cudaMemcpyHostToDevice) );
    CHECK_CUDA_API( cudaMemcpy(d_type_map, h_type_map, sizeof(int)*ntypes, cudaMemcpyHostToDevice) );

    auto t1 = std::chrono::high_resolution_clock::now();

    // Launch kernel
    correct_zbl_ef_kernel<CoordType> KERNEL_ARG2(grid_size, block_size) (
        d_betot_ptr,
        d_bforce,
        rmax,
        rmin,
        d_cks,
        d_dks,
        batch_size,
        natoms_pad,
        d_binum,
        d_bilist,
        d_bnumneigh,
        d_bfirstneigh,
        d_brcs,
        d_btypes,
        ntypes,
        d_type_map,
        umax_num_neigh_atoms,
        nghost);

    CHECK_CUDA_API( cudaDeviceSynchronize() );
    CHECK_CUDA_API( cudaGetLastError() );

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "cost time: " << duration.count() << " ms.\n";

    CHECK_CUDA_API( cudaMemcpy(h_betot_ptr, d_betot_ptr, sizeof(CoordType)*batch_size, cudaMemcpyDeviceToHost) );
    CHECK_CUDA_API( cudaMemcpy(h_bforce, d_bforce, sizeof(CoordType)*batch_size*(natoms_pad+nghost)*3, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_API( cudaFree(d_betot_ptr) );
    CHECK_CUDA_API( cudaFree(d_bforce) );
    CHECK_CUDA_API( cudaFree(d_cks) );
    CHECK_CUDA_API( cudaFree(d_dks) );
    CHECK_CUDA_API( cudaFree(d_binum) );
    CHECK_CUDA_API( cudaFree(d_bilist) );
    CHECK_CUDA_API( cudaFree(d_bnumneigh) );
    CHECK_CUDA_API( cudaFree(d_bfirstneigh) );
    CHECK_CUDA_API( cudaFree(d_brcs) );
    CHECK_CUDA_API( cudaFree(d_btypes) );
    CHECK_CUDA_API( cudaFree(d_type_map) );
}


};  // namespace : correction
};  // namespace : ai2pot

#endif
