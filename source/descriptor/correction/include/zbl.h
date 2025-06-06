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

#ifndef AI2POT_ZBL_H
#define AI2POT_ZBL_H

#include <stdio.h>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <vector>


namespace ai2pot {
namespace correction {


double K_C_SP = 14.399645; // 1/(4*PI*epsilon_0*e^2)

template <typename CoordType>
class PairZBL {
public:
    PairZBL();

    PairZBL(int Zi,
        int Zj,
        CoordType rmax,
        CoordType rmin,
        CoordType *ck,
        CoordType *dk);

    PairZBL(const PairZBL &rhs);

    PairZBL(PairZBL &&rhs);

    PairZBL& operator=(const PairZBL &rhs);

    PairZBL& operator=(PairZBL &&rhs);

    ~PairZBL();

    CoordType find_switch_func(CoordType distance_ij);

    CoordType find_switch_func_der2rij(CoordType distance_ij);

    CoordType find_phi_func(CoordType distance_ij);

    CoordType find_phi_func_der2rij(CoordType distance_ij);

    CoordType find_pair_energy(CoordType distance_ij);

    void add_atomic_energy_one(CoordType &atomic_energy,
                               CoordType distance_ij);
    
    void add_atomic_force_one(CoordType *force,
                              CoordType *neigh_vec);

    void add_virial_one(CoordType *virial,
                        CoordType *neigh_vec);

private:
    int _Zi = 0;
    int _Zj = 0;
    CoordType _rmax = 0;
    CoordType _rmin = 0;
    CoordType *_ck = nullptr; // c1=0.18175, c2=0.50986, c3=0.28022, c4=0.02817
    CoordType *_dk = nullptr; // d1=3.1998,  d2=0.94229, d3=0.4029,  d4=0.20162
};  // class : PairZBL


template <typename CoordType>
class GroupZBL {
public:
    GroupZBL();

    GroupZBL(int ntypes,
             int *Zis,
             int *Zjs,
             CoordType rmax,
             CoordType rmin,
             CoordType *cks,
             CoordType *dks);

    GroupZBL(const GroupZBL& rhs);

    GroupZBL(GroupZBL&& rhs);

    GroupZBL& operator=(const GroupZBL& rhs);

    GroupZBL& operator=(GroupZBL&& rhs);

    ~GroupZBL();

    void correct_efv(CoordType &etot,
                     CoordType* atomic_forces,
                     CoordType* virial,
                     int inum,
                     int* ilist,
                     int* numneigh,
                     int* firstneigh,
                     CoordType (*rcs)[3],
                     int* types,
                     int ntypes,
                     int* type_map,
                     int umax_num_neigh_atoms,
                     int nghost);

    void correct_ef(CoordType &etot,
                    CoordType* atomic_forces,
                    int inum,
                    int* ilist,
                    int* numneigh,
                    int* firstneigh,
                    CoordType (*rcs)[3],
                    int* types,
                    int ntypes,
                    int* type_map,
                    int umax_num_neigh_atoms,
                    int nghost);


private:
    int _ntypes = 0;
    CoordType _rmax = 0.0;
    CoordType _rmin = 0.0;
    std::vector<PairZBL<CoordType>> _pair_zbl_vector;
};  // class : GroupZBL





template <typename CoordType>
PairZBL<CoordType>::PairZBL(int Zi,
                    int Zj,
                    CoordType rmax,
                    CoordType rmin,
                    CoordType *ck,
                    CoordType *dk)
{
    this->_Zi = Zi;
    this->_Zj = Zj;
    this->_rmax = rmax;
    this->_rmin = rmin;
    this->_ck = (CoordType*)malloc(sizeof(CoordType) * 4);
    this->_dk = (CoordType*)malloc(sizeof(CoordType) * 4);
    for (int ii=0; ii<4; ii++) {
        this->_ck[ii] = ck[ii];
        this->_dk[ii] = dk[ii];
    }
}


template <typename CoordType>
PairZBL<CoordType>::PairZBL(const PairZBL &rhs)
{
    this->_Zi = rhs._Zi;
    this->_Zj = rhs._Zj;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    this->_ck = (CoordType*)malloc(sizeof(CoordType) * 4);
    this->_dk = (CoordType*)malloc(sizeof(CoordType) * 4);
    for (int ii=0; ii<4; ii++) {
        this->_ck[ii] = rhs._ck[ii];
        this->_dk[ii] = rhs._dk[ii];
    }
}


template <typename CoordType>
PairZBL<CoordType>::PairZBL(PairZBL &&rhs) {
    this->_Zi = rhs._Zi;
    rhs._Zi = 0;
    this->_Zj = rhs._Zj;
    rhs._Zj = 0;
    this->_rmax = rhs._rmax;
    rhs._rmax = 0;
    this->_rmin = rhs._rmin;
    rhs._rmin = 0;

    this->_ck = rhs._ck;
    rhs._ck = nullptr;
    this->_dk = rhs._dk;
    rhs._dk = nullptr;
}


template <typename CoordType>
PairZBL<CoordType>& PairZBL<CoordType>::operator=(const PairZBL &rhs) {
    this->_Zi = rhs._Zi;
    this->_Zj = rhs._Zj;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    for (int ii=0; ii<4; ii++) {
        this->_ck[ii] = rhs._ck[ii];
        this->_dk[ii] = rhs._dk[ii];
    }

    return *this;
}


template <typename CoordType>
PairZBL<CoordType>& PairZBL<CoordType>::operator=(PairZBL &&rhs) {
    this->_Zi = rhs._Zi;
    rhs._Zi = 0;
    this->_Zj = rhs._Zj;
    rhs._Zj = 0;
    this->_rmax = rhs._rmax;
    rhs._rmax = 0;
    this->_rmin = rhs._rmin;
    rhs._rmin = 0;

    this->_ck = rhs._ck;
    rhs._ck = nullptr;
    this->_dk = rhs._dk;
    rhs._dk = nullptr;
    
    return *this;
}


template <typename CoordType>
PairZBL<CoordType>::~PairZBL() {
    this->_Zi = 0;
    this->_Zj = 0;
    this->_rmax = 0;
    this->_rmin = 0;
    free(this->_ck);
    free(this->_dk);
}


template <typename CoordType>
CoordType PairZBL<CoordType>::find_switch_func(CoordType distance_ij)
{
    if (distance_ij < this->_rmin)
        return 1.0;
    else if ((distance_ij>=this->_rmin) && (distance_ij<=this->_rmax)) {
        CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);
        return std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1;
    }
    else
        return 0.0;
}


template <typename CoordType>
CoordType PairZBL<CoordType>::find_switch_func_der2rij(CoordType distance_ij)
{
    if (distance_ij < this->_rmin)
        return 0.0;
    else if ((distance_ij>=this->_rmin) && (distance_ij<=this->_rmax)) {
        CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);
        return 1/(this->_rmax - this->_rmin) * (-30*std::pow(uu, 4) + 60*std::pow(uu, 3) - 30*std::pow(uu, 2));
    }
    else
        return 0.0;
}


template <typename CoordType>
CoordType PairZBL<CoordType>::find_phi_func(CoordType distance_ij)
{
    CoordType phi_val = 0.0;
    CoordType a = 0.46848 / (std::pow(this->_Zi, 0.23) + std::pow(this->_Zj, 0.23));
    CoordType x = distance_ij / a;

    for (int ii=0; ii<4; ii++)
        phi_val += this->_ck[ii] * std::exp(-this->_dk[ii] * x);

    return phi_val;
}


template <typename CoordType>
CoordType PairZBL<CoordType>::find_phi_func_der2rij(CoordType distance_ij)
{
    CoordType phi_der2rij = 0.0;
    CoordType a = 0.46848 / (std::pow(this->_Zi, 0.23) + std::pow(this->_Zj, 0.23));

    for (int ii=0; ii<4; ii++)
        phi_der2rij += - this->_ck[ii] * this->_dk[ii] / a 
                         * std::exp(-this->_dk[ii] * distance_ij / a);

    return phi_der2rij;
}


template <typename CoordType>
CoordType PairZBL<CoordType>::find_pair_energy(CoordType distance_ij)
{
    CoordType phi_val = this->find_phi_func(distance_ij);
    CoordType switch_val = this->find_switch_func(distance_ij);
    CoordType pair_energy = K_C_SP * (this->_Zi*this->_Zj) / distance_ij 
                            * phi_val 
                            * switch_val;
    return pair_energy;
}


template <typename CoordType>
void PairZBL<CoordType>::add_atomic_energy_one(CoordType &atomic_energy,
                                           CoordType distance_ij)
{
    CoordType half_pair_energy = 0.5 * this->find_pair_energy(distance_ij);
    atomic_energy += half_pair_energy;
}


template <typename CoordType>
void PairZBL<CoordType>::add_atomic_force_one(CoordType *atomic_force,
                                          CoordType *neigh_vec)
{
    CoordType distance_ij = std::sqrt( std::pow(neigh_vec[0], 2) 
                                       + std::pow(neigh_vec[1], 2)
                                       + std::pow(neigh_vec[2], 2) );
    CoordType A = K_C_SP*this->_Zi*this->_Zj/distance_ij;
    CoordType B = this->find_phi_func(distance_ij);
    CoordType C = this->find_switch_func(distance_ij);
    
    CoordType A_der = -K_C_SP*this->_Zi*this->_Zj/std::pow(distance_ij, 2);
    CoordType B_der = this->find_phi_func_der2rij(distance_ij);
    CoordType C_der = this->find_switch_func_der2rij(distance_ij);

    for (int aa=0; aa<3; aa++) {
        atomic_force[aa] += (A_der*B*C 
                             + A*B_der*C
                             + A*B*C_der) * neigh_vec[aa] / distance_ij;
    }
}


template <typename CoordType>
void PairZBL<CoordType>::add_virial_one(CoordType *virial,
                                    CoordType *neigh_vec)
{
    CoordType distance_ij = std::sqrt( std::pow(neigh_vec[0], 2) 
                                       + std::pow(neigh_vec[1], 2)
                                       + std::pow(neigh_vec[2], 2) );
    CoordType A = K_C_SP*this->_Zi*this->_Zj/distance_ij;
    CoordType B = this->find_phi_func(distance_ij);
    CoordType C = this->find_switch_func(distance_ij);
    
    CoordType A_der = -K_C_SP*this->_Zi*this->_Zj/std::pow(distance_ij, 2);
    CoordType B_der = this->find_phi_func_der2rij(distance_ij);
    CoordType C_der = this->find_switch_func_der2rij(distance_ij);

    for (int aa=0; aa<3; aa++) {
        for (int bb=0; bb<3; bb++) {
            CoordType Fijb = (A_der*B*C
                             + A*B_der*C
                             + A*B*C_der) * neigh_vec[bb] / distance_ij;
            virial[aa*3 + bb] += -0.5 * neigh_vec[aa] * Fijb;
        }
    }
}


template <typename CoordType>
GroupZBL<CoordType>::GroupZBL() {}


template <typename CoordType>
GroupZBL<CoordType>::GroupZBL(int ntypes,
                              int* Zis,
                              int* Zjs,
                              CoordType rmax,
                              CoordType rmin,
                              CoordType* cks,
                              CoordType* dks)
{
    this->_ntypes = ntypes;
    this->_rmax = rmax;
    this->_rmin = rmin;
    for (int ii=0; ii<this->_ntypes; ii++) {
        for (int jj=0; jj<this->_ntypes; jj++) {
            int idx = ii * this->_ntypes + jj;
            this->_pair_zbl_vector.push_back( PairZBL<CoordType>(Zis[ii],
                                                                 Zjs[jj],
                                                                 this->_rmax,
                                                                 this->_rmin,
                                                                 &cks[idx*4],
                                                                 &dks[idx*4]));
        }
    }
}


template <typename CoordType>
GroupZBL<CoordType>::GroupZBL(const GroupZBL& rhs) {
    this->_ntypes = rhs._ntypes;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;

    this->_pair_zbl_vector.resize(this->_ntypes * this->_ntypes);
    for (int ii=0; ii<this->_ntypes*this->_ntypes; ii++)
        this->_pair_zbl_vector[ii] = rhs._pair_zbl_vector[ii];
}


template <typename CoordType>
GroupZBL<CoordType>::GroupZBL(GroupZBL&& rhs) {
    this->_ntypes = rhs._ntypes;
    rhs._ntypes = 0;
    this->_rmax = rhs._rmax;
    rhs._rmax = 0.0;
    this->_rmin = rhs._rmin;
    rhs._rmin = 0.0;

    this->_pair_zbl_vector = std::move(rhs._pair_zbl_vector);
}


template <typename CoordType>
GroupZBL<CoordType>& GroupZBL<CoordType>::operator=(const GroupZBL& rhs) {
    this->_ntypes = rhs._ntypes;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;

    this->_pair_zbl_vector.clear();
    this->_pair_zbl_vector.resize(this->_ntypes * this->_ntypes);
    for (int ii=0; ii<this->_ntypes*this->_ntypes; ii++)
        this->_pair_zbl_vector[ii] = rhs._pair_zbl_vector[ii];

    return *this;
}


template <typename CoordType>
GroupZBL<CoordType>& GroupZBL<CoordType>::operator=(GroupZBL&& rhs) {
    this->_ntypes = rhs._ntypes;
    rhs._ntypes = 0;
    this->_rmax = rhs._rmax;
    rhs._rmax = 0.0;
    this->_rmin = rhs._rmin;
    rhs._rmin = 0.0;

    this->_pair_zbl_vector.clear();
    this->_pair_zbl_vector = std::move(rhs._pair_zbl_vector);

    return *this;
}


template <typename CoordType>
GroupZBL<CoordType>::~GroupZBL() {
    this->_ntypes = 0;
    this->_rmax = 0.0;
    this->_rmin = 0.0;
}


template <typename CoordType>
void GroupZBL<CoordType>::correct_efv(CoordType &etot,
                                      CoordType* atomic_forces,
                                      CoordType* virial,
                                      int inum,
                                      int* ilist,
                                      int* numneigh,
                                      int* firstneigh,
                                      CoordType (*rcs)[3],
                                      int* types,
                                      int ntypes,
                                      int* type_map,
                                      int umax_num_neigh_atoms,
                                      int nghost)
{
    assert(this->_ntypes == ntypes);
    int center_idx;
    int neigh_idx;
    int type_inner;
    int type_outer;
    int Zi;
    int Zj;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    for (int ii=0; ii<inum; ii++) {
        center_idx = ilist[ii];
        type_inner = types[center_idx];
        Zi = type_map[type_inner];

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            Zj = type_map[type_outer];
            int zbl_idx = type_inner * this->_ntypes + type_outer;
            PairZBL<CoordType> pair_zbl = this->_pair_zbl_vector[zbl_idx];
            
            for (int aa=0; aa<3; aa++)
                neigh_vec[aa] = rcs[ii*umax_num_neigh_atoms + jj][aa];
            
            distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                     + std::pow(neigh_vec[1], 2)
                                     + std::pow(neigh_vec[2], 2) );
            distance_ij_inv = 1.0 / distance_ij;
            if ( distance_ij>this->_rmax )
                continue;

            pair_zbl.add_atomic_energy_one(etot, distance_ij);
            pair_zbl.add_atomic_force_one(&atomic_forces[ii*3+0], neigh_vec);
            pair_zbl.add_virial_one(virial, neigh_vec);
        }
    }
}



template <typename CoordType>
void GroupZBL<CoordType>::correct_ef(CoordType &etot,
                                    CoordType* atomic_forces,
                                    int inum,
                                    int* ilist,
                                    int* numneigh,
                                    int* firstneigh,
                                    CoordType (*rcs)[3],
                                    int* types,
                                    int ntypes,
                                    int* type_map,
                                    int umax_num_neigh_atoms,
                                    int nghost)
{
    assert(this->_ntypes == ntypes);
    int center_idx;
    int neigh_idx;
    int type_inner;
    int type_outer;
    int Zi;
    int Zj;
    CoordType neigh_vec[3];
    CoordType distance_ij;
    CoordType distance_ij_inv;

    for (int ii=0; ii<inum; ii++) {
        center_idx = ilist[ii];
        type_inner = types[center_idx];
        Zi = type_map[type_inner];

        for (int jj=0; jj<numneigh[ii]; jj++) {
            neigh_idx = firstneigh[ii*umax_num_neigh_atoms + jj];
            type_outer = types[neigh_idx];
            Zj = type_map[type_outer];
            int zbl_idx = type_inner * this->_ntypes + type_outer;
            PairZBL<CoordType> pair_zbl = this->_pair_zbl_vector[zbl_idx];
            
            for (int aa=0; aa<3; aa++)
                neigh_vec[aa] = rcs[ii*umax_num_neigh_atoms + jj][aa];
            
            distance_ij = std::sqrt( std::pow(neigh_vec[0], 2)
                                     + std::pow(neigh_vec[1], 2)
                                     + std::pow(neigh_vec[2], 2) );
            distance_ij_inv = 1.0 / distance_ij;
            if ( distance_ij>this->_rmax )
                continue;

            pair_zbl.add_atomic_energy_one(etot, distance_ij);
            pair_zbl.add_atomic_force_one(&atomic_forces[ii*3+0], neigh_vec);
        }
    }
}


};  // namespace : correction
};  // namespace : ai2pot

#endif
