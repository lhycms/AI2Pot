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
#include "./zbl_utilities.cuh"


namespace ai2pot {
namespace correction {

double K_C_SP = 14.399645; // 1/(4*PI*epsilon_0*e^2)

template <typename CoordType>
class PairZBL {
public:
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
                              CoordType distance_ij,
                              CoordType *ck,
                              CoordType *dk);
};



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

    for (int ii=0; ii<4; ii++)
        phi_val += ck[ii] * std::exp(-dk[ii] * x);
    
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
    CoordType pair_energy = K_C_SP * Zi * Zj / distance_ij
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
    CoordType half_pair_energy = 0.5 * this->find_pair_energy(distance_ij);
    atomic_energy += half_pair_energy;
}


template <typename CoordType>
__host__ __device__
void PairZBL<CoordType>::add_atomic_force_one(CoordType *atomic_force,
                                              int Zi,
                                              int Zj,
                                              CoordType rmax,
                                              CoordType rmin,
                                              CoordType distance_ij,
                                              CoordType *ck,
                                              CoordType *dk)
{
    
}


};  // namespace : correction
};  // namespace : ai2pot

#endif