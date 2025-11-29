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

#ifndef AI2POT_NEP_NEP_H
#define AI2POT_NEP_NEP_H


namespace ai2pot {
namespace nep {


template <typename CoordType>
class Nep {
public:
    static void find_ef(
        CoordType &etot,
        CoordType (*force)[3],
        int chebyshev_size,
        int n_radial_basis,
        int n_angular_basis,
        int l_max,
        CoordType *coeffs,
        CoordType *w0,
        CoordType *w1,
        CoordType *type_bias,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        CoordType (*rcs)[3],
        int *types,
        int ntypes,
        int *type_map,
        int umax_num_neigh_atoms,
        int nghost,
        CoordType rmax,
        CoordType rmin,
        CoordType *q_scaler);

};  // class : Nep


template <typename CoordType>
void Nep<CoordType>::find_ef(
    CoordType &etot,
    CoordType (*force)[3],
    int chebyshev_size,
    int n_radial_basis,
    int n_angular_basis,
    int l_max,
    CoordType *coeffs,
    CoordType *w0,
    CoordType *w1,
    CoordType *type_bias,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    CoordType (*rcs)[3],
    int *types,
    int ntypes,
    int *type_map,
    int umax_num_neigh_atoms,
    int nghost,
    CoordType rmax,
    CoordType rmin,
    CoordType *q_scaler)
{
    
}


};  // namespace : nep
};  // namespace : ai2pot



#endif