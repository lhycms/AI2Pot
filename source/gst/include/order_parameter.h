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

#ifndef AI2POT_GST_ORDER_PARAMETER
#define AI2POT_GST_ORDER_PARAMETER

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "../../nblist/include/structure.h"
#include "../../nblist/include/neighborList.h"


namespace ai2pot {
namespace gst {

template <typename CoordType>
class OrderParameter {
public:
    static void find_order_parameters(CoordType* order_parameters,
                                      int inum,
                                      int* ilist,
                                      int* numneigh,
                                      int* firstneigh,
                                      CoordType* rcs,
                                      int* types,
                                      int nghost,
                                      int umax_num_neigh_atoms);
};  // class : OrderParameter


template <typename CoordType>
void OrderParameter<CoordType>::find_order_parameters(CoordType* order_parameters,
                                                      int inum,
                                                      int* ilist,
                                                      int* numneigh,
                                                      int* firstneigh,
                                                      CoordType* rcs,
                                                      int* types,
                                                      int nghost,
                                                      int umax_num_neigh_atoms)
{
    memset(order_parameters, 0, inum * sizeof(CoordType));
    CoordType* neigh_vec1;
    CoordType* neigh_vec2;
    CoordType* neigh_vec3;
    CoordType* neigh_vec4;
    for (int ii=0; ii<inum; ii++) {
        neigh_vec1 = &rcs[ii*umax_num_neigh_atoms*3 + 0*3];
        neigh_vec2 = &rcs[ii*umax_num_neigh_atoms*3 + 1*3];
        neigh_vec3 = &rcs[ii*umax_num_neigh_atoms*3 + 2*3];
        neigh_vec4 = &rcs[ii*umax_num_neigh_atoms*3 + 3*3];
        CoordType distance1 = std::sqrt( std::pow(neigh_vec1[0], 2) 
                                         + std::pow(neigh_vec1[1], 2)
                                         + std::pow(neigh_vec1[2], 2));
        CoordType distance2 = std::sqrt( std::pow(neigh_vec2[0], 2)
                                         + std::pow(neigh_vec2[1], 2)
                                         + std::pow(neigh_vec2[2], 2));
        CoordType distance3 = std::sqrt( std::pow(neigh_vec3[0], 2) 
                                         + std::pow(neigh_vec3[1], 2)
                                         + std::pow(neigh_vec3[2], 2));
        CoordType distance4 = std::sqrt( std::pow(neigh_vec4[0], 2)
                                         + std::pow(neigh_vec4[1], 2)
                                         + std::pow(neigh_vec4[2], 2) );

        order_parameters[ii] = (3 * distance4) / (distance1 + distance2 + distance3);
    }
}


};  // namespace : gst
};  // namespace : ai2pot

#endif