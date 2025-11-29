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

#ifndef AI2POT_NEP_NEP_UTILITIES_H
#define AI2POT_NEP_NEP_UTILITIES_H
#include <cassert>
#include <math.h>


#define L_MAX 4

namespace ai2pot {
namespace nep {

class NepIndex {
public:
    static int get_Clm_index(
        int l,
        int mp) {
        assert(l >= 1);
        assert(mp >= 0);

        return (l*l - 1) + mp;
    }


    static int get_Sinlm_index(
        int l_max,
        int mu,
        int l,
        int mp) {
        assert(l >= 1);
        assert(l <= l_max);
        assert(mp >= 0);

        return mu*(l_max*l_max + 2*l_max) + (l*l - 1) + mp;
    }


    static int get_qinl_index(
        int l_max,
        int mu,
        int l) {
        assert(l >= 1);
        assert(l <= l_max);
        
        return mu*l_max + (l-1);
    }
};  // class : NepIndex


template <typename CoordType>
class Blm {
public:
    inline static void find_blm_val(
        CoordType *blm_ptr,
        int l,
        int mp,
        CoordType *rc,
        CoordType distance_ij)
    {
        assert((l>=1) && (l<=L_MAX));
        assert((mp>=0) && (mp<(2*l+1)));

        switch(l) {
            case 1:
                switch(mp) {
                    case 0:
                        (*blm_ptr) = rc[2];
                        break;
                    case 1:
                        (*blm_ptr) = rc[0];
                        break;
                    case 2:
                        (*blm_ptr) = rc[1];
                        break;
                }
                break;
            case 2:
                switch(mp) {
                    case 0:
                        (*blm_ptr) = 3*std::pow(rc[2], 2) - std::pow(distance_ij, 2);
                        break;
                    case 1:
                        (*blm_ptr) = rc[0] * rc[2];
                        break;
                    case 2:
                        (*blm_ptr) = rc[1] * rc[2];
                        break;
                    case 3:
                        (*blm_ptr) = std::pow(rc[0], 2) - std::pow(rc[1], 2);
                        break;
                    case 4:
                        (*blm_ptr) = 2*rc[0]*rc[1];
                        break;
                }
                break;
            case 3:
                switch(mp) {
                    case 0:
                        (*blm_ptr) = (5*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2)) * rc[2];
                        break;
                    case 1:
                        (*blm_ptr) = (5*std::pow(rc[2], 2) - std::pow(distance_ij, 2)) * rc[0];
                        break;
                    case 2:
                        (*blm_ptr) = (5*std::pow(rc[2], 2) - std::pow(distance_ij, 2)) * rc[1];
                        break;
                    case 3:
                        (*blm_ptr) = (std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[2];
                        break;
                    case 4:
                        (*blm_ptr) = 2*rc[0]*rc[1]*rc[2];
                        break;
                    case 5:
                        (*blm_ptr) = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0];
                        break;
                    case 6:
                        (*blm_ptr) = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[1];
                        break;
                }
                break;
            case 4:
                switch(mp) {
                    case 0:
                        (*blm_ptr) = (35*std::pow(rc[2], 2) - 30*std::pow(distance_ij, 2))*std::pow(rc[2], 2)
                                     + 3*std::pow(distance_ij, 4);
                        break;
                    case 1:
                        (*blm_ptr) = (7*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2))*rc[0]*rc[2];
                        break;
                    case 2:
                        (*blm_ptr) = (7*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2))*rc[1]*rc[2];
                        break;
                    case 3:
                        (*blm_ptr) = (7*std::pow(rc[2], 2) - std::pow(distance_ij, 2)) 
                                     * (std::pow(rc[0], 2) - std::pow(rc[1], 2));
                        break;
                    case 4:
                        (*blm_ptr) = (7*std::pow(rc[2], 2) - std::pow(distance_ij, 2))
                                     * 2 * rc[0] * rc[1];
                        break;
                    case 5:
                        (*blm_ptr) = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0] * rc[2];
                        break;
                    case 6:
                        (*blm_ptr) = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[1] * rc[2];
                        break;
                    case 7:
                        (*blm_ptr) = std::pow(std::pow(rc[0], 2) - std::pow(rc[1], 2), 2)
                                     - 4*std::pow(rc[0], 2)*std::pow(rc[1], 2);
                        break;
                    case 8:
                        (*blm_ptr) = 4*(std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[0] * rc[1];
                        break;
                }
                break;
        }
    }

    inline static void find_blm_der2xyz(
        CoordType *blm_der2xyz,
        int l,
        int mp,
        CoordType *rc,
        CoordType distance_ij)
    {
        assert((l>=1) && (l<=L_MAX));
        assert((mp>=0) && (mp<(2*l+1)));

        switch(l) {
            case 1:
                switch(mp) {
                    case 0:
                        blm_der2xyz[0] = 0.0;
                        blm_der2xyz[1] = 0.0;
                        blm_der2xyz[2] = 1.0;
                        break;
                    case 1:
                        blm_der2xyz[0] = 1.0;
                        blm_der2xyz[1] = 0.0;
                        blm_der2xyz[2] = 0.0;
                        break;
                    case 2:
                        blm_der2xyz[0] = 0.0;
                        blm_der2xyz[1] = 1.0;
                        blm_der2xyz[2] = 0.0;
                        break;
                }
                break;
            case 2:
                switch(mp) {
                    case 0:
                        blm_der2xyz[0] = -2*rc[0];
                        blm_der2xyz[1] = -2*rc[1];
                        blm_der2xyz[2] = 4*rc[2];
                        break;
                    case 1:
                        blm_der2xyz[0] = rc[2];
                        blm_der2xyz[1] = 0.0;
                        blm_der2xyz[2] = rc[0];
                        break;
                    case 2:
                        blm_der2xyz[0] = 0.0;
                        blm_der2xyz[1] = rc[2];
                        blm_der2xyz[2] = rc[1];
                        break;
                    case 3:
                        blm_der2xyz[0] = 2*rc[0];
                        blm_der2xyz[1] = -2*rc[1];
                        blm_der2xyz[2] = 0.0;
                        break;
                    case 4:
                        blm_der2xyz[0] = 2*rc[1];
                        blm_der2xyz[1] = 2*rc[0];
                        blm_der2xyz[2] = 0.0;
                        break;
                }
                break;
            case 3:
                switch(mp) {
                    case 0:
                        blm_der2xyz[0] = -6*rc[0]*rc[2];
                        blm_der2xyz[1] = -6*rc[1]*rc[2];
                        blm_der2xyz[2] = 9*std::pow(rc[2], 2) - 3*std::pow(distance_ij, 2);
                        break;
                    case 1:
                        blm_der2xyz[0] = -2*std::pow(rc[0], 2) + 5*std::pow(rc[2], 2)
                                         - std::pow(distance_ij, 2);
                        blm_der2xyz[1] = -2*rc[0]*rc[1];
                        blm_der2xyz[2] = 8*rc[0]*rc[2];
                        break;
                    case 2:
                        blm_der2xyz[0] = -2*rc[0]*rc[1];
                        blm_der2xyz[1] = -2*std::pow(rc[1], 2) + 5*std::pow(rc[2], 2) 
                                         - std::pow(distance_ij, 2);
                        blm_der2xyz[2] = 8*rc[1]*rc[2];
                        break;
                    case 3:
                        blm_der2xyz[0] = 2*rc[0]*rc[2];
                        blm_der2xyz[1] = -2*rc[1]*rc[2];
                        blm_der2xyz[2] = std::pow(rc[0], 2) - std::pow(rc[1], 2);
                        break;
                    case 4:
                        blm_der2xyz[0] = 2*rc[1]*rc[2];
                        blm_der2xyz[1] = 2*rc[0]*rc[2];
                        blm_der2xyz[2] = 2*rc[0]*rc[1];
                        break;
                    case 5:
                        blm_der2xyz[0] = 3*std::pow(rc[0], 2) - 3*std::pow(rc[1], 2);
                        blm_der2xyz[1] = -6*rc[0]*rc[1];
                        blm_der2xyz[2] = 0.0;
                        break;
                    case 6:
                        blm_der2xyz[0] = 6*rc[0]*rc[1];
                        blm_der2xyz[1] = 3*std::pow(rc[0], 2) - 3*std::pow(rc[1], 2);
                        blm_der2xyz[2] = 0.0;
                        break;
                }
                break;
            case 4:
                switch(mp) {
                    case 0:
                        blm_der2xyz[0] = -60*rc[0]*std::pow(rc[2], 2) 
                                         + 12*std::pow(distance_ij, 2)*rc[0];
                        blm_der2xyz[1] = -60*rc[1]*std::pow(rc[2], 2) 
                                         + 12*std::pow(distance_ij, 2)*rc[1];
                        blm_der2xyz[2] = 80*std::pow(rc[2], 3) 
                                         - 48*std::pow(distance_ij, 2)*rc[2];
                        break;
                    case 1:
                        blm_der2xyz[0] = -6*std::pow(rc[0], 2)*rc[2] 
                                         + 7*std::pow(rc[2], 3) 
                                         - 3*std::pow(distance_ij, 2)*rc[2];
                        blm_der2xyz[1] = -6*rc[0]*rc[1]*rc[2];;
                        blm_der2xyz[2] = 15*rc[0]*std::pow(rc[2], 2) 
                                         - 3*std::pow(distance_ij, 2)*rc[0];
                        break;
                    case 2:
                        blm_der2xyz[0] = -6*rc[0]*rc[1]*rc[2];
                        blm_der2xyz[1] = -6*std::pow(rc[1], 2)*rc[2] 
                                         + 7*std::pow(rc[2], 3) 
                                         - 3*std::pow(distance_ij, 2)*rc[2];
                        blm_der2xyz[2] = 15*rc[1]*std::pow(rc[2], 2) 
                                         - 3*std::pow(distance_ij, 2)*rc[1];
                        break;
                    case 3:
                        blm_der2xyz[0] = -2*std::pow(rc[0], 3) 
                                         + 2*rc[0]*std::pow(rc[1], 2) 
                                         + 14*rc[0]*std::pow(rc[2], 2) 
                                         - 2*std::pow(distance_ij, 2)*rc[0];
                        blm_der2xyz[1] = -2*std::pow(rc[0], 2)*rc[1] 
                                         + 2*std::pow(rc[1], 3) 
                                         - 14*rc[1]*std::pow(rc[2], 2) 
                                         + 2*std::pow(distance_ij, 2)*rc[1];
                        blm_der2xyz[2] = 12*rc[2] * (std::pow(rc[0], 2) - std::pow(rc[1], 2));
                        break;
                    case 4:
                        blm_der2xyz[0] = -4*std::pow(rc[0], 2)*rc[1] 
                                         + 14*rc[1]*std::pow(rc[2], 2) 
                                         - 2*std::pow(distance_ij, 2)*rc[1];
                        blm_der2xyz[1] = -4*rc[0]*std::pow(rc[1], 2) 
                                         + 14*rc[0]*std::pow(rc[2], 2) 
                                         - 2*std::pow(distance_ij, 2)*rc[0];;
                        blm_der2xyz[2] = 24*rc[0]*rc[1]*rc[2];
                        break;
                    case 5:
                        blm_der2xyz[0] = 3*std::pow(rc[0], 2)*rc[2] 
                                         - 3*std::pow(rc[1], 2)*rc[2];
                        blm_der2xyz[1] = -6*rc[0]*rc[1]*rc[2];
                        blm_der2xyz[2] = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0];
                        break;
                    case 6:
                        blm_der2xyz[0] = 6*rc[0]*rc[1]*rc[2];
                        blm_der2xyz[1] = 3*std::pow(rc[0], 2)*rc[2] 
                                         - 3*std::pow(rc[1], 2)*rc[2];
                        blm_der2xyz[2] = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[1];
                        break;
                    case 7:
                        blm_der2xyz[0] = 4*std::pow(rc[0], 3) - 12*rc[0]*std::pow(rc[1], 2);
                        blm_der2xyz[1] = 4*std::pow(rc[1], 3) - 12*std::pow(rc[0], 2)*rc[1];
                        blm_der2xyz[2] = 0.0;
                        break;
                    case 8:
                        blm_der2xyz[0] = 12*std::pow(rc[0], 2)*rc[1] 
                                         - 4*std::pow(rc[1], 3);
                        blm_der2xyz[1] = 4*std::pow(rc[0], 3) 
                                         - 12*rc[0]*std::pow(rc[1], 2);
                        blm_der2xyz[2] = 0.0;
                        break;
                }
                break;
        }
    }
};  // class : Blm

};  // namespace : nep
};  // namespace : ai2pot


#endif
