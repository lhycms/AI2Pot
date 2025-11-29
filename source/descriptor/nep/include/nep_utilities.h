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
    static void find_blm_val(
        CoordType *blm_ptr,
        int l,
        int mp,
        CoordType (*rc)[3])
    {

    }

    static void find_blm_der2xyz(
        CoordType (*blm_der2xyz)[3],
        int l,
        int mp,
        CoordType (*rc)[3])
    {
        
    }
};  // class : Blm

};  // namespace : nep
};  // namespace : ai2pot


#endif