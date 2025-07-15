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

#ifndef AI2POT_NNMTP_NNMTP_UTILS_H
#define AI2POT_NNMTP_NNMTP_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

namespace ai2pot {
namespace nnmtp {


template <typename CoordType>
class TanhActivationFunc {
public:
    static void find_val(
        CoordType &val, 
        CoordType hidden_val);

    static void find_der(
        CoordType &der,
        CoordType hidden_val);

    static void find_der2der(
        CoordType &der2der,
        CoordType hidden_val);

    static void find_all_info(
        CoordType &val,
        CoordType &der,
        CoordType &der2der,
        CoordType hidden_val);
};  // class : TanhActivationFunc



template <typename CoordType>
void TanhActivationFunc<CoordType>::find_val(
    CoordType &val,
    CoordType hidden_val)
{
    val = (std::exp(hidden_val) - std::exp(-hidden_val)) 
          / (std::exp(hidden_val) + std::exp(-hidden_val));
}


template <typename CoordType>
void TanhActivationFunc<CoordType>::find_der(
    CoordType &der,
    CoordType hidden_val)
{
    CoordType val = (std::exp(hidden_val) - std::exp(-hidden_val)) 
                    / (std::exp(hidden_val) + std::exp(-hidden_val));
    der = 1 - std::pow(val, 2);
}


template <typename CoordType>
void TanhActivationFunc<CoordType>::find_der2der(
    CoordType &der2der,
    CoordType hidden)
{

}


};  // namespace : nnmtp
};  // namespace : ai2pot

#endif