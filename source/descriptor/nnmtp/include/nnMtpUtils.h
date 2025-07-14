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