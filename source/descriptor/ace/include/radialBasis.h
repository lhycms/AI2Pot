#ifndef AI2POT_ACE_RADIAL_BASIS_H
#define AI2POT_ACE_RADIAL_BASIS_H
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cassert>

namespace ai2pot {
namespace ace {

template <typename CoordType>
class SwitchFunction {
public:
    SwitchFunction();

    SwitchFunction(CoordType rmax, CoordType rmin);

    CoordType val(CoordType distance_ij);

    CoordType der2r(CoordType distance_ij);

    CoordType rmax() const;

    CoordType rmin() const;

private:
    CoordType _rmax = 0;
    CoordType _rmin = 0;
};  // class : SwitchFunction



template <typename CoordType>
SwitchFunction<CoordType>::SwitchFunction() {
    this->_rmax = 0;
    this->_rmin = 0;
}

template <typename CoordType>
SwitchFunction<CoordType>::SwitchFunction(CoordType rmax, CoordType rmin) {
    this->_rmax = rmax;
    this->_rmin = rmin;
}

template <typename CoordType>
CoordType SwitchFunction<CoordType>::val(CoordType distance_ij) {
    // ZBL: assert(distance_ij>=this->_rmin && distance_ij<=this->_rmax);
    CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);
    
    if (distance_ij < this->_rmin) {
        return 0;
    } else if ( (distance_ij>=this->_rmin) && (distance_ij<=this->_rmax)) {
        return std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1;
    } else {
        return 0;
    }
}

template <typename CoordType>
CoordType SwitchFunction<CoordType>::der2r(CoordType distance_ij) {
    // ZBL: assert(distance_ij>=this->_rmin && distance_ij<=this->_rmax);
    CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);

    if (distance_ij < this->_rmin) {
        return 0;
    } else if ( (distance_ij>=this->_rmin) && (distance_ij<=this->_rmax) ) {
        return 1 / (this->_rmax - this->_rmin) * ( -30*std::pow(uu, 4) + 60*std::pow(uu, 3) - 30*std::pow(uu, 2) );
    } else {
        return 0;
    }
}

template <typename CoordType>
CoordType SwitchFunction<CoordType>::rmin() const {
    return this->_rmin;
}

template <typename CoordType>
CoordType SwitchFunction<CoordType>::rmax() const {
    return this->_rmax;
}


};  // namespace : ace
};  // namespace : ai2pot

#endif

