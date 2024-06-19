#ifndef AI2POT_ACE_RADIAL_BASIS_H
#define AI2POT_ACE_RADIAL_BASIS_H
#include <stdio.h>
#include <iostream>
#include <cstdlib>
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
class RB_CChebyshev {
public:
    RB_CChebyshev();

    RB_CChebyshev(int size,
                  CoordType rmax,
                  CoordType rmin,
                  CoordType lambda_val);

    //RB_CChebyshev(const RB_CChebyshev &rhs);

    //RB_CChebyshev(RB_CChebyshev &&rhs);

    //RB_CChebyshev& operator=(const RB_CChebyshev &rhs);

    //RB_CChebyshev& operator=(RB_CChebyshev &&rhs);

    void build(CoordType distance_ij);

    ~RB_CChebyshev();

    const int size() const;

    const CoordType rmax() const;

    const CoordType rmin() const;

    const CoordType lambda_val() const;

    const CoordType* vals() const;

    const CoordType* ders2vv() const;

    const CoordType* ders2r() const;

private:
    int _size = 0;
    CoordType _rmax = 0.0;
    CoordType _rmin = 0.0;
    CoordType _lambda_val = 0.0;
    CoordType *_vals = nullptr;
    CoordType *_ders2vv = nullptr;
    CoordType *_ders2r = nullptr;
};  // class : RB_CChebyshev





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


template <typename CoordType>
RB_CChebyshev<CoordType>::RB_CChebyshev() {
    this->_size = 0;
    this->_rmin = 0.0;
    this->_rmax = 0.0;
    this->_lambda_val = 0.0;
    this->_vals = nullptr;
    this->_ders2vv = nullptr;
    this->_ders2r = nullptr;
}

template <typename CoordType>
RB_CChebyshev<CoordType>::RB_CChebyshev(int size,
                                        CoordType rmax,
                                        CoordType rmin,
                                        CoordType lambda_val)
{
    this->_size = size;
    this->_rmin = rmin;
    this->_rmax = rmax;
    this->_lambda_val = lambda_val;
    this->_vals = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    memset(this->_vals, 0, this->_size * sizeof(CoordType));
    this->_ders2vv = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    memset(this->_ders2vv, 0, this->_size * sizeof(CoordType));
    this->_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    memset(this->_ders2r, 0, this->_size * sizeof(CoordType));
}

template <typename CoordType>
void RB_CChebyshev<CoordType>::build(CoordType distance_ij)
{
    CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);
    CoordType vv = 1 - 2 * ( exp( -this->_lambda_val*(uu-1) ) - 1 ) / (exp(this->_lambda_val) - 1);

    for (int ii=0; ii<this->_size; ii++) {
        if (ii==0) {
            this->_vals[ii] = 1;
            this->_ders2vv[ii] = 0;
            this->_ders2r[ii] = 0;
        } else if (ii==1) {
            this->_vals[ii] = vv;
            this->_ders2vv[ii] = 1;
            this->_ders2r[ii] = this->_ders2vv[ii]
                                * 2*this->_lambda_val/(this->_rmax-this->_rmin) 
                                * exp(-this->_lambda_val*((distance_ij-this->_rmin)/(this->_rmax-this->_rmin)-1)) 
                                / (exp(this->_lambda_val)-1);
        } else {
            this->_vals[ii] = 2 * vv * this->_vals[ii-1] - this->_vals[ii-2];
            this->_ders2vv[ii] = 2*this->_vals[ii-1] + 2*vv*this->_ders2vv[ii-1] - this->_ders2vv[ii-2];
            this->_ders2r[ii] = this->_ders2vv[ii]
                                * 2*this->_lambda_val/(this->_rmax-this->_rmin) 
                                * exp(-this->_lambda_val*((distance_ij-this->_rmin)/(this->_rmax-this->_rmin)-1)) 
                                / (exp(this->_lambda_val)-1);
        }
    }
}


template <typename CoordType>
RB_CChebyshev<CoordType>::~RB_CChebyshev() {
    free(this->_vals);
    free(this->_ders2vv);
    free(this->_ders2r);
}

template <typename CoordType>
const int RB_CChebyshev<CoordType>::size() const {
    return this->_size;
}

template <typename CoordType>
const CoordType RB_CChebyshev<CoordType>::rmax() const {
    return this->_rmax;
}

template <typename CoordType>
const CoordType RB_CChebyshev<CoordType>::rmin() const {
    return this->_rmin;
}

template <typename CoordType>
const CoordType RB_CChebyshev<CoordType>::lambda_val() const {
    return this->_lambda_val;
}

template <typename CoordType>
const CoordType* RB_CChebyshev<CoordType>::vals() const {
    return this->_vals;
}

template <typename CoordType>
const CoordType* RB_CChebyshev<CoordType>::ders2vv() const {
    return this->_ders2vv;
}

template <typename CoordType>
const CoordType* RB_CChebyshev<CoordType>::ders2r() const {
    return this->_ders2r;
}

};  // namespace : ace
};  // namespace : ai2pot

#endif

