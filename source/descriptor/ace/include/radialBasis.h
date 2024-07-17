#ifndef AI2POT_ACE_RADIAL_BASIS_H
#define AI2POT_ACE_RADIAL_BASIS_H
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string.h>
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

    RB_CChebyshev(const RB_CChebyshev &rhs);

    RB_CChebyshev(RB_CChebyshev &&rhs);

    RB_CChebyshev& operator=(const RB_CChebyshev &rhs);

    RB_CChebyshev& operator=(RB_CChebyshev &&rhs);

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
class RQ_CChebyshev {
public:
    RQ_CChebyshev();

    RQ_CChebyshev(int size,
                  CoordType rmax,
                  CoordType rmin,
                  CoordType lambda_val);
    
    RQ_CChebyshev(const RQ_CChebyshev &rhs);

    RQ_CChebyshev(RQ_CChebyshev &&rhs);

    RQ_CChebyshev& operator=(const RQ_CChebyshev &rhs);

    RQ_CChebyshev& operator=(RQ_CChebyshev &&rhs);

    ~RQ_CChebyshev();

    void build(CoordType distance_ij);

    const int size() const;

    const CoordType rmax() const;

    const CoordType rmin() const;

    const CoordType lambda_val() const;

    const CoordType* vals() const;

    const CoordType* ders2r() const;


private:
    int _size = 0;
    CoordType _rmax = 0.0;
    CoordType _rmin = 0.0;
    CoordType _lambda_val = 0.0;
    RB_CChebyshev<CoordType>* _ptr_rb = nullptr;
    CoordType *_vals = nullptr;
    CoordType *_ders2r = nullptr;
};  // class : RQ_CChebyshev


template <typename CoordType>
class Gn {
public:
    Gn();

    Gn(int size,
       CoordType rmax,
       CoordType rmin,
       CoordType lambda_val);

    Gn(RQ_CChebyshev<CoordType> *ptr_rq);
    
    Gn(const Gn &rhs);

    Gn(Gn &&rhs);

    Gn<CoordType>& operator=(const Gn &rhs);

    Gn<CoordType>& operator=(Gn &&rhs);

    ~Gn();

    void build(CoordType distance_ij,
               CoordType *ptr_coeffs);     // located using (ntypes, ntypes, n^r_max / n^a_max)
    
    const int chebyshev_size() const;

    const RQ_CChebyshev<CoordType>* ptr_rq() const;

    const CoordType val() const;

    const CoordType* der2coeffs() const;

    const CoordType der2r() const;

    void show() const;

private:
    int _chebyshev_size = 0;
    RQ_CChebyshev<CoordType> *_ptr_rq = nullptr;
    CoordType _val = 0.0;
    CoordType *_der2coeffs = nullptr;
    CoordType _der2r = 0.0;
};  // class : Gn


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
    if (this->_size > 0) {
        this->_vals = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        memset(this->_vals, 0, this->_size * sizeof(CoordType));
        this->_ders2vv = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        memset(this->_ders2vv, 0, this->_size * sizeof(CoordType));
        this->_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        memset(this->_ders2r, 0, this->_size * sizeof(CoordType));
    }
}

template <typename CoordType>
RB_CChebyshev<CoordType>::RB_CChebyshev(const RB_CChebyshev &rhs)
{
    if ((rhs.size()!=0) && (rhs.rmax()>rhs.rmin())) {
        this->_size = rhs._size;
        this->_rmax = rhs._rmax;
        this->_rmin = rhs._rmin;
        this->_lambda_val = rhs._lambda_val;
        this->_vals = (CoordType*)malloc(this->_size * sizeof(CoordType));
        this->_ders2vv = (CoordType*)malloc(this->_size * sizeof(CoordType));
        this->_ders2r = (CoordType*)malloc(this->_size * sizeof(CoordType));
        for (int ii=0; ii<this->_size; ii++) {
            this->_vals[ii] = rhs._vals[ii];
            this->_ders2vv[ii] = rhs._ders2vv[ii];
            this->_ders2r[ii] = rhs._ders2r[ii];
        }
    } else {
        this->_size = rhs._size;
        this->_rmax = rhs._rmax;
        this->_rmin = rhs._rmin;
        this->_lambda_val = rhs._lambda_val;
        this->_vals = nullptr;
        this->_ders2vv = nullptr;
        this->_ders2r = nullptr;
    }
}

template <typename CoordType>
RB_CChebyshev<CoordType>::RB_CChebyshev(RB_CChebyshev &&rhs)
{
    if (this != &rhs) {
        this->_size = rhs._size;
        rhs._size = 0;
        this->_rmax = rhs._rmax;
        rhs._rmax = 0.0;
        this->_rmin = rhs._rmin;
        rhs._rmin = 0.0;
        this->_lambda_val = rhs._lambda_val;
        rhs._lambda_val = 0.0;
        this->_vals = rhs._vals;
        rhs._vals = nullptr;
        this->_ders2vv = rhs._ders2vv;
        rhs._ders2vv = nullptr;
        this->_ders2r = rhs._ders2r;
        rhs._ders2r = nullptr;
    }
}

template <typename CoordType>
RB_CChebyshev<CoordType>& RB_CChebyshev<CoordType>::operator=(const RB_CChebyshev<CoordType> &rhs)
{
    if ((rhs._size!=0) && (rhs._rmax>rhs._rmin)) {
        this->_size = rhs._size;
        this->_rmax = rhs._rmax;
        this->_rmin = rhs._rmin;
        this->_lambda_val = rhs._lambda_val;
        this->_vals = (CoordType*)malloc(this->_size * sizeof(CoordType));
        this->_ders2vv = (CoordType*)malloc(this->_size * sizeof(CoordType));
        this->_ders2r = (CoordType*)malloc(this->_size * sizeof(CoordType));
        for (int ii=0; ii<this->_size; ii++) {
            this->_vals[ii] = rhs._vals[ii];
            this->_ders2vv[ii] = rhs._ders2vv[ii];
            this->_ders2r[ii] = rhs._ders2r[ii];
        }
    } else {
        this->_size = rhs._size;
        this->_rmax = rhs._rmax;
        this->_rmin = rhs._rmin;
        this->_lambda_val = rhs._lambda_val;
        this->_vals = nullptr;
        this->_ders2vv = nullptr;
        this->_ders2r = nullptr;
    }
    return *this;
}

template <typename CoordType>
RB_CChebyshev<CoordType>& RB_CChebyshev<CoordType>::operator=(RB_CChebyshev<CoordType> &&rhs)
{
    if (this != &rhs) {
        if (this->_size > 0) {
            free(this->_vals);
            free(this->_ders2vv);
            free(this->_ders2r);
        }

        this->_size = rhs._size;
        rhs._size = 0;
        this->_rmax = rhs._rmax;
        rhs._rmax = 0.0;
        this->_rmin = rhs._rmin;
        rhs._rmin = 0.0;
        this->_lambda_val = rhs._lambda_val;
        rhs._lambda_val = 0.0;
        this->_vals = rhs._vals;
        rhs._vals = nullptr;
        this->_ders2vv = rhs._ders2vv;
        rhs._ders2vv = nullptr;
        this->_ders2r = rhs._ders2r;
        rhs._ders2r = nullptr;
    }
    return *this;
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
    if (this->_size != 0) {
        free(this->_vals);
        free(this->_ders2vv);
        free(this->_ders2r);
    }
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



template <typename CoordType>
RQ_CChebyshev<CoordType>::RQ_CChebyshev()
{
    this->_size = 0;
    this->_rmax = 0.0;
    this->_rmin = 0.0;
    this->_lambda_val = 0.0;
    this->_ptr_rb = nullptr;
    this->_vals = nullptr;
    this->_ders2r = nullptr;
}

template <typename CoordType>
RQ_CChebyshev<CoordType>::RQ_CChebyshev(int size, CoordType rmax, CoordType rmin, CoordType lambda_val)
{
    this->_size = size;
    this->_rmax = rmax;
    this->_rmin = rmin;
    this->_lambda_val = lambda_val;
    if (this->_size > 0) {
        this->_ptr_rb = new RB_CChebyshev<CoordType>(this->_size, this->_rmax, this->_rmin, this->_lambda_val);
        this->_vals = (CoordType*)malloc(this->_size * sizeof(CoordType));
        this->_ders2r = (CoordType*)malloc(this->_size * sizeof(CoordType));
    }
}

template <typename CoordType>
RQ_CChebyshev<CoordType>::RQ_CChebyshev(const RQ_CChebyshev &rhs)
{
    this->_size = rhs._size;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    this->_lambda_val = rhs._lambda_val;
    if ((this->_size>0) && (this->_rmax>this->_rmin)) {
        this->_ptr_rb = new RB_CChebyshev(*rhs._ptr_rb);
        this->_vals = (CoordType*)malloc(this->_size * sizeof(CoordType));
        this->_ders2r = (CoordType*)malloc(this->_size * sizeof(CoordType));
        for (int ii=0; ii<this->_size; ii++) {
            this->_vals[ii] = rhs._vals[ii];
            this->_ders2r[ii] = rhs._ders2r[ii];
        }
    } else {
        this->_ptr_rb = nullptr;
        this->_vals = nullptr;
        this->_ders2r = nullptr;
    }
}

template <typename CoordType>
RQ_CChebyshev<CoordType>::RQ_CChebyshev(RQ_CChebyshev &&rhs)
{
    if (this != &rhs) {
        this->_size = rhs._size;
        rhs._size = 0;
        this->_rmax = rhs._rmax;
        rhs._rmax = 0.0;
        this->_rmin = rhs._rmin;
        rhs._rmin = 0.0;
        this->_lambda_val = rhs._lambda_val;
        rhs._lambda_val = 0.0;
        this->_ptr_rb = rhs._ptr_rb;
        rhs._ptr_rb = nullptr;
        this->_vals = rhs._vals;
        rhs._vals = nullptr;
        this->_ders2r = rhs._ders2r;
        rhs._ders2r = nullptr;
    }
}

template <typename CoordType>
RQ_CChebyshev<CoordType>& RQ_CChebyshev<CoordType>::operator=(const RQ_CChebyshev &rhs)
{
    if (this->_size > 0) {
        delete this->_ptr_rb;
        free(this->_vals);
        free(this->_ders2r);
        this->_size = 0;
    }
    this->_size = rhs._size;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    this->_lambda_val = rhs._lambda_val;
    if (this->_size > 0) {
        this->_ptr_rb = new RB_CChebyshev(*rhs._ptr_rb);
        this->_vals = (CoordType*)malloc(this->_size * sizeof(CoordType));
        this->_ders2r = (CoordType*)malloc(this->_size * sizeof(CoordType));
        for (int ii=0; ii<this->_size; ii++) {
            this->_vals[ii] = rhs._vals[ii];
            this->_ders2r[ii] = rhs._ders2r[ii];
        }
    } else {
        this->_ptr_rb = nullptr;
        this->_vals = nullptr;
        this->_ders2r = nullptr;
    }
    return *this;
}

template <typename CoordType>
RQ_CChebyshev<CoordType>& RQ_CChebyshev<CoordType>::operator=(RQ_CChebyshev<CoordType> &&rhs)
{
    if (this->_size > 0) {
        delete this->_ptr_rb;
        free(this->_vals);
        free(this->_ders2r);
        this->_size = 0;
    }
    if (this != &rhs) {
        this->_size = rhs._size;
        rhs._size = 0;
        this->_rmax = rhs._rmax;
        rhs._rmax = 0.0;
        this->_rmin = rhs._rmin;
        rhs._rmin = 0.0;
        this->_lambda_val = rhs._lambda_val;
        rhs._lambda_val = 0.0;
        this->_ptr_rb = rhs._ptr_rb;
        rhs._ptr_rb = nullptr;
        this->_vals = rhs._vals;
        rhs._vals = nullptr;
        this->_ders2r = rhs._ders2r;
        rhs._ders2r = nullptr;
    }
    return *this;
}

template <typename CoordType>
RQ_CChebyshev<CoordType>::~RQ_CChebyshev()
{
    if (this->_size != 0) {
        delete this->_ptr_rb;
        free(this->_vals);
        free(this->_ders2r);
    }
}

template <typename CoordType>
void RQ_CChebyshev<CoordType>::build(CoordType distance_ij)
{
    this->_ptr_rb->build(distance_ij);
    SwitchFunction<CoordType> switch_func(this->_rmax, this->_rmin);
    for (int ii=0; ii<this->size(); ii++) {
        this->_vals[ii] = switch_func.val(distance_ij) * this->_ptr_rb->vals()[ii];
        this->_ders2r[ii] = switch_func.der2r(distance_ij) * this->_ptr_rb->vals()[ii] 
                            + switch_func.val(distance_ij) * this->_ptr_rb->ders2r()[ii];
    }
}

template <typename CoordType>
const int RQ_CChebyshev<CoordType>::size() const {
    return this->_size;
}

template <typename CoordType>
const CoordType RQ_CChebyshev<CoordType>::rmax() const {
    return this->_rmax;
}

template <typename CoordType>
const CoordType RQ_CChebyshev<CoordType>::rmin() const {
    return this->_rmin;
}

template <typename CoordType>
const CoordType RQ_CChebyshev<CoordType>::lambda_val() const {
    return this->_lambda_val;
}

template <typename CoordType>
const CoordType* RQ_CChebyshev<CoordType>::vals() const {
    return this->_vals;
}

template <typename CoordType>
const CoordType* RQ_CChebyshev<CoordType>::ders2r() const {
    return this->_ders2r;
}


template <typename CoordType>
Gn<CoordType>::Gn() 
{
    this->_chebyshev_size = 0;
    this->_ptr_rq = nullptr;
    this->_val = 0.0;
    this->_der2coeffs = nullptr;
    this->_der2r = 0.0;
}

template <typename CoordType>
Gn<CoordType>::Gn(int size,
                  CoordType rmax,
                  CoordType rmin,
                  CoordType lambda_val)
{
    this->_chebyshev_size = size;
    this->_ptr_rq = new RQ_CChebyshev<CoordType>(size, rmax, rmin, lambda_val);
    this->_val = 0.0;
    this->_der2coeffs = (CoordType*)malloc(this->_chebyshev_size * sizeof(CoordType));
    memset(this->_der2coeffs, 0, this->_chebyshev_size * sizeof(CoordType));
    this->_der2r = 0.0;
}

template <typename CoordType>
Gn<CoordType>::Gn(RQ_CChebyshev<CoordType> *ptr_rq)
{
    this->_chebyshev_size = ptr_rq->size();
    this->_ptr_rq = new RQ_CChebyshev<CoordType>(*ptr_rq);
    this->_val = 0.0;
    this->_der2coeffs = (CoordType*)malloc(this->_chebyshev_size * sizeof(CoordType));
    memset(this->_der2coeffs, 0, this->_chebyshev_size * sizeof(CoordType));
    this->_der2r = 0.0;
}

template <typename CoordType>
Gn<CoordType>::Gn(const Gn &rhs) 
{
    this->_chebyshev_size = rhs._chebyshev_size;
    if (this->_chebyshev_size > 0) {
        this->_ptr_rq = new RQ_CChebyshev<CoordType>(*rhs.ptr_rq());
        this->_val = rhs._val;
        this->_der2coeffs = (double*)malloc(this->_chebyshev_size * sizeof(CoordType));
        for (int ii=0; ii<this->_chebyshev_size; ii++)
            this->_der2coeffs[ii] = this->_der2coeffs[ii];
        this->_der2r = rhs._der2r;
    } else {
        this->_ptr_rq = nullptr;
        this->_val = 0.0;
        this->_der2coeffs = nullptr;
        this->_der2r = 0.0;
    }
}

template <typename CoordType>
Gn<CoordType>::Gn(Gn &&rhs)
{
    if (this != &rhs) {
        this->_chebyshev_size = rhs._chebyshev_size;
        rhs._chebyshev_size = 0;
        this->_ptr_rq = rhs._ptr_rq;
        rhs._ptr_rq = nullptr;
        this->_val = rhs._val;
        rhs._val = 0.0;
        this->_der2coeffs = rhs._der2coeffs;
        rhs._der2coeffs = nullptr;
        this->_der2r = rhs._der2r;
        rhs._der2r = 0.0;
    }
}

template <typename CoordType>
Gn<CoordType>& Gn<CoordType>::operator=(const Gn &rhs)
{
    if (this->_chebyshev_size > 0) {
        delete this->_ptr_rq;
        free(this->_der2coeffs);
        this->_chebyshev_size = 0;
        this->_val = 0.0;
        this->_der2r = 0.0;
    }

    this->_chebyshev_size = rhs._chebyshev_size;
    if (this->_chebyshev_size > 0) {
        this->_ptr_rq = new RQ_CChebyshev<CoordType>(*rhs.ptr_rq());
        this->_val = rhs._val;
        this->_der2coeffs = (CoordType*)malloc(this->_chebyshev_size * sizeof(CoordType));
        this->_der2r = rhs._der2r;
    }
    else {
        this->_ptr_rq = nullptr;
        this->_val = 0.0;
        this->_der2coeffs = nullptr;
        this->_der2r = 0.0;
    }
}

template <typename CoordType>
Gn<CoordType>& Gn<CoordType>::operator=(Gn<CoordType> &&rhs)
{
    if (this != &rhs) {
        if (this->_chebyshev_size > 0) {
            delete this->_ptr_rq;
            free(this->_der2coeffs);
            this->_chebyshev_size = 0;
            this->_val = 0.0;
            this->_der2r = 0.0;
        }

        this->_chebyshev_size = rhs._chebyshev_size;
        rhs._chebyshev_size = 0;
        this->_ptr_rq = rhs._ptr_rq;
        rhs._ptr_rq = nullptr;
        this->_val = rhs._val;
        rhs._val = 0.0;
        this->_der2coeffs = rhs._der2coeffs;
        rhs._der2coeffs = nullptr;
        this->_der2r = rhs._der2r;
        rhs._der2r;
    }
}

template <typename CoordType>
Gn<CoordType>::~Gn()
{
    delete this->_ptr_rq;
    free(this->_der2coeffs);
    this->_chebyshev_size = 0;
    this->_val = 0.0;
    this->_der2r = 0.0;
}

template <typename CoordType>
void Gn<CoordType>::build(CoordType distance_ij,
                          CoordType *ptr_coeffs)
{
    this->_ptr_rq->build(distance_ij);
    this->_val = 0.0;
    memset(this->_der2coeffs, 0, this->_chebyshev_size * sizeof(CoordType));
    this->_der2r = 0.0;
    for (int ii=0; ii<this->_chebyshev_size; ii++) {
        this->_val += ptr_coeffs[ii] * this->_ptr_rq->vals()[ii];
        this->_der2coeffs[ii] = this->_ptr_rq->vals()[ii];
        this->_der2r += ptr_coeffs[ii] * this->_ptr_rq->ders2r()[ii];
    }
}

template <typename CoordType>
const int Gn<CoordType>::chebyshev_size() const
{
    return this->_chebyshev_size;
}

template <typename CoordType>
const RQ_CChebyshev<CoordType>* Gn<CoordType>::ptr_rq() const
{
    return this->_ptr_rq;
}

template <typename CoordType>
const CoordType Gn<CoordType>::val() const {
    return this->_val;
}

template <typename CoordType>
const CoordType* Gn<CoordType>::der2coeffs() const {
    return this->_der2coeffs;
}

template <typename CoordType>
const CoordType Gn<CoordType>::der2r() const {
    return this->_der2r;
}

template <typename CoordType>
void Gn<CoordType>::show() const {
    printf("Gn information:\n");
    printf("\t1. chebyshev_size = %d\n", this->_chebyshev_size);
    printf("\t2. rmax = %g\n", this->_ptr_rq->rmax());
    printf("\t3. rmin = %g\n", this->_ptr_rq->rmin());
    printf("\t4. lambda_val = %g\n", this->_ptr_rq->lambda_val());
    printf("\t5. val = %g\n", this->_val);
    printf("\t6. der2coeffs = ");
    for (int ii=0; ii<this->_chebyshev_size; ii++)
        printf("%g, ", this->_der2coeffs[ii]);
    printf("\n");
    printf("\t7. der2r = %g\n", this->_der2r);
}

};  // namespace : ace
};  // namespace : ai2pot

#endif
