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

#ifndef AI2POT_MTPR_BASIS_H
#define AI2POT_MTPR_BASIS_H
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <stdio.h>
#include <iostream>
#include <cmath>

namespace ai2pot {
namespace mtpr {

template <typename CoordType>
class SwitchFunction {
public:
    SwitchFunction() {};

    SwitchFunction(CoordType rmax, CoordType rmin);

    CoordType val(CoordType distance_ij);

    CoordType der2r(CoordType distance_ij);

    CoordType der2r_der2r(CoordType distance_ij);

private:
    CoordType _rmax = 0;
    CoordType _rmin = 0;
};  // class : SwitchFunction


template <typename CoordType>
class RB_Chebyshev {
public:
    RB_Chebyshev();

    RB_Chebyshev(
        int size,
        CoordType rmax,
        CoordType rmin);

    RB_Chebyshev(const RB_Chebyshev& rhs);

    RB_Chebyshev(RB_Chebyshev&& rhs);

    RB_Chebyshev& operator=(const RB_Chebyshev& rhs);

    RB_Chebyshev& operator=(RB_Chebyshev&& rhs);

    void build(CoordType distance_ij);
        
    ~RB_Chebyshev();

    const int size() const;

    const CoordType rmax() const;

    const CoordType rmin() const;

    const CoordType* vals() const;

    const CoordType* ders2uu() const;
    
    const CoordType* ders2r() const;

    const CoordType* ders2uu_ders2uu() const;

    const CoordType* ders2r_ders2r() const;

    void show() const;

private:
    int _size = 0;
    CoordType _rmax = 0;
    CoordType _rmin = 0;
    CoordType* _vals = nullptr;
    CoordType* _ders2uu = nullptr;
    CoordType* _ders2r = nullptr;
    CoordType* _ders2uu_ders2uu = nullptr;
    CoordType* _ders2r_ders2r = nullptr;
};  // class : RB_ChebyShev


template <typename CoordType>
class RQ_Chebyshev {
public:
    RQ_Chebyshev();

    RQ_Chebyshev(
        int size,
        CoordType rmax,
        CoordType rmin);

    RQ_Chebyshev(const RQ_Chebyshev& rhs);

    RQ_Chebyshev(RQ_Chebyshev&& rhs);

    RQ_Chebyshev& operator=(const RQ_Chebyshev& rhs);

    RQ_Chebyshev& operator=(RQ_Chebyshev&& rhs);

    void build(CoordType distance_ij);

    ~RQ_Chebyshev();

    const int size() const;

    const CoordType rmax() const;

    const CoordType rmin() const;

    const CoordType* vals() const;

    const CoordType* ders2r() const;

    const CoordType* ders2r_ders2r() const;

    void show() const;

private:
    int _size = 0;
    CoordType _rmax = 0;
    CoordType _rmin = 0;
    SwitchFunction<CoordType> _switch_func;
    RB_Chebyshev<CoordType>* _rb_ptr = nullptr;
    CoordType* _vals = nullptr;
    CoordType* _ders2r = nullptr;
    CoordType* _ders2r_ders2r = nullptr;
};  // class : RQ_Chebyshev


template <typename CoordType>
SwitchFunction<CoordType>::SwitchFunction(CoordType rmax, CoordType rmin)
{
    this->_rmax = rmax;
    this->_rmin = rmin;
}

template <typename CoordType>
CoordType SwitchFunction<CoordType>::val(CoordType distance_ij)
{
    // ZBL: assert( (distance_ij>=this->_rmin) && (distance_ij<=this->_rmax) );
    CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);
    
    if (distance_ij < this->_rmin) {
        return 0;
    } else if ( (distance_ij>=this->_rmin) && (distance_ij<this->_rmax) ) {
        return std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1;
    } else {
        return 0;
    }
}

template <typename CoordType>
CoordType SwitchFunction<CoordType>::der2r(CoordType distance_ij)
{
    // ZBL: assert( (distance_ij>=this->_rmin) && (distance_ij<=this->_rmax) );
    CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);

    if (distance_ij < this->_rmin) {
        return 0;
    } else if ( (distance_ij>=this->_rmin) && (distance_ij<this->_rmax) ) {
        return 1 / (this->_rmax - this->_rmin) * (-30*std::pow(uu, 4) + 60*std::pow(uu, 3) - 30*std::pow(uu, 2));
    } else {
        return 0;
    }
}


template <typename CoordType>
CoordType SwitchFunction<CoordType>::der2r_der2r(CoordType distance_ij)
{
    CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);

    if (distance_ij < this->_rmin) {
        return 0;
    } else if ( (distance_ij>=this->_rmin) && (distance_ij<this->_rmax) ) {
        return 1 / std::pow(this->_rmax - this->_rmin, 2) * (-120 * std::pow(uu, 3) + 180*std::pow(uu, 2) - 60*uu);
    } else {
        return 0;
    }
}


template <typename CoordType>
RB_Chebyshev<CoordType>::RB_Chebyshev()
{
    this->_size = 0;
    this->_rmax = 0;
    this->_rmin = 0;
    this->_vals = nullptr;
    this->_ders2uu = nullptr;
    this->_ders2r = nullptr;
    this->_ders2uu_ders2uu = nullptr;
    this->_ders2r_ders2r = nullptr;
}

template <typename CoordType>
RB_Chebyshev<CoordType>::RB_Chebyshev(
    int size,
    CoordType rmax,
    CoordType rmin)
{
    this->_size = size;
    this->_rmax = rmax;
    this->_rmin = rmin;
    this->_vals = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    //memset(this->_vals, 0, sizeof(CoordType) * this->_size);
    this->_ders2uu = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    //memset(this->_ders2uu, 0, sizeof(CoordType) * this->_size);
    this->_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    //memset(this->_ders2r, 0, sizeof(CoordType) * this->_size);
    this->_ders2uu_ders2uu = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    //memset(this->_ders2uu_ders2uu, 0, sizeof(CoordType) * this->_size);
    this->_ders2r_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    //memset(this->_ders2r_ders2r, 0, sizeof(CoordType) * this->_size);
}

template <typename CoordType>
RB_Chebyshev<CoordType>::RB_Chebyshev(const RB_Chebyshev& rhs)
{
    this->_size = rhs._size;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    
    if (this->_size != 0) {
        this->_vals = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2uu = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2uu_ders2uu = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2r_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        for (int ii=0; ii<this->_size; ii++) {
            this->_vals[ii] = rhs._vals[ii];
            this->_ders2uu[ii] = rhs._ders2uu[ii];
            this->_ders2r[ii] = rhs._ders2r[ii];
            this->_ders2uu_ders2uu[ii] = rhs._ders2uu_ders2uu[ii];
            this->_ders2r_ders2r[ii] = rhs._ders2r_ders2r[ii];
        }
    }
}

template <typename CoordType>
RB_Chebyshev<CoordType>::RB_Chebyshev(RB_Chebyshev&& rhs)
{
    if (this != &rhs) {
        this->_size = rhs._size;
        this->_rmax = rhs._rmax;
        this->_rmin = rhs._rmin;
        this->_vals = rhs._vals;
        this->_ders2uu = rhs._ders2uu;
        this->_ders2r = rhs._ders2r;
        this->_ders2uu_ders2uu = rhs._ders2uu_ders2uu;
        this->_ders2r_ders2r = rhs._ders2r_ders2r;

        rhs._size = 0;
        rhs._rmax = 0.0;
        rhs._rmin = 0.0;
        rhs._vals = nullptr;
        rhs._ders2uu = nullptr;
        rhs._ders2r = nullptr;
        rhs._ders2uu_ders2uu = nullptr;
        rhs._ders2r_ders2r = nullptr;
    }
}

template <typename CoordType>
RB_Chebyshev<CoordType>& RB_Chebyshev<CoordType>::operator=(const RB_Chebyshev& rhs)
{
    if (this->_size != 0) {
        free(this->_vals);
        free(this->_ders2uu);
        free(this->_ders2r);
        free(this->_ders2uu_ders2uu);
        free(this->_ders2r_ders2r);
        this->_size = 0;
        this->_rmax = 0.0;
        this->_rmin = 0.0;
    }

    this->_size = rhs._size;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    if (this->_size != 0) {
        this->_vals = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2uu = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2uu_ders2uu = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2r_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);

        for (int ii=0; ii<this->_size; ii++) {
            this->_vals[ii] = rhs._vals[ii];
            this->_ders2uu[ii] = rhs._ders2uu[ii];
            this->_ders2r[ii] = rhs._ders2r[ii];
            this->_ders2uu_ders2uu[ii] = rhs._ders2uu_ders2uu[ii];
            this->_ders2r_ders2r[ii] = rhs._ders2r_ders2r[ii];
        }
    }
    return *this;
}

template <typename CoordType>
RB_Chebyshev<CoordType>& RB_Chebyshev<CoordType>::operator=(RB_Chebyshev&& rhs)
{
    if (this != &rhs) {
        if (this->_size != 0) {
            free(this->_vals);
            free(this->_ders2uu);
            free(this->_ders2r);
            free(this->_ders2uu_ders2uu);
            free(this->_ders2r_ders2r);
            this->_size = 0;
            this->_rmax = 0.0;
            this->_rmin = 0.0;
        }

        this->_size = rhs._size;
        this->_rmax = rhs._rmax;
        this->_rmin = rhs._rmin;
        this->_vals = rhs._vals;
        this->_ders2uu = rhs._ders2uu;
        this->_ders2r = rhs._ders2r;
        this->_ders2uu_ders2uu = rhs._ders2uu_ders2uu;
        this->_ders2r_ders2r = rhs._ders2r_ders2r;

        rhs._size = 0;
        rhs._rmax = 0.0;
        rhs._rmin = 0.0;
        rhs._vals = nullptr;
        rhs._ders2uu = nullptr;
        rhs._ders2r = nullptr;
        rhs._ders2uu_ders2uu = nullptr;
        rhs._ders2r_ders2r = nullptr;
    }

    return *this;
}

template <typename CoordType>
void RB_Chebyshev<CoordType>::build(CoordType distance_ij)
{   
    //assert( (distance_ij >= this->_rmin) && (distance_ij <= this->_rmax) );
    CoordType uu = (2*distance_ij - (this->_rmax + this->_rmin)) / (this->_rmax - this->_rmin);
    CoordType uu_coeff = 2 / (this->_rmax - this->_rmin);
    CoordType uu_coeff_sq = 4 / std::pow(this->_rmax - this->_rmin, 2);
    
    for (int ii=0; ii<this->_size; ii++) {
        if (ii == 0) {
            this->_vals[ii] = 1;
            this->_ders2uu[ii] = 0;
            this->_ders2r[ii] = this->_ders2uu[ii] * uu_coeff;
            this->_ders2uu_ders2uu[ii] = 0;
            this->_ders2r_ders2r[ii] = this->_ders2uu_ders2uu[ii] * uu_coeff_sq;
        } else if (ii == 1) {
            this->_vals[ii] = uu;
            this->_ders2uu[ii] = 1;
            this->_ders2r[ii] = this->_ders2uu[ii] * uu_coeff;
            this->_ders2uu_ders2uu[ii] = 0;
            this->_ders2r_ders2r[ii] = this->_ders2uu_ders2uu[ii] * uu_coeff_sq;
        } else {
            this->_vals[ii] = 2*uu*this->_vals[ii-1] - this->_vals[ii-2];
            this->_ders2uu[ii] = 2*this->_vals[ii-1] + 2*uu*this->_ders2uu[ii-1] - this->_ders2uu[ii-2];
            this->_ders2r[ii] = this->_ders2uu[ii] * uu_coeff;
            this->_ders2uu_ders2uu[ii] = 4*this->_ders2uu[ii-1] + 2*uu*this->_ders2uu_ders2uu[ii-1] - this->_ders2uu_ders2uu[ii-2];
            this->_ders2r_ders2r[ii] = this->_ders2uu_ders2uu[ii] * uu_coeff_sq;
        }
    }
}

template <typename CoordType>
RB_Chebyshev<CoordType>::~RB_Chebyshev()
{
    free(this->_vals);
    free(this->_ders2uu);
    free(this->_ders2r);
    free(this->_ders2uu_ders2uu);
    free(this->_ders2r_ders2r);
}

template <typename CoordType>
const int RB_Chebyshev<CoordType>::size() const
{
    return this->_size;
}

template <typename CoordType>
const CoordType RB_Chebyshev<CoordType>::rmax() const
{
    return this->_rmax;
}

template <typename CoordType>
const CoordType RB_Chebyshev<CoordType>::rmin() const
{
    return this->_rmin;
}

template <typename CoordType>
const CoordType* RB_Chebyshev<CoordType>::vals() const
{
    return this->_vals;
}

template <typename CoordType>
const CoordType* RB_Chebyshev<CoordType>::ders2uu() const
{
    return this->_ders2uu;
}

template <typename CoordType>
const CoordType* RB_Chebyshev<CoordType>::ders2r() const
{
    return this->_ders2r;
}

template <typename CoordType>
const CoordType* RB_Chebyshev<CoordType>::ders2uu_ders2uu() const
{
    return this->_ders2uu_ders2uu;
}

template <typename CoordType>
const CoordType* RB_Chebyshev<CoordType>::ders2r_ders2r() const
{
    return this->_ders2r_ders2r;
}

template <typename CoordType>
void RB_Chebyshev<CoordType>::show() const 
{
    printf("1. Radial basis vals :\n\t");
    for (int ii=0; ii<this->_size; ii++) 
        printf("%10lf, ", this->_vals[ii]);
    printf("\n");

    printf("2. Radial basis ders2uu : \n\t");
    for (int ii=0; ii<this->_size; ii++)
        printf("%10lf, ", this->_ders2uu[ii]);
    printf("\n");

    printf("3. Radial basis ders2r : \n\t");
    for (int ii=0; ii<this->_size; ii++)
        printf("%10lf, ", this->_ders2r[ii]);
    printf("\n");

    printf("4. Radial basis ders2uu_ders2uu : \n\t");
    for (int ii=0; ii<this->_size; ii++)
        printf("%10lf, ", this->_ders2uu_ders2uu[ii]);
    printf("\n");

    printf("5. Radial basis ders2r_ders2r : \n\t");
    for (int ii=0; ii<this->_size; ii++)
        printf("%10lf, ", this->_ders2r_ders2r[ii]);
    printf("\n");  
}

template <typename CoordType>
RQ_Chebyshev<CoordType>::RQ_Chebyshev()
{
    this->_size = 0;
    this->_rmax = 0;
    this->_rmin = 0;
    this->_rb_ptr = nullptr;
    this->_vals = nullptr;
    this->_ders2r = nullptr;
    this->_ders2r_ders2r = nullptr;
}

template <typename CoordType>
RQ_Chebyshev<CoordType>::RQ_Chebyshev(
    int size,
    CoordType rmax,
    CoordType rmin)
{
    this->_size = size;
    this->_rmax = rmax;
    this->_rmin = rmin;
    this->_switch_func = SwitchFunction<CoordType>(rmax, rmin);
    this->_rb_ptr = new RB_Chebyshev<CoordType>(size, rmax, rmin);
    this->_vals = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    //memset(this->_vals, 0, sizeof(CoordType) * this->_size);
    this->_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    //memset(this->_ders2r, 0, sizeof(CoordType) * this->_size);
    this->_ders2r_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
    //memset(this->_ders2r_ders2r, 0, sizeof(CoordType) * this->_size);
}

template <typename CoordType>
RQ_Chebyshev<CoordType>::RQ_Chebyshev(const RQ_Chebyshev& rhs)
{
    this->_size = rhs._size;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    this->_switch_func = rhs._switch_func;
    this->_rb_ptr = new RB_Chebyshev<CoordType>();
    *(this->_rb_ptr) = *(rhs._rb_ptr);
    if (this->_size != 0) {
        this->_vals = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2r_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        for (int ii=0; ii<this->_size; ii++) {
            this->_vals[ii] = rhs._vals[ii];
            this->_ders2r[ii] = rhs._ders2r[ii];
            this->_ders2r_ders2r[ii] = rhs._ders2r_ders2r[ii];
        }
    } else {
        this->_vals = nullptr;
        this->_ders2r = nullptr;
        this->_ders2r_ders2r = nullptr;
    }
}

template <typename CoordType>
RQ_Chebyshev<CoordType>::RQ_Chebyshev(RQ_Chebyshev&& rhs)
{
    if (this != &rhs) {
        this->_size = rhs._size;
        this->_rmax = rhs._rmax;
        this->_rmin = rhs._rmin;
        this->_switch_func = rhs._switch_func;
        this->_rb_ptr = rhs._rb_ptr;
        this->_vals = rhs._vals;
        this->_ders2r = rhs._ders2r;
        this->_ders2r_ders2r = rhs._ders2r_ders2r;

        rhs._size = 0;
        rhs._rmax = 0;
        rhs._rmin = 0;
        rhs._rb_ptr = nullptr;
        rhs._vals = nullptr;
        rhs._ders2r = nullptr;
        rhs._ders2r_ders2r = nullptr;
    }
}

template <typename CoordType>
RQ_Chebyshev<CoordType>& RQ_Chebyshev<CoordType>::operator=(const RQ_Chebyshev& rhs)
{
    if (this->_size != 0) {
        delete this->_rb_ptr;
        free(this->_vals);
        free(this->_ders2r);
        free(this->_ders2r_ders2r);
        this->_size = 0;
        this->_rmax = 0;
        this->_rmin = 0;
    }

    this->_size = rhs._size;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    this->_switch_func = rhs._switch_func;
    this->_rb_ptr = new RB_Chebyshev<CoordType>();
    *(this->_rb_ptr) = *(rhs._rb_ptr);
    if (rhs._size != 0) {
        this->_vals = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        this->_ders2r_ders2r = (CoordType*)malloc(sizeof(CoordType) * this->_size);
        for (int ii=0; ii<this->_size; ii++) {
            this->_vals[ii] = rhs._vals[ii];
            this->_ders2r[ii] = rhs._ders2r[ii];
            this->_ders2r_ders2r[ii] = rhs._ders2r_ders2r[ii];
        }
    } else {
        this->_vals = nullptr;
        this->_ders2r = nullptr;
        this->_ders2r_ders2r = nullptr;
    }

    return *this;
}

template <typename CoordType>
RQ_Chebyshev<CoordType>& RQ_Chebyshev<CoordType>::operator=(RQ_Chebyshev&& rhs)
{
    if (this != &rhs) {
        if (this->_size != 0) {
            delete this->_rb_ptr;
            free(this->_vals);
            free(this->_ders2r);
            free(this->_ders2r_ders2r);
            this->_size = 0;
            this->_rmax = 0;
            this->_rmin = 0;
        } 

        this->_size = rhs._size;
        this->_rmax = rhs._rmax;
        this->_rmin = rhs._rmin;
        this->_switch_func = rhs._switch_func;
        this->_rb_ptr = rhs._rb_ptr;
        this->_vals = rhs._vals;
        this->_ders2r = rhs._ders2r;
        this->_ders2r_ders2r = rhs._ders2r_ders2r;

        rhs._size = 0;
        rhs._rmax = 0;
        rhs._rmin = 0;
        rhs._rb_ptr = nullptr;
        rhs._vals = nullptr;
        rhs._ders2r = nullptr;
        rhs._ders2r_ders2r = nullptr;
    }

    return *this;
}

template <typename CoordType>
void RQ_Chebyshev<CoordType>::build(CoordType distance_ij)
{
    //assert( (distance_ij >= this->_rmin) && (distance_ij <= this->_rmax) );
    this->_rb_ptr->build(distance_ij);

    for (int ii=0; ii<this->_size; ii++) {
        this->_vals[ii] = this->_switch_func.val(distance_ij) * this->_rb_ptr->vals()[ii];
        this->_ders2r[ii] = (
            this->_switch_func.der2r(distance_ij) * this->_rb_ptr->vals()[ii] 
            + this->_switch_func.val(distance_ij) * this->_rb_ptr->ders2r()[ii]);
        this->_ders2r_ders2r[ii] = (
            this->_switch_func.der2r_der2r(distance_ij) * this->_rb_ptr->vals()[ii]
            + this->_switch_func.der2r(distance_ij) * this->_rb_ptr->ders2r()[ii]
            + this->_switch_func.der2r(distance_ij) * this->_rb_ptr->ders2r()[ii]
            + this->_switch_func.val(distance_ij) * this->_rb_ptr->ders2r_ders2r()[ii]);
    }
}

template <typename CoordType>
RQ_Chebyshev<CoordType>::~RQ_Chebyshev()
{
    delete this->_rb_ptr;
    free(this->_vals);
    free(this->_ders2r);
    free(this->_ders2r_ders2r);
}

template <typename CoordType>
const int RQ_Chebyshev<CoordType>::size() const 
{
    return this->_size;
}

template <typename CoordType>
const CoordType RQ_Chebyshev<CoordType>::rmax() const
{
    return this->_rmax;
}

template <typename CoordType>
const CoordType RQ_Chebyshev<CoordType>::rmin() const
{
    return this->_rmin;
}

template <typename CoordType>
const CoordType* RQ_Chebyshev<CoordType>::vals() const
{
    return this->_vals;
}

template <typename CoordType>
const CoordType* RQ_Chebyshev<CoordType>::ders2r() const
{
    return this->_ders2r;
}

template <typename CoordType>
const CoordType* RQ_Chebyshev<CoordType>::ders2r_ders2r() const
{
    return this->_ders2r_ders2r;
}

template <typename CoordType>
void RQ_Chebyshev<CoordType>::show() const
{
    printf("1. Radial Q vals:\n\t");
    for (int ii=0; ii<this->_size; ii++)
        printf("%10lf, ", this->_vals[ii]);
    printf("\n");
    printf("2. Radial Q ders2r:\n\t");
    for (int ii=0; ii<this->_size; ii++)
        printf("%10lf, ", this->_ders2r[ii]);
    printf("\n");
}


};  // namespace : mtpr
};  // namespace : ai2pot

#endif
