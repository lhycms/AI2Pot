#ifndef AI2POT_ACE_DESCRIPTOR_H
#define AI2POT_ACE_DESCRIPTOR_H
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <string.h>
#include "../include/radialBasis.h"


namespace ai2pot {
namespace ace {

template <typename CoordType>
class Descriptor2B {
public:
    Descriptor2B();

    Descriptor2B(int size,
                 CoordType rmax,
                 CoordType rmin,
                 CoordType lambda_val);

    Descriptor2B(RQ_CChebyshev<CoordType> *ptr_rq);
    
    Descriptor2B(const Descriptor2B &rhs);

    Descriptor2B(Descriptor2B &&rhs);

    Descriptor2B<CoordType>& operator=(const Descriptor2B &rhs);

    Descriptor2B<CoordType>& operator=(Descriptor2B &&rhs);

    ~Descriptor2B();

    void find_val_der2r(CoordType *ptr_val, 
                        CoordType *ptr_der2coeffs,
                        CoordType *ptr_der2r, 
                        CoordType distance_ij,
                        CoordType *ptr_coeffs);
    
    const int chebyshev_size() const;

    const RQ_CChebyshev<CoordType>* ptr_rq() const;

private:
    int _chebyshev_size = 0;
    RQ_CChebyshev<CoordType> *_ptr_rq = nullptr;
};  // class : Descriptor2B




template <typename CoordType>
Descriptor2B<CoordType>::Descriptor2B() 
{
    this->_chebyshev_size = 0;
    this->_ptr_rq = nullptr;
}

template <typename CoordType>
Descriptor2B<CoordType>::Descriptor2B(int size,
                                      CoordType rmax,
                                      CoordType rmin,
                                      CoordType lambda_val)
{
    this->_chebyshev_size = size;
    this->_ptr_rq = new RQ_CChebyshev<CoordType>(size, rmax, rmin, lambda_val);
}

template <typename CoordType>
Descriptor2B<CoordType>::Descriptor2B(RQ_CChebyshev<CoordType> *ptr_rq)
{
    this->_chebyshev_size = ptr_rq->size();
    this->_ptr_rq = new RQ_CChebyshev<CoordType>(*ptr_rq);
}

template <typename CoordType>
Descriptor2B<CoordType>::Descriptor2B(const Descriptor2B &rhs) 
{
    this->_chebyshev_size = rhs._chebyshev_size;
    if (this->_chebyshev_size > 0)
        this->_ptr_rq = new RQ_CChebyshev<CoordType>(*rhs.ptr_rq());
    else
        this->_ptr_rq = nullptr;
}

template <typename CoordType>
Descriptor2B<CoordType>::Descriptor2B(Descriptor2B &&rhs)
{
    if (this != &rhs) {
        this->_chebyshev_size = rhs._chebyshev_size;
        rhs._chebyshev_size = 0;
        this->_ptr_rq = rhs._ptr_rq;
        rhs._ptr_rq = nullptr;
    }
}

template <typename CoordType>
Descriptor2B<CoordType>& Descriptor2B<CoordType>::operator=(const Descriptor2B &rhs)
{
    if (this->_chebyshev_size > 0) {
        this->_chebyshev_size = 0;
        delete this->_ptr_rq;
    }

    this->_chebyshev_size = rhs._chebyshev_size;
    if (this->_chebyshev_size > 0) 
        this->_ptr_rq = new RQ_CChebyshev<CoordType>(*rhs.ptr_rq());
    else
        this->_ptr_rq = nullptr;
}

template <typename CoordType>
Descriptor2B<CoordType>& Descriptor2B<CoordType>::operator=(Descriptor2B<CoordType> &&rhs)
{
    if (this != &rhs) {
        if (this->_chebyshev_size > 0) {
            this->_chebyshev_size = 0;
            delete this->_ptr_rq;
        }

        this->_chebyshev_size = rhs._chebyshev_size;
        rhs._chebyshev_size = 0;
        this->_ptr_rq = rhs._ptr_rq;
        rhs._ptr_rq = nullptr;
    }
}

template <typename CoordType>
Descriptor2B<CoordType>::~Descriptor2B()
{
    this->_chebyshev_size = 0;
    delete this->_ptr_rq;
}

template <typename CoordType>
void Descriptor2B<CoordType>::find_val_der2r(CoordType *ptr_val,
                                             CoordType *ptr_der2coeffs,
                                             CoordType *ptr_der2r,
                                             CoordType distance_ij,
                                             CoordType *ptr_coeffs)
{
    this->_ptr_rq->build(distance_ij);
    memset(ptr_val, 0.0, sizeof(CoordType));
    memset(ptr_der2coeffs, 0.0, this->_chebyshev_size * sizeof(CoordType));
    memset(ptr_der2r, 0.0, sizeof(CoordType));
    for (int ii=0; ii<this->_chebyshev_size; ii++) {
        *ptr_val += ptr_coeffs[ii] * this->_ptr_rq->vals()[ii];
        ptr_der2coeffs[ii] = this->_ptr_rq->vals()[ii];
        *ptr_der2r += ptr_coeffs[ii] * this->_ptr_rq->ders2r()[ii];
    }
}

template <typename CoordType>
const int Descriptor2B<CoordType>::chebyshev_size() const
{
    return this->_chebyshev_size;
}

template <typename CoordType>
const RQ_CChebyshev<CoordType>* Descriptor2B<CoordType>::ptr_rq() const
{
    return this->_ptr_rq;
}


};  // namespace : ace
};  // namespace : ai2pot

#endif