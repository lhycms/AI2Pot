#ifndef AI2POT_ACE_ANGULAR_UTILS_H
#define AI2POT_ACE_ANGULAR_UTILS_H
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <cassert>
#include "./radialBasis.h"

namespace ai2pot {
namespace ace {

template <typename CoordType>
void b00(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = 1;
    der2xyz[0] = 0.0;
    der2xyz[1] = 0.0;
    der2xyz[2] = 0.0;
}

template <typename CoordType>
void b10(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = rc[2];
    der2xyz[0] = 0.0;
    der2xyz[1] = 0.0;
    der2xyz[2] = 1.0;
}

template <typename CoordType>
void b11(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = rc[0];
    der2xyz[0] = 1.0;
    der2xyz[1] = 0.0;
    der2xyz[2] = 0.0;
}

template <typename CoordType>
void b12(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = rc[1];
    der2xyz[0] = 0.0;
    der2xyz[1] = 1.0;
    der2xyz[2] = 0.0;
}

template <typename CoordType>
void b20(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2)+std::pow(rc[1], 2)+std::pow(rc[2], 2);
    val = 3*std::pow(rc[2], 2) - r2;
    der2xyz[0] = -2*rc[0];
    der2xyz[1] = -2*rc[1];
    der2xyz[2] = 4*rc[2];
}

template <typename CoordType>
void b21(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = rc[0] * rc[2];
    der2xyz[0] = rc[2];
    der2xyz[1] = 0.0;
    der2xyz[2] = rc[0];
}

template <typename CoordType>
void b22(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = rc[1] * rc[2];
    der2xyz[0] = 0.0;
    der2xyz[1] = rc[2];
    der2xyz[2] = rc[1];
}

template <typename CoordType>
void b23(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = std::pow(rc[0], 2) - std::pow(rc[1], 2);
    der2xyz[0] = 2*rc[0];
    der2xyz[1] = -2*rc[1];
    der2xyz[2] = 0.0;
}

template <typename CoordType>
void b24(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = 2*rc[0]*rc[1];
    der2xyz[0] = 2*rc[1];
    der2xyz[1] = 2*rc[0];
    der2xyz[2] = 0.0;
}

template <typename CoordType>
void b30(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2)+std::pow(rc[1], 2)+std::pow(rc[2], 2);
    val = (5*std::pow(rc[2], 2) - 3*r2) * rc[2];
    der2xyz[0] = -6*rc[0]*rc[2];
    der2xyz[1] = -6*rc[1]*rc[2];
    der2xyz[2] = 9*std::pow(rc[2], 2) - 3*r2;
}

template <typename CoordType>
void b31(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2)+std::pow(rc[1], 2)+std::pow(rc[2], 2);
    val = (5*std::pow(rc[2], 2) - r2) * rc[0];
    der2xyz[0] = -2*std::pow(rc[0], 2) + 5*std::pow(rc[2], 2) - r2;
    der2xyz[1] = -2*rc[0]*rc[1];
    der2xyz[2] = 8*rc[0]*rc[2];
}

template <typename CoordType>
void b32(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2)+std::pow(rc[1],2)+std::pow(rc[2], 2);
    val = (5*std::pow(rc[2], 2) - r2) * rc[1];
    der2xyz[0] = -2*rc[0]*rc[1];
    der2xyz[1] = -2*std::pow(rc[1], 2) + 5*std::pow(rc[2], 2) - r2;
    der2xyz[2] = 8*rc[1]*rc[2];
}

template <typename CoordType>
void b33(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = (std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[2];
    der2xyz[0] = 2*rc[0]*rc[2];
    der2xyz[1] = -2*rc[1]*rc[2];
    der2xyz[2] = std::pow(rc[0], 2) - std::pow(rc[1], 2);
}

template <typename CoordType>
void b34(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = 2*rc[0]*rc[1]*rc[2];
    der2xyz[0] = 2*rc[1]*rc[2];
    der2xyz[1] = 2*rc[0]*rc[2];
    der2xyz[2] = 2*rc[0]*rc[1];
}

template <typename CoordType>
void b35(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0];
    der2xyz[0] = 3*std::pow(rc[0], 2) - 3*std::pow(rc[1], 2);
    der2xyz[1] = -6*rc[0]*rc[1];
    der2xyz[2] = 0.0;
}

template <typename CoordType>
void b36(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[1];
    der2xyz[0] = 6*rc[0]*rc[1];
    der2xyz[1] = 3*std::pow(rc[0], 2) - 3*std::pow(rc[1], 2);
    der2xyz[2] = 0.0;
}

template <typename CoordType>
void b40(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2) + std::pow(rc[1], 2) + std::pow(rc[2], 2);
    val = (35*std::pow(rc[2], 2) - 30*r2)*std::pow(rc[2], 2) + 3*std::pow(r2, 2);
    der2xyz[0] = -60*rc[0]*std::pow(rc[2], 2) + 12*r2*rc[0];
    der2xyz[1] = -60*rc[1]*std::pow(rc[2], 2) + 12*r2*rc[1];
    der2xyz[2] = 80*std::pow(rc[2], 3) - 48*r2*rc[2];
}

template <typename CoordType>
void b41(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2) + std::pow(rc[1], 2) + std::pow(rc[2], 2);
    val = (7*std::pow(rc[2], 2) - 3*r2) * rc[0] * rc[2];
    der2xyz[0] = -6*std::pow(rc[0], 2)*rc[2] + 7*std::pow(rc[2], 3) - 3*r2*rc[2];
    der2xyz[1] = -6*rc[0]*rc[1]*rc[2];
    der2xyz[2] = 15*rc[0]*std::pow(rc[2], 2) - 3*r2*rc[0];
}

template <typename CoordType>
void b42(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2) + std::pow(rc[1], 2) + std::pow(rc[2], 2);
    val = (7*std::pow(rc[2], 2) - 3*r2) * rc[1] * rc[2];
    der2xyz[0] = -6*rc[0]*rc[1]*rc[2];
    der2xyz[1] = -6*std::pow(rc[1], 2)*rc[2] + 7*std::pow(rc[2], 3) - 3*r2*rc[2];
    der2xyz[2] = 15*rc[1]*std::pow(rc[2], 2) - 3*r2*rc[1];
}

template <typename CoordType>
void b43(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2) + std::pow(rc[1], 2) + std::pow(rc[2], 2);
    val = (7*std::pow(rc[2], 2) - r2) * (std::pow(rc[0], 2) - std::pow(rc[1], 2));
    der2xyz[0] = -2*std::pow(rc[0], 3) + 2*rc[0]*std::pow(rc[1], 2) + 14*rc[0]*std::pow(rc[2], 2) - 2*r2*rc[0];
    der2xyz[1] = -2*std::pow(rc[0], 2)*rc[1] + 2*std::pow(rc[1], 3) - 14*rc[1]*std::pow(rc[2], 2) + 2*r2*rc[1];
    der2xyz[2] = 12*rc[2] * (std::pow(rc[0], 2) - std::pow(rc[1], 2));
}

template <typename CoordType>
void b44(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType r2 = std::pow(rc[0], 2) + std::pow(rc[1], 2) + std::pow(rc[2], 2);
    val = (7*std::pow(rc[2], 2) - r2)*2*rc[0]*rc[1];
    der2xyz[0] = -4*std::pow(rc[0], 2)*rc[1] + 14*rc[1]*std::pow(rc[2], 2) - 2*r2*rc[1];
    der2xyz[1] = -4*rc[0]*std::pow(rc[1], 2) + 14*rc[0]*std::pow(rc[2], 2) - 2*r2*rc[0];
    der2xyz[2] = 24*rc[0]*rc[1]*rc[2];
}

template <typename CoordType>
void b45(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0] * rc[2];
    der2xyz[0] = 3*std::pow(rc[0], 2)*rc[2] - 3*std::pow(rc[1], 2)*rc[2];
    der2xyz[1] = -6*rc[0]*rc[1]*rc[2];
    der2xyz[2] = (std::pow(rc[0], 2) - 3*std::pow(rc[1], 2)) * rc[0];
}

template <typename CoordType>
void b46(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    val = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2))*rc[1]*rc[2];
    der2xyz[0] = 6*rc[0]*rc[1]*rc[2];
    der2xyz[1] = 3*std::pow(rc[0], 2)*rc[2] - 3*std::pow(rc[1], 2)*rc[2];
    der2xyz[2] = (3*std::pow(rc[0], 2) - std::pow(rc[1], 2)) * rc[1];
}

template <typename CoordType>
void b47(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType x2_minus_y2 = std::pow(rc[0], 2) - std::pow(rc[1], 2);
    val = std::pow(x2_minus_y2, 2) - 4*std::pow(rc[0], 2)*std::pow(rc[1], 2);
    der2xyz[0] = 4*std::pow(rc[0], 3) - 12*rc[0]*std::pow(rc[1], 2);
    der2xyz[1] = 4*std::pow(rc[1], 3) - 12*std::pow(rc[0], 2)*rc[1];
    der2xyz[2] = 0.0;
}

template <typename CoordType>
void b48(CoordType &val, CoordType *der2xyz, CoordType *rc) {
    CoordType x2_minus_y2 = std::pow(rc[0], 2) - std::pow(rc[1], 2);
    val = 4*x2_minus_y2*rc[0]*rc[1];
    der2xyz[0] = 12*std::pow(rc[0], 2)*rc[1] - 4*std::pow(rc[1], 3);
    der2xyz[1] = 4*std::pow(rc[0], 3) - 12*rc[0]*std::pow(rc[1], 2);
    der2xyz[2] = 0.0;
}


template <typename CoordType>
class Sinlm {
public:
    Sinlm();

    Sinlm(CoordType lambda_val, 
          CoordType rmax_r,
          CoordType rmin_r,
          int n_r_max,
          int n_r_basis,
          int max_body,
          CoordType rmax_a,
          CoordType rmin_a,
          int n_a_max,
          int n_a_basis,
          int l_3b_max);
    
    Sinlm(Gn<CoordType> *ptr_gn_r,
          Gn<CoordType> *ptr_gn_a,
          int n_r_max,
          int max_body,
          int n_a_max,
          int l_3b_max);

    Sinlm(const Sinlm<CoordType> &rhs);

    Sinlm(Sinlm<CoordType> &&rhs);

    Sinlm<CoordType>& operator=(const Sinlm<CoordType> &rhs);

    Sinlm<CoordType>& operator=(Sinlm<CoordType> &&rhs);

    ~Sinlm();

    void accum_val_der_r_one(
        CoordType &val_r_one,
        CoordType *der2xyz_r_one,
        CoordType *der2coeffs_r_one,
        CoordType *neigh_vec,
        CoordType distance_ij,
        CoordType *coeffs_r_one);

    void accum_val_der_a_one_lm(
        CoordType &val_a_one,
        CoordType *der2xyz_a_one,
        CoordType *der2coeffs_a_one,
        CoordType *neigh_vec,
        CoordType distance_ij,
        CoordType *coeffs_a_one,
        int l,
        void (*ptr_blm)(CoordType&, CoordType*, CoordType*));

    void find_val_der(
        CoordType *val_r,
        CoordType *val_a,
        CoordType *der2xyz_r,
        CoordType *der2xyz_a,
        CoordType *der2coeffs_r,
        CoordType *der2coeffs_a,
        int inum,
        int *ilist,
        int *numneigh,
        int *firstneigh,
        int ntypes,
        int *types,
        CoordType *rcs,
        int umax_num_neighs,
        CoordType *coeffs_r,
        CoordType *coeffs_a);

    const Gn<CoordType>* ptr_gn_r() const;
    
    const Gn<CoordType>* ptr_gn_a() const;

    const CoordType rmax_r() const;

    const CoordType rmin_r() const;

    const CoordType rmax_a() const;

    const CoordType rmin_a() const;

    const int n_r_max() const;

    const int n_r_basis() const;

    const int max_body() const;

    const int n_a_max() const;

    const int n_a_basis() const;

    const int l_3b_max() const;

    const int num_s_a() const;

private:
    Gn<CoordType> *_ptr_gn_r = nullptr;
    Gn<CoordType> *_ptr_gn_a = nullptr;
    int _n_r_max = 0;
    int _n_r_basis = 0;
    int _max_body = 0;
    int _n_a_max = 0;
    int _n_a_basis = 0;
    int _l_3b_max = 0;   // l_3b_max >= 1
    int _num_s_a = 0;    // size of Sinlm = (n_c, n^a_{max}, \sum^{l^{3b}_{max}}_{l=1}{(2l+1)}). This dimension is determined by l and m.
};  // class : Sinlm


template <typename CoordType>
Sinlm<CoordType>::Sinlm() {     // 2+4 
    this->_ptr_gn_r = new Gn<CoordType>();
    this->_ptr_gn_a = new Gn<CoordType>();
    this->_n_r_max = 0;
    this->_n_r_basis = 0;
    this->_max_body = 0;
    this->_n_a_max = 0;
    this->_n_a_basis = 0;
    this->_l_3b_max = 0;
    this->_num_s_a = 0;
}

template <typename CoordType>
Sinlm<CoordType>::Sinlm(CoordType lambda_val,
                        CoordType rmax_r,
                        CoordType rmin_r,
                        int n_r_max,
                        int n_r_basis,
                        int max_body,
                        CoordType rmax_a,
                        CoordType rmin_a,                        
                        int n_a_max,
                        int n_a_basis,
                        int l_3b_max)
{
    assert((l_3b_max >= 1) && (l_3b_max <= 4));
    assert((max_body >= 2) && (max_body <= 5));
    this->_ptr_gn_r = new Gn<CoordType>(n_r_basis, rmax_r, rmin_r, lambda_val);
    this->_ptr_gn_a = new Gn<CoordType>(n_a_basis, rmax_a, rmin_a, lambda_val);
    this->_n_r_max = n_r_max;
    this->_n_r_basis = this->_ptr_gn_r->chebyshev_size();
    this->_max_body = max_body;
    this->_n_a_max = n_a_max;
    this->_n_a_basis = this->_ptr_gn_a->chebyshev_size();
    this->_l_3b_max = l_3b_max;
    this->_num_s_a = 0;
    for (int ii=1; ii<=this->_l_3b_max; ii++)
        this->_num_s_a += 2*ii+1;
}

template <typename CoordType>
Sinlm<CoordType>::Sinlm(Gn<CoordType> *ptr_gn_r,
                        Gn<CoordType> *ptr_gn_a,
                        int n_r_max,
                        int max_body,
                        int n_a_max,
                        int l_3b_max)
{
    assert(l_3b_max <= 4);
    assert((max_body >=2) && (max_body<=5));
    this->_ptr_gn_r = new Gn<CoordType>(*ptr_gn_r);
    this->_ptr_gn_a = new Gn<CoordType>(*ptr_gn_a);
    this->_n_r_max = n_r_max;
    this->_n_r_basis = this->_ptr_gn_r->chebyshev_size();
    this->_max_body = max_body;
    this->_n_a_max = n_a_max;
    this->_n_a_basis = this->_ptr_gn_a->chebyshev_size();
    this->_l_3b_max = l_3b_max;
    this->_num_s_a = 0;
    for (int ii=1; ii<=this->_l_3b_max; ii++)
        this->_num_s_a += 2*ii+1;
}

template <typename CoordType>
Sinlm<CoordType>::Sinlm(const Sinlm<CoordType> &rhs)
{
    if (rhs.ptr_gn_r()->chebyshev_size() > 0)
        this->_ptr_gn_r = new Gn<CoordType>(*rhs.ptr_gn_r());
    else
        this->_ptr_gn_r = new Gn<CoordType>();
    if (rhs.ptr_gn_a()->chebyshev_size() > 0)
        this->_ptr_gn_a = new Gn<CoordType>(*rhs.ptr_gn_a());
    else
        this->_ptr_gn_a = new Gn<CoordType>();
    this->_n_r_max = rhs._n_r_max;
    this->_n_r_basis = rhs._n_r_basis;
    this->_max_body = rhs._max_body;
    this->_n_a_max = rhs._n_a_max;
    this->_n_a_basis = rhs._n_a_basis;
    this->_l_3b_max = rhs._l_3b_max;
    this->_num_s_a = rhs._num_s_a;
}

template <typename CoordType>
Sinlm<CoordType>::Sinlm(Sinlm<CoordType> &&rhs)
{
    if (this != &rhs) {
        this->_ptr_gn_r = rhs._ptr_gn_r;
        rhs._ptr_gn_r = nullptr;
        this->_ptr_gn_a = rhs._ptr_gn_a;
        rhs._ptr_gn_a = nullptr;
        this->_n_r_max = rhs._n_r_max;
        rhs._n_r_max = 0;
        this->_n_r_basis = rhs._n_r_basis;
        rhs._n_r_basis = 0;
        this->_max_body = rhs._max_body;
        rhs._max_body = 0;
        this->_n_a_max = rhs._n_a_max;
        rhs._n_a_max = 0;
        this->_n_a_basis = rhs._n_a_basis;
        rhs._n_a_basis = 0;
        this->_l_3b_max = rhs._l_3b_max;
        rhs._l_3b_max = 0;
        this->_num_s_a = rhs._num_s_a;
        rhs._num_s_a = 0;
    }
}

template <typename CoordType>
Sinlm<CoordType>& Sinlm<CoordType>::operator=(const Sinlm<CoordType> &rhs)
{
    delete this->_ptr_gn_r;
    delete this->_ptr_gn_a;
    
    this->_ptr_gn_r = new Gn<CoordType>(*rhs.ptr_gn_r());
    this->_ptr_gn_a = new Gn<CoordType>(*rhs.ptr_gn_a());
    this->_n_r_max = rhs._n_r_max;
    this->_n_r_basis = rhs._n_r_basis;
    this->_max_body = rhs._max_body;
    this->_n_a_max = rhs._n_a_max;
    this->_n_a_basis = rhs._n_a_basis;
    this->_l_3b_max = rhs._l_3b_max;
    this->_num_s_a = rhs._num_s_a;
    return *this;
}

template <typename CoordType>
Sinlm<CoordType>& Sinlm<CoordType>::operator=(Sinlm<CoordType> &&rhs)
{
    delete this->_ptr_gn_r;
    delete this->_ptr_gn_a;
    
    if (this != &rhs) {
        this->_ptr_gn_r = rhs._ptr_gn_r;
        rhs._ptr_gn_r = nullptr;
        this->_ptr_gn_a = rhs._ptr_gn_a;
        rhs._ptr_gn_a = nullptr;
        this->_n_r_max = rhs._n_r_max;
        rhs._n_r_max = 0;
        this->_n_r_basis = rhs._n_r_basis;
        rhs._n_r_basis = 0;
        this->_max_body = rhs._max_body;
        rhs._max_body = 0;
        this->_n_a_max = rhs._n_a_max;
        rhs._n_a_max = 0;
        this->_n_a_basis = rhs._n_a_basis;
        rhs._n_a_basis = 0;
        this->_l_3b_max = rhs._l_3b_max;
        rhs._l_3b_max = 0;
        this->_num_s_a = rhs._num_s_a;
        rhs._num_s_a = 0;
    }
    return *this;
}

template <typename CoordType>
Sinlm<CoordType>::~Sinlm()
{
    delete this->_ptr_gn_r;
    delete this->_ptr_gn_a;
    this->_n_r_max = 0;
    this->_n_r_basis = 0;
    this->_max_body = 0;
    this->_n_a_max = 0;
    this->_n_a_basis = 0;
    this->_l_3b_max = 0;
    this->_num_s_a = 0;
}


template <typename CoordType>
void Sinlm<CoordType>::accum_val_der_r_one(
    CoordType &val_r_one,
    CoordType *der2xyz_r_one,
    CoordType *der2coeffs_r_one,
    CoordType *neigh_vec,
    CoordType distance_ij,
    CoordType *coeffs_r_one)
{
    this->_ptr_gn_r->build(distance_ij, coeffs_r_one);
    val_r_one += this->_ptr_gn_r->val();
    for (int ii=0; ii<this->_n_r_basis; ii++)
        der2coeffs_r_one[ii] += this->_ptr_gn_r->der2coeffs()[ii];
    for (int ii=0; ii<3; ii++)
        der2xyz_r_one[ii] = this->_ptr_gn_r->der2r() * neigh_vec[ii] / distance_ij;
}


template <typename CoordType>
void Sinlm<CoordType>::accum_val_der_a_one_lm(
    CoordType &val_a_one,
    CoordType *der2xyz_a_one,
    CoordType *der2coeffs_a_one,
    CoordType *neigh_vec,
    CoordType distance_ij,
    CoordType *coeffs_a,
    int l,
    void (*ptr_blm)(CoordType&, CoordType*, CoordType*))
{
    this->_ptr_gn_a->build(distance_ij, coeffs_a);
    CoordType blm_val;
    CoordType blm_der2xyz[3];
    ptr_blm(blm_val, blm_der2xyz, neigh_vec);
    val_a_one += this->_ptr_gn_a->val() / std::pow(distance_ij, l) * blm_val;
    for (int ii=0; ii<this->_n_r_basis; ii++)
        der2coeffs_a_one[ii] += this->_ptr_gn_a->der2coeffs()[ii] / std::pow(distance_ij, l) * blm_val;
    for (int ii=0; ii<3; ii++)
        der2xyz_a_one[ii] = this->_ptr_gn_a->der2r() * neigh_vec[ii] / std::pow(distance_ij, l+1) * blm_val
                            - l * this->_ptr_gn_a->val() * neigh_vec[ii] / std::pow(distance_ij, l+2) * blm_val
                            + this->_ptr_gn_a->val() / std::pow(distance_ij, l) * blm_der2xyz[ii];
}


template <typename CoordType>
void Sinlm<CoordType>::find_val_der(
    CoordType *val_r,
    CoordType *val_a,
    CoordType *der2xyz_r,
    CoordType *der2xyz_a,
    CoordType *der2coeffs_r,
    CoordType *der2coeffs_a,
    int inum,
    int *ilist,
    int *numneigh,
    int *firstneigh,
    int ntypes,
    int *types,
    CoordType *rcs,
    int umax_num_neighs,
    CoordType *coeffs_r,
    CoordType *coeffs_a)
{
    int tot_num_s_r = inum * this->_n_r_max;
    int tot_num_s_a = inum * this->_n_a_max * this->_num_s_a;
    int num_coeffs_r = ntypes * ntypes * this->_n_r_max * this->_n_r_basis;
    int num_coeffs_a = ntypes * ntypes * this->_n_a_max * this->_n_a_basis;
    memset(val_r, 0, tot_num_s_r * sizeof(CoordType));
    memset(val_a, 0, tot_num_s_a * sizeof(CoordType));
    memset(der2xyz_r, 0, tot_num_s_r * umax_num_neighs * 3 * sizeof(CoordType));  // Note : Don't repeat index.
    memset(der2xyz_a, 0, tot_num_s_a * umax_num_neighs * 3 * sizeof(CoordType));
    memset(der2coeffs_r, 0, tot_num_s_r * ntypes * this->_n_r_basis * sizeof(CoordType));
    memset(der2coeffs_a, 0, tot_num_s_a * ntypes * this->_n_a_basis * sizeof(CoordType));
    // For 3b/4b/5b
    int tmp_blm_idx;
    int tmp_sidx;
    CoordType *tmp_der2xyz;
    CoordType *tmp_der2coeffs_a;
    int tmp_coeffs_idx;
    CoordType *tmp_coeffs_a;

    for (int ii=0; ii<inum; ii++) {
        int cidx = ilist[ii];
        int itype = types[cidx];
        for (int jj=0; jj<numneigh[ii]; jj++) {
            int nidx = firstneigh[ii*umax_num_neighs+jj];
            int jtype = types[nidx];
            CoordType *neigh_vec = &rcs[ii*umax_num_neighs*3 + jj*3]; // Note : 
            CoordType distance_ij = std::sqrt(std::pow(neigh_vec[0], 2)
                                              + std::pow(neigh_vec[1], 2)
                                              + std::pow(neigh_vec[2], 2));

            // For 2b
            if ( (distance_ij >= this->_ptr_gn_r->ptr_rq()->rmin()) && (distance_ij <= this->_ptr_gn_r->ptr_rq()->rmax()) )
            {
                for (int kk=0; kk<this->_n_r_max; kk++) {
                    int s_idx = ii*this->_n_r_max + kk;
                    int coeffs_r_idx = (itype*ntypes + jtype)*this->_n_r_max*this->_n_r_basis + kk*this->_n_r_basis + 0;
                    CoordType *tmp_coeffs_r = &coeffs_r[coeffs_r_idx];    // Note :
                    this->_ptr_gn_r->build(distance_ij, tmp_coeffs_r);
                    val_r[s_idx] += this->_ptr_gn_r->val();
                    for (int ll=0; ll<this->_n_r_basis; ll++) 
                        der2coeffs_r[s_idx*ntypes*this->_n_r_basis
                                    + jtype*this->_n_r_basis
                                    + ll] += this->_ptr_gn_r->der2coeffs()[ll];
                    der2xyz_r[s_idx*umax_num_neighs*3
                              + jj*3 + 0] = this->_ptr_gn_r->der2r() * neigh_vec[0] / distance_ij;
                    der2xyz_r[s_idx*umax_num_neighs*3
                              + jj*3 + 1] = this->_ptr_gn_r->der2r() * neigh_vec[1] / distance_ij;
                    der2xyz_r[s_idx*umax_num_neighs*3
                              + jj*3 + 2] = this->_ptr_gn_r->der2r() * neigh_vec[2] / distance_ij;
                }
            }
            
            // For 3b/4b/5b
            if ( (this->_max_body >= 3)
                 && (distance_ij >= this->_ptr_gn_a->ptr_rq()->rmin()) 
                 && (distance_ij <= this->_ptr_gn_a->ptr_rq()->rmax()) )
            {
                for (int kk=0; kk<this->_n_a_max; kk++) {
                    tmp_coeffs_idx = (itype*ntypes+jtype)*this->_n_a_max*this->_n_a_basis + kk*this->_n_a_basis + 0;
                    tmp_coeffs_a = &coeffs_a[tmp_coeffs_idx];
                    // l=1
                    if (this->_l_3b_max >= 1) {
                        // b10
                        tmp_blm_idx = 0;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 1, &b10<CoordType>);
                        // b11
                        tmp_blm_idx = 1;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 1, &b11<CoordType>);
                        // b12
                        tmp_blm_idx = 2;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 1, &b12<CoordType>);
                    }
                    // l=2
                    if (this->_l_3b_max >= 2) {
                        // b20
                        tmp_blm_idx = 3;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 2, &b20<CoordType>);
                        // b21
                        tmp_blm_idx = 4;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 2, &b21<CoordType>);
                        // b22
                        tmp_blm_idx = 5;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 2, &b22<CoordType>);
                        // b23
                        tmp_blm_idx = 6;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 2, &b23<CoordType>);
                        // b24
                        tmp_blm_idx = 7;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 2, &b24<CoordType>);
                    }
                    // l=3
                    if (this->_l_3b_max >= 3) {
                        // 30
                        tmp_blm_idx = 8;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 3, &b30<CoordType>);
                        // b31
                        tmp_blm_idx = 9;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 3, &b31<CoordType>);
                        // b32
                        tmp_blm_idx = 10;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 3, &b32<CoordType>);
                        // b33
                        tmp_blm_idx = 11;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 3, &b33<CoordType>);
                        // b34
                        tmp_blm_idx = 12;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 3, &b34<CoordType>);
                        // b35
                        tmp_blm_idx = 13;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 3, &b35<CoordType>);
                        // b36
                        tmp_blm_idx = 14;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 3, &b36<CoordType>);
                    }
                    // l=4
                    if (this->_l_3b_max >= 4) {
                        // b40
                        tmp_blm_idx = 15;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b40<CoordType>);
                        // b41
                        tmp_blm_idx = 16;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b41<CoordType>);
                        // b42
                        tmp_blm_idx = 17;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b42<CoordType>);
                        // b43
                        tmp_blm_idx = 18;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b43<CoordType>);
                        // b44
                        tmp_blm_idx = 19;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b44<CoordType>);
                        // b45
                        tmp_blm_idx = 20;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b45<CoordType>);
                        // b46
                        tmp_blm_idx = 21;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b46<CoordType>);
                        // b47
                        tmp_blm_idx = 22;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b47<CoordType>);
                        // b48
                        tmp_blm_idx = 23;
                        tmp_sidx = ii*this->_n_a_max*this->_num_s_a + kk*this->_num_s_a + tmp_blm_idx;
                        tmp_der2xyz = &der2xyz_a[tmp_sidx*umax_num_neighs*3 + jj*3 + 0];
                        tmp_der2coeffs_a = &der2coeffs_a[tmp_sidx*ntypes*this->_n_a_basis + jtype*this->_n_a_basis + 0];
                        this->accum_val_der_a_one_lm(val_a[tmp_sidx], tmp_der2xyz, tmp_der2coeffs_a, neigh_vec, distance_ij, tmp_coeffs_a, 4, &b48<CoordType>);
                    }
                }
            }
        }
    }
}

template <typename CoordType>
const Gn<CoordType>* Sinlm<CoordType>::ptr_gn_r() const {
    return this->_ptr_gn_r;
}

template <typename CoordType>
const Gn<CoordType>* Sinlm<CoordType>::ptr_gn_a() const {
    return this->_ptr_gn_a;
}

template <typename CoordType>
const CoordType Sinlm<CoordType>::rmax_r() const {
    return this->_ptr_gn_r->ptr_rq()->rmax();
}

template <typename CoordType>
const CoordType Sinlm<CoordType>::rmin_r() const {
    return this->_ptr_gn_r->ptr_rq()->rmin();
}

template <typename CoordType>
const CoordType Sinlm<CoordType>::rmax_a() const {
    return this->_ptr_gn_a->ptr_rq()->rmax();
}

template <typename CoordType>
const CoordType Sinlm<CoordType>::rmin_a() const {
    return this->_ptr_gn_a->ptr_rq()->rmin();
}

template <typename CoordType>
const int Sinlm<CoordType>::n_r_max() const {
    return this->_n_r_max;
}

template <typename CoordType>
const int Sinlm<CoordType>::n_r_basis() const {
    return this->_n_r_basis;
}

template <typename CoordType>
const int Sinlm<CoordType>::max_body() const {
    return this->_max_body;
}

template <typename CoordType>
const int Sinlm<CoordType>::n_a_max() const {
    return this->_n_a_max;
}

template <typename CoordType>
const int Sinlm<CoordType>::n_a_basis() const {
    return this->_n_a_basis;
}

template <typename CoordType>
const int Sinlm<CoordType>::l_3b_max() const {
    return this->_l_3b_max;
}

template <typename CoordType>
const int Sinlm<CoordType>::num_s_a() const {
    return this->_num_s_a;
}


};  // namespace : ace
};  // namespace : ai2pot

#endif
