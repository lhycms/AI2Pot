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
          int l_3b_max);    // 5+6
    
    Sinlm(Gn<CoordType> *ptr_gn_r,
          Gn<CoordType> *ptr_gn_a,
          int n_r_max,
          int max_body,
          int n_a_max,
          int l_3b_max);    // 1+3

    Sinlm(const Sinlm<CoordType> &rhs);

    Sinlm(Sinlm<CoordType> &&rhs);

    Sinlm<CoordType>& operator=(const Sinlm<CoordType> &rhs);

    Sinlm<CoordType>& operator=(Sinlm<CoordType> &&rhs);

    ~Sinlm();

    // Calculate Sinlm for single neighbor atom, fix ii, jj, and n_a_idx
    void find_val_der4blm_neigh_atom(
        CoordType *val_a,
        CoordType *der2xyz_a,
        CoordType *der2coeffs_a,
        int ii,
        int jj,
        int n_a_idx,
        int ntypes,
        int itype,
        int jtype,
        CoordType *coeffs_a,
        CoordType *rc,
        CoordType distance_ij,
        int umax_num_neighs,
        int l,
        int blm_idx,
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
void Sinlm<CoordType>::find_val_der4blm_neigh_atom(
    CoordType *val_a,
    CoordType *der2xyz_a,
    CoordType *der2coeffs_a,
    int ii,
    int jj,
    int n_a_idx,
    int ntypes,
    int itype,
    int jtype,
    CoordType *coeffs_a,
    CoordType *rc,
    CoordType distance_ij,
    int umax_num_neighs,
    int l,
    int blm_idx,
    void (*ptr_blm)(CoordType&, CoordType*, CoordType*))
{
    CoordType b_val;
    CoordType b_der2xyz[3];
    int coeffs_idx = (itype*ntypes+jtype)*this->_n_a_max*this->_n_a_basis + n_a_idx*this->_n_a_basis + 0;
    CoordType *tmp_coeffs_a = &coeffs_a[coeffs_idx];
    this->_ptr_gn_a->build(distance_ij, coeffs_a);

    int sidx = ii*this->_n_a_max*this->_num_s_a + n_a_idx*this->_num_s_a + blm_idx;
    ptr_blm(b_val, b_der2xyz, rc);
    val_a[sidx] += this->_ptr_gn_a->val() / std::pow(distance_ij, l) * b_val;
    for (int ii=0; ii<this->_n_a_basis; ii++)
        der2coeffs_a[sidx*ntypes*ntypes*this->_n_a_basis
                     + (itype*ntypes+jtype)*this->_n_a_basis 
                     + ii] += this->_ptr_gn_a->der2coeffs()[ii] / std::pow(distance_ij, l) * b_val;
    der2xyz_a[sidx*umax_num_neighs*3 + jj*3 + 0] = 
        this->_ptr_gn_a->der2r() * rc[0] / distance_ij / std::pow(distance_ij, l) * b_val
        - l * this->_ptr_gn_a->val() * rc[0] / std::pow(distance_ij, l+2) * b_val
        + this->_ptr_gn_a->val() / std::pow(distance_ij, l) * b_der2xyz[0];
    der2xyz_a[sidx*umax_num_neighs*3 + jj*3 + 1] =
        this->_ptr_gn_a->der2r() * rc[1] / distance_ij / std::pow(distance_ij, l) * b_val
        - l * this->_ptr_gn_a->val() * rc[1] / std::pow(distance_ij, l+2) * b_val
        + this->_ptr_gn_a->val() / std::pow(distance_ij, l) * b_der2xyz[1];
    der2xyz_a[sidx*umax_num_neighs*3 + jj*3 + 2] = 
        this->_ptr_gn_a->der2r() * rc[2] / distance_ij / std::pow(distance_ij, l) * b_val
        - l * this->_ptr_gn_a->val() * rc[2] / std::pow(distance_ij, l+2) * b_val
        + this->_ptr_gn_a->val() / std::pow(distance_ij, l) * b_der2xyz[2];
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
    memset(der2coeffs_r, 0, tot_num_s_r * ntypes * ntypes * this->_n_r_basis * sizeof(CoordType));
    memset(der2coeffs_a, 0, tot_num_s_a * ntypes * ntypes * this->_n_a_basis * sizeof(CoordType));

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
                        der2coeffs_r[s_idx*ntypes*ntypes*this->_n_r_basis
                                    + (itype*ntypes+jtype)*this->_n_r_basis
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
                    // l=1 
                    if (this->_l_3b_max >= 1) {
                        // b10
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            1, 0, &b10<CoordType>);
                        // b11
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            1, 1, &b11<CoordType>);
                        // b12
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            1, 2, &b12<CoordType>);
                    }
                    // l=2
                    if (this->_l_3b_max >= 2) {
                        // b20
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            2, 3, &b20<CoordType>);
                        // b21
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            2, 4, &b21<CoordType>);
                            // b22
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            2, 5, &b22<CoordType>);
                            // b23
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            2, 6, &b23<CoordType>);
                            // b24
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            2, 7, &b24<CoordType>);
                    }
                    // l=3
                    if (this->_l_3b_max >= 3) {
                            // 30
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            3, 8, &b30<CoordType>);
                            // b31
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            3, 9, &b31<CoordType>);
                            // b32
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            3, 10, &b32<CoordType>);
                            // b33
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            3, 11, &b33<CoordType>);
                            // b34
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            3, 12, &b34<CoordType>);
                            // b35
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            3, 13, &b35<CoordType>);
                            // b36
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            3, 14, &b36<CoordType>);
                    }
                    // l=4
                    if (this->_l_3b_max >= 4) {
                        // b40
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 15, &b40<CoordType>);
                        // b41
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 16, &b41<CoordType>);
                        // b42
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 17, &b42<CoordType>);
                        // b43
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 18, &b43<CoordType>);
                        // b44
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 19, &b44<CoordType>);
                        // b45
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 20, &b45<CoordType>);
                        // b46
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 21, &b46<CoordType>);
                        // b47
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 22, &b21<CoordType>);
                        // b48
                        this->find_val_der4blm_neigh_atom(val_a, der2xyz_a, der2coeffs_a, ii, jj, kk,
                            ntypes, itype, jtype, coeffs_a, neigh_vec, distance_ij, umax_num_neighs,
                            4, 23, &b21<CoordType>);
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