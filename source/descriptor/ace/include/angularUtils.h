#ifndef AI2POT_ACE_ANGULAR_UTILS_H
#define AI2POT_ACE_ANGULAR_UTILS_H
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include "radialBasis.h"

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



private:
    Gn<CoordType> *_ptr_gn_r = nullptr;
    Gn<CoordType> *_ptr_gn_a = nullptr;
    int _n_r_max = 0;
    int _n_r_basis = 0;
    int _max_body = 0;
    int _n_a_max = 0;
    int _n_a_basis = 0;
    int _l_3b_max = 0;   // l_3b_max >= 1
};  // class : Sinlm


template <typename CoordType>
Sinlm<CoordType>::Sinlm() {     // 2+4 
    this->_ptr_gn_r = nullptr;
    this->_ptr_gn_a = nullptr;
    this->_n_r_max = 0;
    this->_n_r_basis = 0;
    this->_max_body = 0;
    this->_n_a_max = 0;
    this->_n_a_basis = 0;
    this->_l_3b_max = 0;
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
    this->_ptr_gn_r = new Gn<CoordType>(n_r_basis, rmax_r, rmin_r, lambda_val);
    this->_ptr_gn_a = new Gn<CoordType>(n_a_basis, rmax_a, rmin_a, lambda_val);
    this->_n_r_max = n_r_max;
    this->_n_r_basis = this->_ptr_gn_r->chebyshev_size();
    this->_max_body = max_body;
    this->_n_a_max = n_a_max;
    this->_n_a_basis = this->_ptr_gn_a->chebyshev_size();
    this->_l_3b_max = l_3b_max;
}

template <typename CoordType>
Sinlm<CoordType>::Sinlm(Gn<CoordType> *ptr_gn_r,
                        Gn<CoordType> *ptr_gn_a,
                        int n_r_max,
                        int max_body,
                        int n_a_max,
                        int l_3b_max)
{
    this->_ptr_gn_r = new Gn<CoordType>(*ptr_gn_r);
    this->_ptr_gn_a = new Gn<CoordType>(*ptr_gn_a);
    this->_n_r_max = n_r_max;
    this->_n_r_basis = this->_ptr_gn_r->chebyshev_size();
    this->_max_body = max_body;
    this->_n_a_max = n_a_max;
    this->_n_a_basis = this->_ptr_gn_a->chebyshev_size();
    this->_l_3b_max = l_3b_max;
    
}



};  // namespace : ace
};  // namespace : ai2pot

#endif