#include <stdio.h>
#include <cmath>
#include <math.h>
#include <stdlib.h>


namespace ai2pot {
namespace correction {


double K_C_SP = 14.399645; // 1/(4*PI*epsilon_0*e^2)

template <typename CoordType>
class ZBL {
public:
    ZBL();

    ZBL(int Zi,
        int Zj,
        CoordType rmax,
        CoordType rmin,
        CoordType *ck,
        CoordType *dk);

    ZBL(const ZBL &rhs);

    ZBL(ZBL &&rhs);

    ZBL& operator=(const ZBL &rhs);

    ZBL& operator=(ZBL &&rhs);

    ~ZBL();

    CoordType find_switch_func(CoordType distance_ij);

    CoordType find_phi_func(CoordType distance_ij);

    CoordType find_pair_energy(CoordType distance_ij);

    void add_atomic_energy(CoordType &atomic_energy,
                           CoordType distance_ij);
    
    void add_atomic_force(CoordType *force,
                          CoordType *neigh_vec);

    void add_virial(CoordType *virial,
                    CoordType *neigh_vec);

private:
    int _Zi = 0;
    int _Zj = 0;
    CoordType _rmax = 0;
    CoordType _rmin = 0;
    CoordType *_ck; // c1=0.18175, c2=0.50986, c3=0.28022, c4=0.02817
    CoordType *_dk; // d1=3.1998,  d2=0.94229, d3=0.4029,  d4=0.20162
};  // class : ZBL



template <typename CoordType>
ZBL<CoordType>::ZBL(int Zi,
                    int Zj,
                    CoordType rmax,
                    CoordType rmin,
                    CoordType *ck,
                    CoordType *dk)
{
    this->_Zi = Zi;
    this->_Zj = Zj;
    this->_rmax = rmax;
    this->_rmin = rmin;
    this->_ck = (CoordType*)malloc(sizeof(CoordType) * 4);
    this->_dk = (CoordType*)malloc(sizeof(CoordType) * 4);
    for (int ii=0; ii<4; ii++) {
        this->_ck[ii] = ck[ii];
        this->_dk[ii] = dk[ii];
    }
}


template <typename CoordType>
ZBL<CoordType>::ZBL(const ZBL &rhs)
{
    this->_Zi = rhs._Zi;
    this->_Zj = rhs._Zj;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    this->_ck = (CoordType*)malloc(sizeof(CoordType) * 4);
    this->_dk = (CoordType*)malloc(sizeof(CoordType) * 4);
    for (int ii=0; ii<4; ii++) {
        this->_ck = rhs._ck[ii];
        this->_dk = rhs._dk[ii];
    }
}


template <typename CoordType>
ZBL<CoordType>::ZBL(ZBL &&rhs) {
    this->_Zi = rhs._Zi;
    rhs._Zi = 0;
    this->_Zj = rhs._Zj;
    rhs._Zj = 0;
    this->_rmax = rhs._rmax;
    rhs._rmax = 0;
    this->_rmin = rhs._rmin;
    rhs._rmin = 0;

    this->_ck = rhs._ck;
    rhs._ck = nullptr;
    this->_dk = rhs._dk;
    rhs._dk = nullptr;
}


template <typename CoordType>
ZBL<CoordType>& ZBL<CoordType>::operator=(const ZBL &rhs) {
    this->_Zi = rhs._Zi;
    this->_Zj = rhs._Zj;
    this->_rmax = rhs._rmax;
    this->_rmin = rhs._rmin;
    for (int ii=0; ii<4; ii++) {
        this->_ck[ii] = rhs._ck[ii];
        this->_dk[ii] = rhs._dk[ii];
    }

    return *this;
}


template <typename CoordType>
ZBL<CoordType>& ZBL<CoordType>::operator=(ZBL &&rhs) {
    this->_Zi = rhs._Zi;
    rhs._Zi = 0;
    this->_Zj = rhs._Zj;
    rhs._Zj = 0;
    this->_rmax = rhs._rmax;
    rhs._rmax = 0;
    this->_rmin = rhs._rmin;
    rhs._rmin = 0;

    this->_ck = rhs._ck;
    rhs._ck = nullptr;
    this->_dk = rhs._dk;
    rhs._dk = nullptr;
    
    return *this;
}


template <typename CoordType>
ZBL<CoordType>::~ZBL() {
    this->_Zi = 0;
    this->_Zj = 0;
    this->_rmax = 0;
    this->_rmin = 0;
    free(this->_ck);
    free(this->_dk);
}


template <typename CoordType>
CoordType ZBL<CoordType>::find_switch_func(CoordType distance_ij)
{
    if (distance_ij < this->_rmin)
        return 1.0;
    else if ((distance_ij>=this->_rmin) && (distance_ij<=this->_rmax)) {
        CoordType uu = (distance_ij - this->_rmin) / (this->_rmax - this->_rmin);
        return std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1;
    }
    else
        return 0.0;
}


template <typename CoordType>
CoordType ZBL<CoordType>::find_phi_func(CoordType distance_ij)
{
    CoordType phi_val = 0;
    CoordType a = 0.46848 / (std::pow(this->_Zi, 0.23)  + std::pow(this->_Zj, 0.23));
    CoordType x = distance_ij / a;

    for (int ii=0; ii<4; ii++)
        phi_val += this->_ck[ii] * std::exp(-this->_dk[ii] * x);

    return phi_val;
}


template <typename CoordType>
CoordType ZBL<CoordType>::find_pair_energy(CoordType distance_ij)
{
    CoordType phi_val = this->find_phi_func(distance_ij);
    CoordType switch_val = this->find_switch_func(distance_ij);
    CoordType pair_energy = K_C_SP * (this->_Zi*this->_Zj) / distance_ij 
                            * phi_val 
                            * switch_val;
    return pair_energy;
}


template <typename CoordType>
void ZBL<CoordType>::add_atomic_energy(CoordType &atomic_energy,
                                       CoordType distance_ij)
{
    CoordType half_pair_energy = this->find_pair_energy(distance_ij) / 2;
    atomic_energy += half_pair_energy;
}


template <typename CoordType>
void ZBL<CoordType>::add_atomic_force(CoordType *atomic_force,
                                      CoordType *neigh_vec)
{
    CoordType distance_ij = std::sqrt( std::pow(neigh_vec[0], 2) 
                                       + std::pow(neigh_vec[1], 2)
                                       + std::pow(neigh_vec[2], 2) );
    CoordType A = K_C_SP*this->_Zi*this->_Zj/distance_ij;
    CoordType B = this->find_phi_func(distance_ij);
    CoordType C = this->find_switch_func(distance_ij);
    CoordType A_ders[3] = {0., 0., 0.};
    CoordType B_ders[3] = {0., 0., 0.};
    CoordType C_ders[3] = {0., 0., 0.};
    A_ders[0] = ;
    A_ders[1] = ;
    A_ders[2] = ;
    B_ders[0] = ;
    B_ders[1] = ;
    B_ders[2] = ;
    C_ders[0] = ;
    C_ders[1] = ;
    C_ders[2] = ;

    for (int ii=0; ii<3; ii++)
        //atomic_force[ii] += ;
}


};  // namespace : correction
};  // namespace : ai2pot
