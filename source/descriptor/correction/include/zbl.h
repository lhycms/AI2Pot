#include <stdio.h>
#include <cmath>
#include <stdlib.h>


namespace ai2pot {
namespace correction {

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
                          CoordType distance_ij);

private:
    int _Zi = 0;
    int _Zj = 0;
    CoordType _rmax = 0;
    CoordType _rmin = 0;
    CoordType *_ck;
    CoordType *_dk;
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



};  // namespace : correction
};  // namespace : ai2pot
