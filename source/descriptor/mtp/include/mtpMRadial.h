#ifndef MATERSDK_MTPM_RADIAL_H
#define MATERSDK_MTPM_RADIAL_H
#include <cmath>
#include <string.h>

namespace matersdk {
namespace mtp {

template <typename CoordType>
class MtpSwitchFunc1 {
public:
    MtpSwitchFunc1();

    MtpSwitchFunc1(CoordType rcut, CoordType rcut_smooth);

    MtpSwitchFunc1(const MtpSwitchFunc1& rhs);

    MtpSwitchFunc1& operator=(const MtpSwitchFunc1& rhs);

    CoordType get_result(CoordType distance_ij) const;

    CoordType get_deriv2r() const;

private:
    CoordType rcut = 0;
    CoordType rcut_smooth = 0;
};  // class SwitchFunction1


template <typename CoordType>
class MtpSwitchFunc2 {
public:
    MtpSwitchFunc2();

    MtpSwitchFunc2(CoordType rcut, CoordType rcut_smooth);

    MtpSwitchFunc2(const MtpSwitchFunc2& rhs);

    MtpSwitchFunc2& operator=(const MtpSwitchFunc2& rhs);

    CoordType get_result(CoordType distance_ij) const;

    CoordType get_deriv2r(CoordType distance_ij) const;

private:
    CoordType rcut = 0;
    CoordType rcut_smooth = 0;
}; // class MtpSwitchFunc2


template <typename CoordType>
class ChebyshevPoly {
public:
    ChebyshevPoly();

    ChebyshevPoly(
        int size,
        CoordType rcut,
        CoordType rcut_smooth);

    ChebyshevPoly(const ChebyshevPoly& rhs);

    ChebyshevPoly(ChebyshevPoly&& rhs);
    
    ~ChebyshevPoly();

    ChebyshevPoly& operator=(const ChebyshevPoly& rhs);

    ChebyshevPoly& operator=(ChebyshevPoly&& rhs);

    void build(CoordType distance_ij);  // Calculate `this->result` and `this->deriv2r`

    const int size() const;

    const CoordType* get_result() const;

    const CoordType* get_deriv2xi() const;

    const CoordType* get_deriv2r() const;

    void show() const;

private:
    int size_ = 0;
    MtpSwitchFunc1<CoordType> switch_func;
    CoordType rcut = 0;
    CoordType rcut_smooth = 0;
    CoordType* result = nullptr;
    CoordType* deriv2xi = nullptr;
    CoordType* deriv2r = nullptr;
};  // class : ChebyshevPoly


template <typename CoordType>
class MtpQ {
public:
    MtpQ();

    MtpQ(int size, CoordType rcut, CoordType rcut_smooth);

    MtpQ(const MtpQ& rhs);

    MtpQ(MtpQ&& rhs);

    ~MtpQ();

    MtpQ& operator=(const MtpQ& rhs);

    MtpQ& operator=(MtpQ&& rhs);

    void build(CoordType distance_ij);

    const int size() const;

    const CoordType* get_result() const;

    const CoordType* get_deriv2r() const;

    void show() const;

private:
    int size_ = 0;
    MtpSwitchFunc2<CoordType> switch_func2;
    ChebyshevPoly<CoordType> chebyshev;
    CoordType* result;
    CoordType* deriv2r;
};  // class : MtpQ





template <typename CoordType>
MtpSwitchFunc1<CoordType>::MtpSwitchFunc1() {};

/**
 * @brief Construct a new Mtp Switch Func 1< Coord Type>:: Mtp Switch Func 1 object
 * Scaling distance_ij to [-1, 1]
 * $$\xi = \frac{2*r_{ij} - (R_{cut} + R_{cut_smooth})}{R_{cut} - R_{cut_smooth}}$$
 * @tparam CoordType 
 * @param rcut 
 * @param rcut_smooth 
 */
template <typename CoordType>
MtpSwitchFunc1<CoordType>::MtpSwitchFunc1(CoordType rcut, CoordType rcut_smooth)
{
    this->rcut = rcut;
    this->rcut_smooth = rcut_smooth;
}

template <typename CoordType>
MtpSwitchFunc1<CoordType>::MtpSwitchFunc1(const MtpSwitchFunc1& rhs)
{
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;
}

template <typename CoordType>
MtpSwitchFunc1<CoordType>& MtpSwitchFunc1<CoordType>::operator=(
    const MtpSwitchFunc1& rhs)
{
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;
    return *this;
}

template <typename CoordType>
CoordType MtpSwitchFunc1<CoordType>::get_result(CoordType distance_ij) const
{
    return (2 * distance_ij - (this->rcut + this->rcut_smooth)) / (this->rcut - this->rcut_smooth);
}

template <typename CoordType>
CoordType MtpSwitchFunc1<CoordType>::get_deriv2r() const
{
    return 2 / (this->rcut - this->rcut_smooth);
}



template <typename CoordType>
MtpSwitchFunc2<CoordType>::MtpSwitchFunc2() {};

/**
 * @brief Construct a new Mtp Switch Func 2< Coord Type>:: Mtp Switch Func 2 object
 *  Scaling distance_ij to [0, 1]
 *      $$uu = /frac{r_ij - rcut}{rcut - rcut_smooth}$$
 *      vv = 
 *          1, r_ij < rcut_smooth
 *          uu^3(-6uu^2 + 15uu -10) + 1, rcut_smooth <= r_ij < rcut
 *          0, r_ij >= rcut
 * @tparam CoordType 
 * @param rcut 
 * @param rcut_smooth 
 */
template <typename CoordType>
MtpSwitchFunc2<CoordType>::MtpSwitchFunc2(CoordType rcut, CoordType rcut_smooth)
{
    this->rcut = rcut;
    this->rcut_smooth = rcut_smooth;
}

template <typename CoordType>
MtpSwitchFunc2<CoordType>::MtpSwitchFunc2(const MtpSwitchFunc2& rhs)
{
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;
}

template <typename CoordType>
MtpSwitchFunc2<CoordType>& MtpSwitchFunc2<CoordType>::operator=(const MtpSwitchFunc2& rhs)
{
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;
    return *this;
}

template <typename CoordType>
CoordType MtpSwitchFunc2<CoordType>::get_result(CoordType distance_ij) const
{
    CoordType result;
    CoordType uu = (distance_ij - this->rcut_smooth) / (this->rcut - this->rcut_smooth);

    if (distance_ij < this->rcut_smooth)
        result = 0;
    else if ((distance_ij>=this->rcut_smooth) && (distance_ij<this->rcut))
        result = std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1;
    else
        result = 0;
    return result;
}

template <typename CoordType>
CoordType MtpSwitchFunc2<CoordType>::get_deriv2r(CoordType distance_ij) const
{
    CoordType deriv2r;
    CoordType uu = (distance_ij - this->rcut_smooth) / (this->rcut - this->rcut_smooth);
    if (distance_ij < this->rcut_smooth)
        deriv2r = 0;
    else if ((distance_ij>=this->rcut_smooth) && (distance_ij<this->rcut))
        deriv2r = 1 / (this->rcut - this->rcut_smooth) * ( -30*std::pow(uu, 4) + 60*std::pow(uu, 3) - 30*std::pow(uu, 2));
    else
        deriv2r = 0;
    return deriv2r;
}


template <typename CoordType>
ChebyshevPoly<CoordType>::ChebyshevPoly() {}

template <typename CoordType>
ChebyshevPoly<CoordType>::ChebyshevPoly(
    int size,
    CoordType rcut,
    CoordType rcut_smooth)
{
    this->size_ = size;
    this->switch_func = MtpSwitchFunc1<CoordType>(rcut, rcut_smooth);
    this->rcut = rcut;
    this->rcut_smooth = rcut_smooth;
    this->result = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    memset(this->result, 0, sizeof(CoordType) * this->size_);
    this->deriv2xi = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    memset(this->deriv2xi, 0, sizeof(CoordType) * this->size_);
    this->deriv2r = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    memset(this->deriv2r, 0, sizeof(CoordType) * this->size_);
}

template <typename CoordType>
ChebyshevPoly<CoordType>::ChebyshevPoly(const ChebyshevPoly& rhs)
{
    this->size_ = rhs.size_;
    this->switch_func = rhs.switch_func;
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;
    this->result = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    this->deriv2xi = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    this->deriv2r = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    for (int ii=0; ii<this->size_; ii++) {
        this->result[ii] = rhs.result[ii];
        this->deriv2xi[ii] = rhs.deriv2xi[ii];
        this->deriv2r[ii] = rhs.deriv2r[ii];
    }
}

template <typename CoordType>
ChebyshevPoly<CoordType>::ChebyshevPoly(ChebyshevPoly&& rhs)
{
    if (this != &rhs) {
        this->size_ = rhs.size_;
        this->switch_func = rhs.switch_func;
        this->rcut = rhs.rcut;
        this->rcut_smooth = rhs.rcut_smooth;
        this->result = rhs.result;
        this->deriv2xi = rhs.deriv2xi;
        this->deriv2r = rhs.deriv2r;

        rhs.size_ = 0;
        rhs.switch_func = MtpSwitchFunc1<CoordType>();
        rhs.rcut = 0;
        rhs.rcut_smooth = 0;
        rhs.result = nullptr;
        rhs.deriv2xi = nullptr;
        rhs.deriv2r = nullptr;
    }
}

template <typename CoordType>
ChebyshevPoly<CoordType>::~ChebyshevPoly() 
{
    this->size_ = 0;
    this->rcut = 0;
    this->rcut_smooth = 0;
    free(this->result);
    free(this->deriv2xi);
    free(this->deriv2r);
}

template <typename CoordType>
ChebyshevPoly<CoordType>& ChebyshevPoly<CoordType>::operator=(const ChebyshevPoly& rhs)
{
    if (this->size_ != 0) {
        this->size_ = 0;
        this->rcut = 0;
        this->rcut_smooth = 0;
        free(this->result);
        free(this->deriv2xi);
        free(this->deriv2r);
    }

    this->size_ = rhs.size_;
    this->switch_func = rhs.switch_func;
    this->rcut = rhs.rcut;
    this->rcut_smooth = rhs.rcut_smooth;
    this->result = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    this->deriv2xi = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    this->deriv2r = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    for (int ii=0; ii<this->size_; ii++) {
        this->result[ii] = rhs.result[ii];
        this->deriv2xi[ii] = rhs.deriv2xi[ii];
        this->deriv2r[ii] = rhs.deriv2r[ii];
    }
    return *this;
}

template <typename CoordType>
ChebyshevPoly<CoordType>& ChebyshevPoly<CoordType>::operator=(ChebyshevPoly&& rhs)
{
    if (this != &rhs) {
        if (this->size_ != 0) {
            this->size_ = 0;
            this->rcut = 0;
            this->rcut_smooth = 0;
            free(this->result);
            free(this->deriv2xi);
            free(this->deriv2r);
        }
        this->size_ = rhs.size_;
        this->switch_func = rhs.switch_func;
        this->rcut = rhs.rcut;
        this->rcut_smooth = rhs.rcut_smooth;
        this->result = rhs.result;
        this->deriv2xi = rhs.deriv2xi;
        this->deriv2r = rhs.deriv2r;

        rhs.size_ = 0;
        rhs.rcut = 0;
        rhs.rcut_smooth = 0;
        rhs.result = nullptr;
        rhs.deriv2xi = nullptr;
        rhs.deriv2r = nullptr;
    }    
}

template <typename CoordType>
void ChebyshevPoly<CoordType>::build(CoordType distance_ij) 
{
    CoordType xi = this->switch_func.get_result(distance_ij);
    if ((distance_ij >= this->rcut_smooth) && (distance_ij < this->rcut)) { // Chebyshev Only defined on [-1, 1].
        for (int ii=0; ii<this->size_; ii++) {
            if (ii == 0) {
                this->result[ii] = 1;
                this->deriv2xi[ii] = 0;
                this->deriv2r[ii] = this->deriv2xi[ii] * this->switch_func.get_deriv2r();
            } else if (ii == 1) {
                this->result[ii] = xi;
                this->deriv2xi[ii] = 1;
                this->deriv2r[ii] = this->deriv2xi[ii] * this->switch_func.get_deriv2r();
            } else {
                this->result[ii] = 2 * xi * this->result[ii-1] - this->result[ii-2];
                this->deriv2xi[ii] = (
                    2 * this->result[ii-1] + 
                    2 * xi * this->deriv2xi[ii-1] - 
                    this->deriv2xi[ii-2]
                );
                this->deriv2r[ii] = this->deriv2xi[ii] * this->switch_func.get_deriv2r();
            }
        }
    } else {
        memset(this->result, 0, sizeof(CoordType) * this->size_);
        memset(this->deriv2xi, 0, sizeof(CoordType) * this->size_);
        memset(this->deriv2r, 0, sizeof(CoordType) * this->size_);
    }
}

template <typename CoordType>
const int ChebyshevPoly<CoordType>::size() const
{
    return (const int)this->size_;
}

template <typename CoordType>
const CoordType* ChebyshevPoly<CoordType>::get_result() const
{
   return (const CoordType*)this->result;
}

template <typename CoordType>
const CoordType* ChebyshevPoly<CoordType>::get_deriv2xi() const
{
    return (const CoordType*)this->deriv2xi;
}

template <typename CoordType>
const CoordType* ChebyshevPoly<CoordType>::get_deriv2r() const
{
    return (const CoordType*)this->deriv2r;
}

template <typename CoordType>
void ChebyshevPoly<CoordType>::show() const
{
    printf("ChebyshevPoly.result:\n\t");
    for (int ii=0; ii<this->size_; ii++) {
        printf("%6f, ", this->result[ii]);
    }
    printf("\n");

    printf("ChebyshevPoly.deriv2xi:\n\t");
    for (int ii=0; ii<this->size_; ii++) {
        printf("%6f, ", this->deriv2xi[ii]);
    }
    printf("\n");

    printf("ChebyshevPoly.deriv2r:\n\t");
    for (int ii=0; ii<this->size_; ii++) {
        printf("%6f, ", this->deriv2r[ii]);
    }
    printf("\n");
}


template <typename CoordType>
MtpQ<CoordType>::MtpQ() {}

template <typename CoordType>
MtpQ<CoordType>::MtpQ(int size, CoordType rcut, CoordType rcut_smooth)
{
    this->size_ = size;
    this->switch_func2 = MtpSwitchFunc2<CoordType>(rcut, rcut_smooth);
    this->chebyshev = ChebyshevPoly<CoordType>(size, rcut, rcut_smooth);
    this->result = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    memset(this->result, 0, sizeof(CoordType) * this->size_);
    this->deriv2r = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    memset(this->deriv2r, 0, sizeof(CoordType) * this->size_);
}

template <typename CoordType>
MtpQ<CoordType>::MtpQ(const MtpQ& rhs)
{
    this->size_ = rhs.size_;
    this->switch_func2 = rhs.switch_func2;
    this->chebyshev = rhs.chebyshev;
    this->result = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    this->deriv2r = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    for (int ii=0; ii<this->size_; ii++) {
        this->result[ii] = rhs.result[ii];
        this->deriv2r[ii] = rhs.deriv2r[ii];
    }
}

template <typename CoordType>
MtpQ<CoordType>::MtpQ(MtpQ&& rhs)
{
    if (this != &rhs) {
        this->size_ = rhs.size_;
        this->switch_func2 = rhs.switch_func2;
        this->chebyshev = rhs.chebyshev;
        this->result = rhs.result;
        this->deriv2r = rhs.deriv2r;

        rhs.size_ = 0;
        rhs.switch_func2 = MtpSwitchFunc2<CoordType>();
        rhs.chebyshev = ChebyshevPoly<CoordType>();
        rhs.result = nullptr;
        rhs.deriv2r = nullptr;
    }
}

template <typename CoordType>
MtpQ<CoordType>::~MtpQ()
{
    this->size_ = 0;
    free(this->result);
    free(this->deriv2r);
}

template <typename CoordType>
MtpQ<CoordType>& MtpQ<CoordType>::operator=(const MtpQ& rhs)
{
    if (this->size_ != 0) {
        this->size_ = 0;
        free(this->result);
        free(this->deriv2r);
    }
    this->size_ = rhs.size_;
    this->switch_func2 = rhs.switch_func2;
    this->chebyshev = rhs.chebyshev;
    this->result = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    this->deriv2r = (CoordType*)malloc(sizeof(CoordType) * this->size_);
    for (int ii=0; ii<this->size_; ii++) {
        this->result[ii] = rhs.result[ii];
        this->deriv2r[ii] = rhs.deriv2r[ii];
    }
}

template <typename CoordType>
MtpQ<CoordType>& MtpQ<CoordType>::operator=(MtpQ&& rhs)
{
    if (this != &rhs) {
        if (this->size_ != 0) {
            this->size_ = 0;
            free(this->result);
            free(this->deriv2r);
        }
        this->size_ = rhs.size_;
        this->switch_func2 = rhs.switch_func2;
        this->chebyshev = rhs.chebyshev;
        this->result = rhs.result;
        this->deriv2r = rhs.deriv2r;

        rhs.size_ = 0;
        rhs.switch_func2 = MtpSwitchFunc2<CoordType>();
        rhs.chebyshev = ChebyshevPoly<CoordType>();
        rhs.result = nullptr;
        rhs.deriv2r = nullptr;
    }
}

template <typename CoordType>
void MtpQ<CoordType>::build(CoordType distance_ij)
{
    CoordType vv_value = this->switch_func2.get_result(distance_ij);
    CoordType vv_deriv2r = this->switch_func2.get_deriv2r(distance_ij);
    this->chebyshev.build(distance_ij);
    for (int ii=0; ii<this->size_; ii++) {
        this->result[ii] = this->chebyshev.get_result()[ii] * vv_value; 
        this->deriv2r[ii] = (
            this->chebyshev.get_deriv2r()[ii] * vv_value + 
            this->chebyshev.get_result()[ii] * vv_deriv2r);
    }
}

template <typename CoordType>
const int MtpQ<CoordType>::size() const 
{
    return (const int)this->size_;
}

template <typename CoordType>
const CoordType* MtpQ<CoordType>::get_result() const
{
    return (const CoordType*)this->result;
}

template <typename CoordType>
const CoordType* MtpQ<CoordType>::get_deriv2r() const
{
    return (const CoordType*)this->deriv2r;
}

template <typename CoordType>
void MtpQ<CoordType>::show() const
{
    printf("MtpQ.result:\n\t");
    for (int ii=0; ii<this->size_; ii++)
        printf("%6f, ", this->result[ii]);
    printf("\n");
    
    printf("MtpQ.deriv2r:\n\t");
    for (int ii=0; ii<this->size_; ii++)
        printf("%6f, ", this->deriv2r[ii]);
    printf("\n");
}


};  // namespace : mtp
};  // namespace : matersdk

#endif
