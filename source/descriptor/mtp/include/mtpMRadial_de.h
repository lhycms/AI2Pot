#ifndef MATERSDK_MTP_H
#define MATERSDK_MTP_H

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cmath>
#include "../../../nblist/include/structure.h"
#include "../../../nblist/include/neighborList.h"
#include "../../../core/include/vec3Operation.h"
#include "../../../core/include/arrayUtils.h"


namespace matersdk {
namespace mtp {


/**
 * @brief Switching Function in MTP like DeepPot-SE
 *          0. uu = \frac{r - r_s}{r_c - r_s}
 *          1. switchFunc(uu) = 
 *              1. 1
 *              2. uu^3(-6uu^2 + 15uu -10) + 1
 *              3. 0
 *          2. the gradient of switchFunc(uu) with respect to uu:
 *              1. 0
 *              2. -30 uu^4 + 60 uu^3 - 30 uu^2
 *              3. 0
 *          3. the gradient of switchFunc(uu) with respect to r:
 *              1. 0
 *              2. (-30 uu^4 _ 60 uu^3 - 30 uu^2) * 1/(r_c-r_s)
 *              3. 0
 * 
 */
template <typename CoordType>
class SwitchFunc {
public:
    SwitchFunc(CoordType rcut, CoordType rcut_smooth);

    CoordType get_result(CoordType r_ji) const;

    CoordType get_deriv2rji(CoordType r_ji) const;

    void show() const;

private:
    CoordType rcut = 0;
    CoordType rcut_smooth = 0;
};   // class : SwitchFunc


template <typename CoordType>
SwitchFunc<CoordType>::SwitchFunc(CoordType rcut, CoordType rcut_smooth) {
    this->rcut = rcut;
    this->rcut_smooth = rcut_smooth;
}


template <typename CoordType>
CoordType SwitchFunc<CoordType>::get_result(CoordType r_ji) const {
    CoordType result;
    CoordType uu = (r_ji - this->rcut_smooth) / (this->rcut - this->rcut_smooth);

    if (r_ji < this->rcut_smooth)
        result = 1;
    else if ((r_ji>=this->rcut_smooth) && (r_ji<this->rcut))
        result = std::pow(uu, 3) * (-6*std::pow(uu, 2) + 15*uu - 10) + 1;
    else
        result = 0;
    
    return result;
}


template <typename CoordType>
CoordType SwitchFunc<CoordType>::get_deriv2rji(CoordType r_ji) const {
    CoordType deriv2rji;
    CoordType uu = (r_ji - this->rcut_smooth) / (this->rcut - this->rcut_smooth);

    if (r_ji < this->rcut_smooth)
        deriv2rji = 0;
    else if ((r_ji>=this->rcut_smooth) && (r_ji<this->rcut))
        deriv2rji = 1/(this->rcut-this->rcut_smooth) * ( -30*std::pow(uu, 4) + 60*std::pow(uu, 3) - 30*std::pow(uu, 2) );
    else
        deriv2rji = 0;
    
    return deriv2rji;
}   


template <typename CoordType>
void SwitchFunc<CoordType>::show() const {
    printf("Inner SwitchFunc:\n");
    printf("\tthis->rcut = %5f\n", this->rcut);
    printf("\tthis->rcut_smooth = %5f\n", this->rcut_smooth);
}




/**
 * @brief 1. Chebyshev Polynomials:
 *              1. T_0(x) = 1
 *              2. T_1(x) = x
 *              3. T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
 *        2. ksi = (rji - (rcut + rcut_smooth)) / (rcut - rcut_smooth)
 *        3. Chebyshev Polynomials derivatives to x:
 *              1. dT_0(x) = 0
 *              2. dT_1(x) = 1
 *              3. dT_{n+1}(x) = 2*T_n(x) + 2 x dT_n(x) - dT_{n-1}(x)
 *              4. Note: dT_n, dT_{n-1} is deriv with `x`
 *        4. Chebyshev Polynomials derivatives to rji:
 *              1. dT_0(x) = 0
 *              2. dT_1(x) = 2 / (rcut - rcut_smooth) 
 *              3. dT_{n+1}(x) = 2 * 2 / (rcut - rcut_smooth) * T_n(x) + 2 x dT_n(x) - dT_{n-1}(x)
 *              4. Note: dT_n, dT_{n-1} is deriv with `x`
 *      
 * @tparam CoordType 
 */
template <typename CoordType>
class RadialBasisChebyshev {
public:
    RadialBasisChebyshev();

    RadialBasisChebyshev(
                CoordType rcut,
                CoordType rcut_smooth,
                int hmju,
                CoordType rji);  // Highest mju (切比雪夫多项式的最高次数)

    void calc_chebyshev_vals();    // Calculate the value of radial basis without multiplying SwitchFunc

    void calc_chebyshev_ders();    // Calculate the deriv (derivative with r_ji) of radial basis without multiplying SwitchFunc

    void calc_rb_vals();    // rb_vals = chebyshev_vals * SwitchFunc

    void calc_rb_ders();    // rb_vals = chebyshev_vals * SwitchFunc

    void show_in_value() const;

    void show_in_deriv() const;

    const CoordType get_rcut() const;

    const CoordType get_rcut_smooth() const;

    const CoordType get_rji() const;

    const int get_hmju() const;

    const CoordType* get_chebyshev_vals() const;

    const CoordType* get_chebyshev_ders() const;

    const CoordType* get_rb_vals() const;

    const CoordType* get_rb_ders() const;


private:
    CoordType rcut = 0;
    CoordType rcut_smooth = 0;
    int hmju = 0;
    CoordType rji = 0;
    CoordType* chebyshev_vals = nullptr;   // Store the chebyshev_vals[0], chebyshev_vals[1], ..., speed up the calculation
    CoordType* chebyshev_ders = nullptr;   // Store the chebyshev_ders[0], chebyshev_ders[1], ..., speed up the calculation
    CoordType* rb_vals = nullptr;
    CoordType* rb_ders = nullptr;
};  // RadialBasisChebyshev




template <typename CoordType>
RadialBasisChebyshev<CoordType>::RadialBasisChebyshev() {
    this->rcut = 0;
    this->rcut_smooth = 0;
    this->hmju = 0;
    this->rji = 0;
    this->chebyshev_vals = nullptr;
    this->chebyshev_ders = nullptr;
    this->rb_vals = nullptr;
    this->rb_ders = nullptr;
}


template <typename CoordType>
RadialBasisChebyshev<CoordType>::RadialBasisChebyshev(
                CoordType rcut,
                CoordType rcut_smooth,
                int hmju,
                CoordType rji) 
{
    this->rcut = rcut;
    this->rcut_smooth = rcut_smooth;
    assert(this->rcut > this->rcut_smooth);
    this->hmju = hmju;
    this->rji = rji;

    this->chebyshev_vals = (CoordType*)malloc( sizeof(CoordType) * (this->hmju+1) );
    this->chebyshev_ders = (CoordType*)malloc( sizeof(CoordType) * (this->hmju+1) );
    this->rb_vals = (CoordType*)malloc( sizeof(CoordType) * (this->hmju+1) );
    this->rb_ders = (CoordType*)malloc( sizeof(CoordType) * (this->hmju+1) );
    
    for (int ii=0; ii<(this->hmju+1); ii++) {
        this->chebyshev_vals[ii] = 0;
        this->chebyshev_ders[ii] = 0;
        this->rb_vals[ii] = 0;
        this->rb_ders[ii] = 0;
    }

    this->calc_chebyshev_vals();
    this->calc_chebyshev_ders();
    this->calc_rb_vals();
    this->calc_rb_ders();
}



template <typename CoordType>
void RadialBasisChebyshev<CoordType>::calc_chebyshev_vals() 
{
    CoordType ksi = (2*this->rji - (this->rcut+this->rcut_smooth)) / (this->rcut-this->rcut_smooth);
    if (this->hmju == 0) {          // hmju = 0
        this->chebyshev_vals[0] = 1;
        return ;
    }
    else if (this->hmju == 1) {     // hmju = 1
        this->chebyshev_vals[0] = 1;
        this->chebyshev_vals[1] = ksi;
        return ;
    }
    else {                          // hmju > 1
        this->chebyshev_vals[0] = 1;
        this->chebyshev_vals[1] = ksi;
        for (int ii=2; ii<(this->hmju+1); ii++) 
            this->chebyshev_vals[ii] = 2 * ksi * this->chebyshev_vals[ii-1] - this->chebyshev_vals[ii-2];
        return ;
    }
}



template <typename CoordType>
void RadialBasisChebyshev<CoordType>::calc_chebyshev_ders()
{
    CoordType ksi = (2*this->rji - (this->rcut+this->rcut_smooth)) / (this->rcut-this->rcut_smooth);
    CoordType ksi_deriv2rji = 2 / (this->rcut - this->rcut_smooth);
    if (this->hmju == 0) {
        this->chebyshev_ders[0] = 0;
        return ;
    } 
    else if (this->hmju == 1) {
        this->chebyshev_ders[0] = 0;
        this->chebyshev_ders[1] = ksi_deriv2rji;
        return ;
    }
    else {
        this->chebyshev_ders[0] = 0;
        this->chebyshev_ders[1] = ksi_deriv2rji;
        for (int ii=2; ii<(this->hmju+1); ii++) 
            this->chebyshev_ders[ii] = (
                2 * ksi_deriv2rji * this->chebyshev_vals[ii-1] + 
                2 * ksi * this->chebyshev_ders[ii-1] -
                this->chebyshev_ders[ii-2]
            );
        return ;
    }
}


template <typename CoordType>
void RadialBasisChebyshev<CoordType>::calc_rb_vals() {
    SwitchFunc<CoordType> switch_func(this->rcut, this->rcut_smooth);
    CoordType switch_func_value = switch_func.get_result(this->rji);

    if (this->hmju == 0) {
        this->rb_vals[0] = switch_func_value * this->chebyshev_vals[0];
        return ;
    }
    else if (this->hmju == 1) {
        this->rb_vals[0] = switch_func_value * this->chebyshev_vals[0];
        this->rb_vals[1] = switch_func_value * this->chebyshev_vals[1];
        return ;
    }
    else {
        this->rb_vals[0] = switch_func_value * this->chebyshev_vals[0];
        this->rb_vals[1] = switch_func_value * this->chebyshev_vals[1];
        for (int ii=2; ii<(this->hmju+1); ii++)
            this->rb_vals[ii] = switch_func_value * this->chebyshev_vals[ii];
        return ;
    }
}


template <typename CoordType>
void RadialBasisChebyshev<CoordType>::calc_rb_ders() {
    SwitchFunc<CoordType> switch_func(this->rcut, this->rcut_smooth);
    CoordType switch_func_value = switch_func.get_result(this->rji);
    CoordType switch_func_deriv2rji = switch_func.get_deriv2rji(this->rji);

    if (this->hmju == 0) {
        this->rb_ders[0] = (
                switch_func_value * this->chebyshev_ders[0] +
                switch_func_deriv2rji * this->chebyshev_vals[0]
        );
        return ;
    }
    else if (this->hmju == 1) {
        this->rb_ders[0] = (
                switch_func_value * this->chebyshev_ders[0] + 
                switch_func_deriv2rji * this->chebyshev_vals[0]
        );
        this->rb_ders[1] = (
                switch_func_value * this->chebyshev_ders[1] + 
                switch_func_deriv2rji * this->chebyshev_vals[1]
        );
        return ;
    }
    else {
        this->rb_ders[0] = (
                switch_func_value * this->chebyshev_ders[0] + 
                switch_func_deriv2rji * this->chebyshev_vals[0]
        );
        this->rb_ders[1] = (
                switch_func_value * this->chebyshev_ders[1] + 
                switch_func_deriv2rji * this->chebyshev_vals[1]
        );
        for (int ii=2; ii<(this->hmju+1); ii++) {
            this->rb_ders[ii] = (
                switch_func_value * this->chebyshev_ders[ii] + 
                switch_func_deriv2rji * this->chebyshev_vals[ii]
            );
        }
        return ;
    }
}


template <typename CoordType>
void RadialBasisChebyshev<CoordType>::show_in_value() const {
    printf("**************** MTP RadialBasisChebyshev ****************\n");
    printf("\t1. rcut = %5f\n", this->rcut);
    printf("\t2. rcut_smooth = %5f\n", this->rcut_smooth);
    printf("\t3. highest mju (starts from 0) = %3d\n", this->hmju);
    printf("\t4. rji = %5f\n", this->rji);
    printf("\t5. ksi = %5f\n", (2*this->rji - this->rcut - this->rcut_smooth) / (this->rcut - this->rcut_smooth));
    
    if (this->chebyshev_vals != nullptr) {
        printf("\t6. chebyshev_vals = [");
        for (int ii=0; ii<(this->hmju+1); ii++) 
            printf("%8f, ", this->chebyshev_vals[ii]);
        printf("]\n");
    
        printf("\t6. rb_vals = [");
        for (int ii=0; ii<(this->hmju+1); ii++) 
            printf("%8f, ", this->rb_vals[ii]);
        printf("]\n");
    }
    printf("**********************************************************\n");
}


template <typename CoordType>
void RadialBasisChebyshev<CoordType>::show_in_deriv() const {
    printf("**************** MTP RadialBasisChebyshev ****************\n");
    printf("\t1. rcut = %5f\n", this->rcut);
    printf("\t2. rcut_smooth = %5f\n", this->rcut_smooth);
    printf("\t3. highest mju (starts from 0) = %3d\n", this->hmju);
    printf("\t4. rji = %5f\n", this->rji);
    printf("\t5. ksi = %5f\n", (2*this->rji - this->rcut - this->rcut_smooth) / (this->rcut - this->rcut_smooth));
    
    if (this->chebyshev_ders != nullptr) {
        printf("\t6. chebyshev_ders = [");
        for (int ii=0; ii<(this->hmju+1); ii++)
            printf("%8f, ", this->chebyshev_ders[ii]);
        printf("]\n");

        printf("\t6. rb_ders = [");
        for (int ii=0; ii<(this->hmju+1); ii++)
            printf("%8f, ", this->rb_ders[ii]);
        printf("]\n");
    }
    printf("**********************************************************\n");
}


template <typename CoordType>
const CoordType RadialBasisChebyshev<CoordType>::get_rcut() const {
    return (const CoordType)this->rcut;
}


template <typename CoordType>
const CoordType RadialBasisChebyshev<CoordType>::get_rcut_smooth() const {
    return (const CoordType)this->rcut_smooth;
}


template <typename CoordType>
const CoordType RadialBasisChebyshev<CoordType>::get_rji() const {
    return (const CoordType)this->rji;
}


template <typename CoordType>
const int RadialBasisChebyshev<CoordType>::get_hmju() const {
    return (const CoordType)this->hmju;
}


template <typename CoordType>
const CoordType* RadialBasisChebyshev<CoordType>::get_chebyshev_vals() const {
    return (const CoordType*)this->chebyshev_vals;
}


template <typename CoordType>
const CoordType* RadialBasisChebyshev<CoordType>::get_chebyshev_ders() const {
    return (const CoordType*)this->chebyshev_ders;
}


template <typename CoordType>
const CoordType* RadialBasisChebyshev<CoordType>::get_rb_vals() const {
    return (const CoordType*)this->rb_vals;
}


template <typename CoordType>
const CoordType* RadialBasisChebyshev<CoordType>::get_rb_ders() const {
    return (const CoordType*)this->rb_ders;
}


}   // namespace : mtp
}   // namespace : matersdk

#endif
