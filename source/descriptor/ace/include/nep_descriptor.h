#ifndef AI2POT_ACE_NEP_DESCRIPTOR_H
#define AI2POT_ACE_NEP_DESCRIPTOR_H
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <array>

#include "./sinlm.h"


namespace ai2pot {
namespace ace {


template <typename CoordType>
class PseudoWigner 
{
public:
    PseudoWigner()
    {
        this->_C3B = {
            0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
            0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
            0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
            1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
            0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
            1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606};
        this->_C4B = {
            -0.007499480826664,
            -0.134990654879954,
            0.067495327439977,
            0.404971964639861,
            -0.809943929279723
        };
        this->_C5B = {
            0.026596810706114, 
            0.053193621412227, 
            0.026596810706114};
    }

    ~PseudoWigner() = default;

    const CoordType* C3B() const {
        return this->_C3B.data();
    }

    const CoordType* C4B() const {
        return this->_C4B.data();
    }

    const CoordType* C5B() const {
        return this->_C5B.data();
    }

private:
    std::array<CoordType, 24> _C3B; // 3+5+7+9=24
    std::array<CoordType, 5> _C4B;
    std::array<CoordType, 3> _C5B;
}; // class: PseudoWigner


template <typename CoordType>
class NepDescriptor {
public:
    static void find_hyperparams(
        int &num_descriptors_r,
        int &num_descriptors_a,
        int &num_l,
        int &l_max,
        int max_body,
        int n_r_max,
        int n_a_max,
        int l_3b_max,
        int l_4b_max,
        int l_5b_max);

    static void find_descriptor_val_der_atom(
        CoordType *nep_val_atom,
        CoordType (*nep_der2xyz_atom)[3],
        CoordType *nep_der2coeffs_r_atom,
        CoordType *nep_der2coeffs_a_atom,
        CoordType *coeffs_r,
        CoordType *coeffs_a,
        CoordType lambda_val,
        CoordType rmax_r,
        CoordType rmin_r,
        int n_r_max,
        int n_r_basis,
        int max_body,
        CoordType rmax_a,
        CoordType rmin_a,
        int n_a_max,
        int n_a_basis,
        int l_max,
        int num_l,
        int num_r_descriptors,
        int num_a_descriptors,
        int iilist,
        int inumneigh,
        int *ifirstneigh,
        CoordType (*ircs)[3],
        int *types,
        int ntype,
        int umax_num_neigh_atoms);

    static void find_descriptor_val_der();

};  // class : NepDescriptor



template <typename CoordType>
void NepDescriptor<CoordType>::find_hyperparams(
    int &num_descriptors_r,
    int &num_descriptors_a,
    int &num_l,
    int &l_max,
    int max_body,
    int n_r_max,
    int n_a_max,
    int l_3b_max,
    int l_4b_max,
    int l_5b_max)
{
    num_descriptors_r = n_r_max;
    num_l = 0;
    if (max_body >= 3) 
        num_l += l_3b_max;
    if (max_body >= 4) 
        num_l += 1;
    if (max_body >= 5)
        num_l += 1;
    num_descriptors_a = n_a_max * num_l;
    l_max = (l_3b_max > 0) ? l_3b_max : 0;
    l_max = (l_max > l_4b_max) ? l_max : l_4b_max;
    l_max = (l_max > l_5b_max) ? l_max : l_5b_max;
}




template <typename CoordType>
static void NepDescriptor<CoordType>::find_descriptor_val_der_atom(
    CoordType *nep_val_atom,
    CoordType (*nep_der2xyz_atom)[3],
    CoordType *nep_der2coeffs_r_atom,
    CoordType *nep_der2coeffs_a_atom,
    CoordType *coeffs_r,
    CoordType *coeffs_a,
    CoordType lambda_val,
    CoordType rmax_r,
    CoordType rmin_r,
    int n_r_max,
    int n_r_basis,
    int max_body,
    CoordType rmax_a,
    CoordType rmin_a,
    int n_a_max,
    int n_a_basis,
    int l_max,
    int num_l,
    int num_descriptors_r,
    int num_descriptors_a,
    int iilist,
    int inumneigh,
    int *ifirstneigh,
    CoordType (*ircs)[3],
    int *types,
    int ntypes,
    int umax_num_neigh_atoms)
{
    int num_descriptors = num_descriptors_r + num_descriptors_a;
    memset(nep_val_atom, 0, sizeof(CoordType) * num_descriptors);
    memset(nep_der2xyz_atom, 0, sizeof(CoordType) * num_descriptors * umax_num_neigh_atoms * 3);
    memset(nep_der2coeffs_r_atom, 0, sizeof(CoordType) * num_descriptors_r * ntypes * ntypes * n_r_basis);
    memset(nep_der2coeffs_a_atom, 0, sizeof(CoordType) * num_descriptors_a * ntypes * ntypes * n_a_basis);
    PseudoWigner<CoordType> pwigner;
    
    Sinlm<double> sinlm(lambda_val,
                        rmax_r,
                        rmin_r,
                        n_r_max,
                        n_r_basis,
                        max_body,
                        rmax_a,
                        rmin_a,
                        n_a_max,
                        n_a_basis,
                        l_max);
    if (max_body >= 2) {
        CoordType *nep_val_r_atom = nep_val_atom;
        CoordType (*nep_der2xyz_r_atom)[3] = nep_der2xyz_atom;
        sinlm.find_val_der_r(nep_val_r_atom,
                             (CoordType*)nep_der2xyz_r_atom,
                             nep_der2coeffs_r_atom,
                             coeffs_r,
                             iilist,
                             inumneigh,
                             ifirstneigh,
                             (CoordType*)ircs,
                             types,
                             ntypes,
                             umax_num_neigh_atoms);
    }
    
    CoordType *s_val_a_atom;
    CoordType *s_der2xyz_a_atom;
    CoordType *s_der2coeffs_a_atom;
    if (max_body > 2) {
        s_val_a_atom = (CoordType*)malloc(sizeof(CoordType) * sinlm.n_a_max() * sinlm.num_s_a());
        s_der2xyz_a_atom = (CoordType (*)[3])malloc(sizeof(CoordType) * sinlm.n_a_max() * sinlm.num_s_a() * umax_num_neigh_atoms * 3);
        s_der2coeffs_a_atom = (CoordType *)malloc(sizeof(CoordType) * sinlm.n_a_max() * sinlm.num_s_a() * ntypes * ntypes * n_a_basis);
        sinlm.find_val_der_a_lm(s_val_a_atom,
                                (CoordType*)(s_der2xyz_a_atom),
                                s_der2coeffs_a_atom,
                                coeffs_a,
                                iilist,
                                inumneigh,
                                ifirstneigh,
                                (CoordType)ircs,
                                types,
                                ntypes,
                                umax_num_neigh_atoms);
    }

    for (int nn=0; nn<n_a_max; nn++) {
        CoordType *nep_val_a_one = &nep_val_atom[num_descriptors_r + nn*num_l];
        CoordType (*nep_der2xyz_a_one)[3] = &nep_der2xyz_atom[(num_descriptors_r + nn*num_l)*umax_num_neigh_atoms];
        CoordType *nep_der2coeffs_a_one = &nep_der2coeffs_a_atom[(num_descriptors_r + nn*num_l)*ntypes*ntypes*n_a_basis];
        int s_idx;
        CoordType s_val_a_one;
        CoordType *s_der2xyz_a_one;
        CoordType *s_der2coeffs_a_one;
        if (num_l >= 1) {
            for (int blm_idx=0; blm_idx<2*1 + 1; blm_idx++) {
                s_idx = nn*sinlm.num_s_a() + blm_idx;
                s_val_a_one = s_val_a_atom[s_idx];
                s_der2xyz_a_one = &s_der2xyz_a_atom[s_idx*umax_num_neigh_atoms*3 + 0];
                s_der2coeffs_a_one = &s_der2coeffs_a_atom[s_idx*ntypes*ntypes*n_a_basis + 0];
                
                nep_val_a_one[1] += pwigner.C3B()[blm_idx] * s_val_a_one * s_val_a_one;
                for (int a=0; a<3; a++)
                    nep_der2xyz_a_one[1*umax_num_neigh_atoms*3 + a] += 2
                                                                       * s_val_a_one
                                                                       * s_der2xyz_a_one[blm_idx*umax_num_neigh_atoms*3 + a];
                for (int kk=0; kk<n_a_basis; kk++)
                    nep_der2coeffs_a_one[]
            }
        }
        if (num_l >= 2) {
            for (int blm_idx=3; blm_idx<3 + 2*2 + 1; blm_idx++) {
                nep_val_a_one[2] = 1;
            }
        }
        if (num_l >= 3) {
            for (int blm_idx=8; blm_idx<8 + 2*3 + 1; blm_idx++) {
                nep_val_a_one[3] = 1;
            }
        }
        if (num_l >= 4) {
            for (int blm_idx=15; blm_idx<15 + 2*4 + 1; blm_idx++) {
                nep_val_a_one[4] = 1;
            }
        }
        if (num_l >= 5) {
            
        }
        if (num_l >= 6) {

        }
    }
}


};  // namespace : ace
};  // namespace : nep
#endif