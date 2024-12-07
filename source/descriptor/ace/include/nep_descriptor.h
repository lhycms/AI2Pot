#ifndef AI2POT_ACE_NEP_DESCRIPTOR_H
#define AI2POT_ACE_NEP_DESCRIPTOR_H
#include <cmath>

#include "./sinlm.h"


namespace ai2pot {
namespace ace {

template <typename CoordType>
class NepDescriptor {
private:
        
    Sinlm<CoordType> _sinlm;


public:
    void find_3b_descriptor_val_der_one();

    void find_4b_descriptor_val_der_one();

    void find_5b_descriptor_val_der_one();

    void find_descriptor_val_der();
};  // class : NepDescriptor


};  // namespace : ace
};  // namespace : nep
#endif