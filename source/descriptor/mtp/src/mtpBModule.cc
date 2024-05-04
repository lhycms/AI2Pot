#include <torch/torch.h>
#include <vector>
#include "../include/mtpLevel.h"
#include "../include/mtpMModule.h"
#include "../include/mtpBModule.h"


namespace matersdk {
namespace mtp {

MtpBModuleImpl::MtpBModuleImpl(
    int64_t max_level,
    //int64_t nmus,
    int64_t ntypes,
    int64_t size,
    at::Tensor& rcuts_tensor)
{   
    this->_max_level = max_level;
    this->_coeff_pair_combs = MtpMCoeffPairCombs(max_level);
    this->_nmus = this->_coeff_pair_combs.nmus();
    this->_ntypes = ntypes;
    this->_size = size;
    this->_rcuts_tensor = rcuts_tensor;
    this->_mtp_m_list->push_back(
        MtpMModule(
            this->_nmus,
            this->_ntypes,
            this->_size,
            this->_rcuts_tensor));
    this->register_module("mtp_m_list", this->_mtp_m_list);
}

at::Tensor MtpBModuleImpl::forward(
    at::Tensor& ilist_tensor,
    at::Tensor& firstneigh_tensor,
    at::Tensor& rcs_tensor,
    at::Tensor& types_tensor)
{
    c10::TensorOptions options = c10::TensorOptions()
        .dtype(rcs_tensor.scalar_type())
        .device(rcs_tensor.device());
    int64_t* ilist_ptr = ilist_tensor.data_ptr<int64_t>();
    int64_t* firstneigh_ptr = firstneigh_tensor.data_ptr<int64_t>();
    int64_t nbatches = ilist_tensor.sizes()[0];
    int64_t ncenters = ilist_tensor.sizes()[1];
    int64_t nbasis = this->_coeff_pair_combs.size();
    at::Tensor mtp_b_tensor = at::zeros({nbatches, ncenters, nbasis}, options);

    int64_t tmp_mu, tmp_nu;
    int64_t tmp_iidx;

    for (int ii=0; ii<nbatches; ii++) {
        for (int jj=0; jj<ncenters; jj++) {
            tmp_iidx = ilist_ptr[ii*ncenters + jj];
            for (int kk=0; kk<nbasis; kk++) {
                std::vector<MtpMCoeffPair> tmp_coeff_pair_comb = this->_coeff_pair_combs.coeff_pair_combs()[kk];
                at::Tensor partial_mtp_b_tensor = torch::tensor(1, options);
                for (int ll=0; ll<tmp_coeff_pair_comb.size(); ll++) {
                    tmp_mu = tmp_coeff_pair_comb[ll].coeff_pair().first;
                    tmp_nu = tmp_coeff_pair_comb[ll].coeff_pair().second;
                    at::Tensor tmp_mtp_m_tensor = this->_mtp_m_list[0]->as<MtpMModule>()->forward(
                        tmp_mu, 
                        tmp_nu,
                        tmp_iidx,
                        firstneigh_tensor[ii][jj],
                        types_tensor,
                        rcs_tensor[ii][jj]);
                    int64_t dim1 = partial_mtp_b_tensor.dim();
                    int64_t dim2 = tmp_mtp_m_tensor.dim();

                    if (dim1 >= dim2) {
                        std::vector<int64_t> dims1(dim2, 0);
                        std::vector<int64_t> dims2(dim2, 0);
                        for (int mm=0; mm<dim2; mm++) {
                            dims1[mm] = dim1 - dim2 + mm;
                            dims2[mm] = mm;
                        }
                        partial_mtp_b_tensor = at::tensordot(
                            partial_mtp_b_tensor, tmp_mtp_m_tensor,
                            dims1, dims2);
                    }
                    else {
                        assert(dim1 == 0);
                        partial_mtp_b_tensor = partial_mtp_b_tensor * tmp_mtp_m_tensor; // scalar tensor * tensor
                    }
                }
                mtp_b_tensor[ii][jj][kk] = partial_mtp_b_tensor;    // one basis.
            }
        }
    }
    return mtp_b_tensor;
}


};  // namespace : mtp
};  // namespace : matersdk