#include <vector>
#include <torch/torch.h>
#include "../include/mtpMRadialOp.h"
#include "../include/mtpMAngularOp.h"
#include "../include/mtpMModule.h"


namespace matersdk {
namespace mtp {

MtpMModuleImpl::MtpMModuleImpl(
    int64_t nmus,
    int64_t ntypes,
    int64_t size,
    at::Tensor& rcuts_tensor)
{
    this->size = size;
    this->nmus = nmus;
    this->ntypes = ntypes;
    this->rcuts_tensor = rcuts_tensor;
    int64_t ncoeffs = nmus * ntypes * ntypes;
    for (int64_t ii=0; ii<ncoeffs; ii++)
        this->cheby_coeff_list->append(at::randn({size}).requires_grad_(true));
    this->register_module("cheby_coeff_list", this->cheby_coeff_list);
}

/**
 * @brief Calculate f_{\mu}(|r_{ij}, z_i, z_j|)
 * @param iidx The index of center atoms
 * @param ifirstneigh_tensor The indices of neigh atoms.
 *          .shape = [nneighs,]
 * @param types The atomic types
 * @param ircs_tensor The relative coordinates of neigh atoms
 *          .shape = [nneighs, 3]
 * @return at::Tensor
 *          .shape = [nneighs,]
 */
at::Tensor MtpMModuleImpl::forward(
    int64_t mu,
    int64_t nu,
    int64_t iidx,
    const at::Tensor& ifirstneigh_tensor,
    const at::Tensor& types,
    const at::Tensor& ircs_tensor)
{
    c10::TensorOptions options = c10::TensorOptions()
        .dtype(ircs_tensor.scalar_type())
        .device(ircs_tensor.device());
    int64_t nneighs = ircs_tensor.sizes()[0];
    std::vector<int64_t> imtp_m_tensor_shape;
    imtp_m_tensor_shape.resize(nu+1, 0); // .shape = [nneighs, 3, ...]
    imtp_m_tensor_shape[0] = nneighs;
    for (int ii=1; ii<nu+1; ii++)
        imtp_m_tensor_shape[ii] = 3;
    at::Tensor imtp_m_tensor = at::zeros(imtp_m_tensor_shape, options)
        .requires_grad_(false);
    int64_t z_i = types[iidx].item<int64_t>();
    int64_t* ifirstneigh = ifirstneigh_tensor.data_ptr<int64_t>();
    at::Tensor imtp_q_tensor = MtpQOp(   // .shape = [nneighs, size]
        this->size,
        this->rcuts_tensor,
        ircs_tensor)[0];
    at::Tensor imtp_angular_tensor = MtpMAngularOp(ircs_tensor, nu);    // .shape = [nneighs, 3, ...]
    for (int64_t ii=0; ii<ifirstneigh_tensor.sizes()[0]; ii++) {
        int64_t z_j = types[ifirstneigh[ii]].item<int64_t>();
        imtp_m_tensor[ii] = at::dot(
            this->cheby_coeff_list[mu*this->ntypes*this->ntypes + z_i*this->ntypes + z_j],
            imtp_q_tensor[ii]) * imtp_angular_tensor[ii];
    }
    return imtp_m_tensor.sum(0);    // [19, 3, 3, 3] -> [3, 3, 3]
}

};  // namespace : mtp
};  // namespace : matersdk
