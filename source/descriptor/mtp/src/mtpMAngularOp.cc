#include <cassert>
#include <cmath>
#include <vector>
#include "../include/mtpMAngularOp.h"

namespace matersdk {
namespace mtp {


torch::autograd::variable_list OuterNu0Function::forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor ircs_tensor)
{
    c10::TensorOptions options = c10::TensorOptions()
        .dtype(ircs_tensor.scalar_type())
        .device(ircs_tensor.device());
    int64_t nneighs = ircs_tensor.sizes()[0];
    at::Tensor result_tensor = at::zeros({nneighs}, options);
    at::Tensor deriv2xyz_tensor = at::zeros({nneighs, 3}, options);
    if (options.dtype() == torch::kFloat32) {
        float* result = result_tensor.data_ptr<float>();
        for (int64_t ii=0; ii<nneighs; ii++)
            result[ii] = 1.0;
    } else {
        double* result = result_tensor.data_ptr<double>();
        for (int64_t ii=0; ii<nneighs; ii++)
            result[ii] = 1.0;
    }
    ctx->save_for_backward({deriv2xyz_tensor});

    return {result_tensor};
}


torch::autograd::variable_list OuterNu0Function::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs)
{
    at::Tensor deriv2xyz_tensor = ctx->get_saved_variables()[0];
    return {deriv2xyz_tensor};
}


torch::autograd::variable_list OuterNu0Op(const at::Tensor& ircs_tensor)
{
    return OuterNu0Function::apply(ircs_tensor);
}


/**
 * @brief Given i,j, and \nu. Calculate $r_{ij} \otimes_{nu} r_{ij}$
 * 
 * @param ircs_tensor .shape = [nneighs, 3]
 * @param nu $\nu$ in r_{ij} \otimes_{nu} r_{ij}
 * @return at::Tensor 
 */
at::Tensor MtpMAngularOp(
    const at::Tensor& ircs_tensor,
    int nu)
{
    c10::TensorOptions options = c10::TensorOptions()
        .device(ircs_tensor.device())
        .dtype(ircs_tensor.scalar_type());
    std::vector<int64_t> imtpm_angular_shape;
    imtpm_angular_shape.resize(nu+1, 0);
    int64_t nneighs = ircs_tensor.sizes()[0];
    imtpm_angular_shape[0] = nneighs;
    for (int ii=1; ii<nu+1; ii++)
        imtpm_angular_shape[ii] = 3;
    at::Tensor imtpm_angular = at::zeros(imtpm_angular_shape, options)
        .requires_grad_(false);
    if (nu == 0) {
        imtpm_angular = OuterNu0Op(ircs_tensor)[0];
    } else if (nu == 1) {
        for (int64_t ii=0; ii<nneighs; ii++)
            imtpm_angular[ii] = ircs_tensor[ii];
    } else {
        for (int64_t ii=0; ii<nneighs; ii++) {
            at::Tensor ijrc_tensor = ircs_tensor[ii];
            at::Tensor ijmtp_angular = ijrc_tensor;
            for (int jj=0; jj<nu-1; jj++) {
                ijmtp_angular = torch::tensordot(
                    ijmtp_angular,
                    ijrc_tensor,
                    {}, {});
            }
            imtpm_angular[ii] = ijmtp_angular;
        }
    }

    return imtpm_angular;
}

};  // namespace : mtp
};  // namespace : matersdk