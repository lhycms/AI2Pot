#include <torch/torch.h>

#include "../include/mtpParamOp.h"
#include "../include/mtpParam.h"


namespace ai2pot {
namespace mtpr {

torch::autograd::variable_list MtpParamFunction::forward(
    torch::autograd::AutogradContext *ctx,
    int mtp_level)
{
    MtpParam mtp_param(mtp_level);
    c10::TensorOptions int_options = c10::TensorOptions()
        .dtype(torch::kInt32);
    at::Tensor alpha_index_basic_tensor = at::zeros({mtp_param.alpha_index_basic_count(), 4}, int_options);
    at::Tensor alpha_index_times_tensor = at::zeros({mtp_param.alpha_index_times_count(), 4}, int_options);
    at::Tensor alpha_moment_mapping_tensor = at::zeros({mtp_param.alpha_scalar_moments()}, int_options);
    at::Tensor num_mus4moms_tensor = at::zeros({mtp_param.alpha_moments_count()}, int_options);
    at::Tensor mus4moms_tensor = at::zeros({mtp_param.alpha_moments_count(), mtp_param.max_num_mus4mom()}, int_options) - 1;
    int *alpha_index_basic = alpha_index_basic_tensor.data_ptr<int>();
    int *alpha_index_times = alpha_index_times_tensor.data_ptr<int>();
    int *alpha_moment_mapping = alpha_moment_mapping_tensor.data_ptr<int>();
    int *num_mus4moms = num_mus4moms_tensor.data_ptr<int>();
    int *mus4moms = mus4moms_tensor.data_ptr<int>();

    for (int ii=0; ii<mtp_param.alpha_index_basic_count(); ii++)
        for (int jj=0; jj<4; jj++)
            alpha_index_basic[ii*4+jj] = mtp_param.alpha_index_basic()[ii][jj];
    for (int ii=0; ii<mtp_param.alpha_index_times_count(); ii++)
        for (int jj=0; jj<4; jj++)
            alpha_index_times[ii*4+jj] = mtp_param.alpha_index_times()[ii][jj];
    for (int ii=0; ii<mtp_param.alpha_scalar_moments(); ii++)
        alpha_moment_mapping[ii] = mtp_param.alpha_moment_mapping()[ii];
    for (int ii=0; ii<mtp_param.alpha_moments_count(); ii++) {
        num_mus4moms[ii] = mtp_param.num_mus4moms()[ii];
        for (int jj=0; jj<mtp_param.num_mus4moms()[ii]; jj++)
            mus4moms[ii*mtp_param.max_num_mus4mom() + jj] = mtp_param.mus4moms_ptr()[ii*mtp_param.max_num_mus4mom() + jj];
    }
    
    return {alpha_index_basic_tensor, 
            alpha_index_times_tensor, 
            alpha_moment_mapping_tensor, 
            num_mus4moms_tensor, 
            mus4moms_tensor,
            torch::tensor(mtp_param.nmus(), int_options)};
}

torch::autograd::variable_list MtpParamFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs)
{
    return {at::Tensor()};
}

torch::autograd::variable_list MtpParamOp(int mtp_level)
{
    return MtpParamFunction::apply(mtp_level);
}

};  // namespace : mtpr
};  // namespace : ai2pot