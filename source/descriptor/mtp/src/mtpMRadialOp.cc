/*
    Copyright 2025 Hanyu Liu
    This file is part of AI2Pot.
    AI2Pot is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    AI2Pot is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <torch/torch.h>
#include <vector>
#include <iterator>
#include "../include/mtpMRadial.h"
#include "../include/mtpMRadialOp.h"


namespace ai2pot {
namespace mtp {

torch::autograd::variable_list MtpQFunction::forward(
    torch::autograd::AutogradContext* ctx,
    int64_t size,
    const at::Tensor& rcuts_tensor,
    const at::Tensor& rcs_tensor)
{
    c10::TensorOptions options = c10::TensorOptions()
        .dtype(rcs_tensor.scalar_type())
        .device(rcs_tensor.device());
    at::IntArrayRef bcn_shape = rcs_tensor.sizes();   // .shape = [nneighs, 3]
    int64_t nneighs = rcs_tensor.sizes()[0];
    std::vector<int64_t> result_shape = {nneighs, size};
    std::vector<int64_t> deriv2xyz_shape = {nneighs, size, 3};
    at::Tensor result_tensor = at::zeros(result_shape, options);
    at::Tensor deriv2xyz_tensor = at::zeros(deriv2xyz_shape, options);
    if (options.dtype() == torch::kFloat32) {
        float* rcs = rcs_tensor.data_ptr<float>();
        MtpQ<float> mtp_q(
            size,
            rcuts_tensor[0].item<float>(),
            rcuts_tensor[1].item<float>());
        float* result = result_tensor.data_ptr<float>();
        float* deriv2xyz = deriv2xyz_tensor.data_ptr<float>();
        float distance;
        for (int ii=0; ii<(int)nneighs; ii++) {
            distance = std::sqrt(
                std::pow(rcs[ii*3+0], 2) + 
                std::pow(rcs[ii*3+1], 2) + 
                std::pow(rcs[ii*3+2], 2)
            );
            mtp_q.build(distance);
            for (int jj=0; jj<size; jj++) {
                result[ii*size + jj] = mtp_q.get_result()[jj];
                if (distance == 0) {
                    deriv2xyz[ii*size*3 + jj*3 + 0] = 0;
                    deriv2xyz[ii*size*3 + jj*3 + 1] = 0;
                    deriv2xyz[ii*size*3 + jj*3 + 2] = 0;
                } else {
                    deriv2xyz[ii*size*3 + jj*3 + 0] = mtp_q.get_deriv2r()[jj] * rcs[ii*3+0] / distance;
                    deriv2xyz[ii*size*3 + jj*3 + 1] = mtp_q.get_deriv2r()[jj] * rcs[ii*3+1] / distance;
                    deriv2xyz[ii*size*3 + jj*3 + 2] = mtp_q.get_deriv2r()[jj] * rcs[ii*3+2] / distance;
                }
            }
        }
        
    } else {
        double* rcs = rcs_tensor.data_ptr<double>();
        MtpQ<double> mtp_q(
            size,
            rcuts_tensor[0].item<double>(),
            rcuts_tensor[1].item<double>());
        double* result = result_tensor.data_ptr<double>();
        double* deriv2xyz = deriv2xyz_tensor.data_ptr<double>();
        double distance;
        for (int ii=0; ii<(int)nneighs; ii++) {
            distance = std::sqrt(
                std::pow(rcs[ii*3+0], 2) + 
                std::pow(rcs[ii*3+1], 2) + 
                std::pow(rcs[ii*3+2], 2)
            );
            mtp_q.build(distance);
            for (int jj=0; jj<size; jj++) {
                result[ii*size + jj] = mtp_q.get_result()[jj];
                if (distance == 0) {
                    deriv2xyz[ii*size*3 + jj*3 + 0] = 0;
                    deriv2xyz[ii*size*3 + jj*3 + 1] = 0;
                    deriv2xyz[ii*size*3 + jj*3 + 2] = 0;
                } else {
                    deriv2xyz[ii*size*3 + jj*3 + 0] = mtp_q.get_deriv2r()[jj] * rcs[ii*3+0] / distance;
                    deriv2xyz[ii*size*3 + jj*3 + 1] = mtp_q.get_deriv2r()[jj] * rcs[ii*3+1] / distance;
                    deriv2xyz[ii*size*3 + jj*3 + 2] = mtp_q.get_deriv2r()[jj] * rcs[ii*3+2] / distance;
                }
            }
        }
        
    }

    ctx->save_for_backward({deriv2xyz_tensor});
    return {result_tensor};
}


torch::autograd::variable_list MtpQFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs)
{
    at::Tensor deriv2xyz_tensor = ctx->get_saved_variables()[0];
    at::Tensor grad_output_tensor = grad_outputs[0];
    if (!grad_output_tensor.is_contiguous()) {
        grad_output_tensor = grad_output_tensor.contiguous();
    }
    assert(deriv2xyz_tensor.scalar_type() == grad_output_tensor.scalar_type());

    c10::TensorOptions options = c10::TensorOptions()
        .dtype(deriv2xyz_tensor.scalar_type())
        .device(deriv2xyz_tensor.device());
    at::Tensor deriv_tensor = at::zeros({deriv2xyz_tensor.sizes()[0], 3}, options);
    int64_t num_neighs = deriv2xyz_tensor.sizes()[0]; // nneighs
    int64_t size = deriv2xyz_tensor.sizes()[1];       // size
    if (options.dtype() == torch::kFloat32) {
        float* deriv2xyz = deriv2xyz_tensor.data_ptr<float>();
        float* grad_output = grad_output_tensor.data_ptr<float>();
        float* deriv = deriv_tensor.data_ptr<float>();
        for (int ii=0; ii<(int)num_neighs; ii++) {
            for (int jj=0; jj<(int)size; jj++) {
                deriv[ii*3 + 0] += grad_output[ii*size + jj] * deriv2xyz[ii*size*3 + jj*3 + 0];
                deriv[ii*3 + 1] += grad_output[ii*size + jj] * deriv2xyz[ii*size*3 + jj*3 + 1];
                deriv[ii*3 + 2] += grad_output[ii*size + jj] * deriv2xyz[ii*size*3 + jj*3 + 2];
            }
        }
    } else {
        double* deriv2xyz = deriv2xyz_tensor.data_ptr<double>();
        double* grad_output = grad_output_tensor.data_ptr<double>();
        double* deriv = deriv_tensor.data_ptr<double>();
        for (int ii=0; ii<(int)num_neighs; ii++) {
            for (int jj=0; jj<(int)size; jj++) {
                deriv[ii*3 + 0] += grad_output[ii*size + jj] * deriv2xyz[ii*size*3 + jj*3 + 0];
                deriv[ii*3 + 1] += grad_output[ii*size + jj] * deriv2xyz[ii*size*3 + jj*3 + 1];
                deriv[ii*3 + 2] += grad_output[ii*size + jj] * deriv2xyz[ii*size*3 + jj*3 + 2];
            }
        }
    }
    return {at::Tensor(), at::Tensor(), deriv_tensor};
}


torch::autograd::variable_list MtpQOp(
    int64_t size,
    const at::Tensor& rcuts_tensor,
    const at::Tensor& rcs_tensor)
{
    return MtpQFunction::apply(size, rcuts_tensor, rcs_tensor);
}

};  // class : mtp
};  // class : ai2pot
