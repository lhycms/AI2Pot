#include <torch/torch.h>

#include "../include/virial_sr_op.h"
#include "../include/virial_sr.h"


namespace ai2pot {
namespace fvt {

torch::autograd::variable_list VirialSrFunction::forward(
    torch::autograd::AutogradContext *ctx,  
    const at::Tensor &bnumneigh_tensor,
    const at::Tensor &brcs_tensor,
    int umax_num_neighs,
    const at::Tensor &bdei_drij_tensor)
{
    assert(bnumneigh_tensor.scalar_type() == torch::kInt32);

    int nbatches = (int)bnumneigh_tensor.size(0);
    int nlocal = (int)bnumneigh_tensor.size(1);
    c10::TensorOptions int_options = c10::TensorOptions()
        .dtype(torch::kInt32)
        .device(brcs_tensor.device());
    c10::TensorOptions float_options;
    at::Tensor bvirial_val_tensor;
    at::Tensor bvirial_der1_tensor;
    at::Tensor bvirial_der2_tensor;

    if (bdei_drij_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat32)
            .device(bdei_drij_tensor.device());
        bvirial_val_tensor = at::zeros({nbatches, 3, 3}, float_options);
        bvirial_der1_tensor = at::zeros({nbatches, 3, 3, nlocal, umax_num_neighs}, float_options);
        bvirial_der2_tensor = at::zeros({nbatches, 3, 3, nlocal, umax_num_neighs}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            float *virial_val = bvirial_val_tensor[bb].data_ptr<float>();
            float *virial_der1 = bvirial_der1_tensor[bb].data_ptr<float>();
            float *virial_der2 = bvirial_der2_tensor[bb].data_ptr<float>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            float *rcs = brcs_tensor[bb].data_ptr<float>();
            float *dei_drij = bdei_drij_tensor[bb].data_ptr<float>();
            VirialSr<float>::find_val_der(
                virial_val,
                virial_der1,
                virial_der2,
                nlocal,
                numneigh,
                rcs,
                umax_num_neighs,
                dei_drij);
        }
    }
    else {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(bdei_drij_tensor.device());
        bvirial_val_tensor = at::zeros({nbatches, 3, 3}, float_options);
        bvirial_der1_tensor = at::zeros({nbatches, 3, 3, nlocal, umax_num_neighs}, float_options);
        bvirial_der2_tensor = at::zeros({nbatches, 3, 3, nlocal, umax_num_neighs}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            double *virial_val = bvirial_val_tensor[bb].data_ptr<double>();
            double *virial_der1 = bvirial_der1_tensor[bb].data_ptr<double>();
            double *virial_der2 = bvirial_der2_tensor[bb].data_ptr<double>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            double *rcs = brcs_tensor[bb].data_ptr<double>();
            double *dei_drij = bdei_drij_tensor[bb].data_ptr<double>();
            VirialSr<double>::find_val_der(
                virial_val,
                virial_der1,
                virial_der2,
                nlocal,
                numneigh,
                rcs,
                umax_num_neighs,
                dei_drij);
        }
    }

    ctx->save_for_backward({bvirial_der1_tensor, bvirial_der2_tensor, bnumneigh_tensor});
    return {bvirial_val_tensor};
}


torch::autograd::variable_list VirialSrFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs)
{
    at::Tensor bgrad_output_tensor = grad_outputs[0];
    if (!bgrad_output_tensor.is_contiguous())
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    at::Tensor bvirial_der1_tensor = ctx->get_saved_variables()[0];
    at::Tensor bvirial_der2_tensor = ctx->get_saved_variables()[1];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[2];
    int nbatches = (int)bvirial_der1_tensor.size(0);
    int nlocal = (int)bvirial_der1_tensor.size(3);
    int umax_num_neighs = (int)bvirial_der1_tensor.size(4);
    c10::TensorOptions int_options = c10::TensorOptions()
        .dtype(torch::kInt32)
        .device(bgrad_output_tensor.device());
    c10::TensorOptions float_options;
    at::Tensor bout_der1_tensor;
    at::Tensor bout_der2_tensor;

    if (bgrad_output_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat32)
            .device(bgrad_output_tensor.device());
        bout_der1_tensor = at::zeros({nbatches, nlocal, umax_num_neighs, 3}, float_options);
        bout_der2_tensor = at::zeros({nbatches, nlocal, umax_num_neighs, 3}, float_options);
        for (int bb=0; bb<nbatches; bb++) {
            float *out_der1 = bout_der1_tensor[bb].data_ptr<float>();
            float *out_der2 = bout_der2_tensor[bb].data_ptr<float>();
            float *grad_output = bgrad_output_tensor[bb].data_ptr<float>();
            float *virial_der1 = bvirial_der1_tensor[bb].data_ptr<float>();
            float *virial_der2 = bvirial_der2_tensor[bb].data_ptr<float>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            for (int ii=0; ii<nlocal; ii++) {
                for (int jj=0; jj<numneigh[ii]; jj++) {
                    for (int aa=0; aa<3; aa++) {
                        for (int bb=0; bb<3; bb++) {
                            int tmp_de_idx = ii*umax_num_neighs*3 + jj*3 + aa;
                            int tmp_rij_idx = ii*umax_num_neighs*3 + jj*3 + bb;
                            out_der1[tmp_de_idx] += grad_output[aa*3 + bb] * virial_der1[(aa*3 + bb)*nlocal*umax_num_neighs + ii*umax_num_neighs + jj];
                            out_der2[tmp_rij_idx] += grad_output[aa*3 + bb] * virial_der2[(aa*3 + bb)*nlocal*umax_num_neighs + ii*umax_num_neighs + jj];
                        }
                    }
                }
            }
        }
    } else {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(c10::kCPU);
        bout_der1_tensor = at::zeros({nbatches, nlocal, umax_num_neighs, 3}, float_options);
        bout_der2_tensor = at::zeros({nbatches, nlocal, umax_num_neighs, 3}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            double *out_der1 = bout_der1_tensor[bb].data_ptr<double>();
            double *out_der2 = bout_der2_tensor[bb].data_ptr<double>();
            double *grad_output = bgrad_output_tensor[bb].data_ptr<double>();
            double *virial_der1 = bvirial_der1_tensor[bb].data_ptr<double>();
            double *virial_der2 = bvirial_der2_tensor[bb].data_ptr<double>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            for (int ii=0; ii<nlocal; ii++) {
                for (int jj=0; jj<numneigh[ii]; jj++) {
                    for (int aa=0; aa<3; aa++) {
                        for (int bb=0; bb<3; bb++) {
                            int tmp_de_idx = ii*umax_num_neighs*3 + jj*3 + aa;
                            int tmp_rij_idx = ii*umax_num_neighs*3 + jj*3 + bb;
                            out_der1[tmp_de_idx] += grad_output[aa*3 + bb] * virial_der1[(aa*3 + bb)*nlocal*umax_num_neighs + ii*umax_num_neighs + jj];
                            out_der2[tmp_rij_idx] += grad_output[aa*3 + bb] * virial_der2[(aa*3 + bb)*nlocal*umax_num_neighs + ii*umax_num_neighs + jj];
                        }
                    }
                }
            }
        }
    }

    return {at::Tensor(), bout_der2_tensor, at::Tensor(), bout_der1_tensor};
}


torch::autograd::variable_list VirialSrOp(
    const at::Tensor &bnumneigh_tensor,
    const at::Tensor &brcs_tensor,
    int umax_num_neighs,
    const at::Tensor &bdei_drij_tensor)
{
    return VirialSrFunction::apply(bnumneigh_tensor,
                                brcs_tensor,
                                umax_num_neighs,
                                bdei_drij_tensor);
}

};  // namespace : fvt
};  // namespace : ai2pot