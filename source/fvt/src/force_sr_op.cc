#include <torch/torch.h>

#include "../include/force_sr.h"
#include "../include/force_sr_op.h"


namespace ai2pot {
namespace fvt {

torch::autograd::variable_list ForceSrFunction::forward(
    torch::autograd::AutogradContext *ctx,
    const at::Tensor &bilist_tensor,
    const at::Tensor &bnumneigh_tensor,
    const at::Tensor &bfirstneigh_tensor,
    int nghost,
    int umax_num_neighs,
    const at::Tensor &bdei_drij_tensor)
{
    assert(bilist_tensor.dtype() == torch::kInt32);
    assert(bnumneigh_tensor.dtype() == torch::kInt32);
    assert(bfirstneigh_tensor.dtype() == torch::kInt32);

    int nbatches = (int)bilist_tensor.size(0);
    int nlocal = (int)bilist_tensor.size(1);
    c10::TensorOptions int_options = c10::TensorOptions()
        .dtype(torch::kInt32)
        .device(bdei_drij_tensor.device());
    c10::TensorOptions float_options;
    at::Tensor bforce_sr_val_tensor;
    at::Tensor bforce_sr_der_tensor;

    if (bdei_drij_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat32)
            .device(bdei_drij_tensor.device());
        bforce_sr_val_tensor = at::zeros({nbatches, nlocal + nghost, 3}, float_options);
        bforce_sr_der_tensor = at::zeros({nbatches, nlocal + nghost, 3, nlocal, umax_num_neighs, 3}, float_options);
        
        for (int bb=0; bb<nbatches; bb++) {
            float *force_sr_val = bforce_sr_val_tensor[bb].data_ptr<float>();
            float *force_sr_der = bforce_sr_der_tensor[bb].data_ptr<float>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            float *dei_drij = bdei_drij_tensor[bb].data_ptr<float>();

            ForceSr<float>::find_val_der(
                force_sr_val,
                force_sr_der,
                nlocal,
                ilist,
                numneigh,
                firstneigh,
                nghost,
                umax_num_neighs,
                dei_drij);
        }
    } else {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat64)
            .device(bdei_drij_tensor.device());
        bforce_sr_val_tensor = at::zeros({nbatches, nlocal + nghost, 3}, float_options);
        bforce_sr_der_tensor = at::zeros({nbatches, nlocal + nghost, 3, nlocal, umax_num_neighs, 3}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            double *force_sr_val = bforce_sr_val_tensor[bb].data_ptr<double>();
            double *force_sr_der = bforce_sr_der_tensor[bb].data_ptr<double>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();
            double *dei_drij = bdei_drij_tensor[bb].data_ptr<double>();

            ForceSr<double>::find_val_der(
                force_sr_val,
                force_sr_der,
                nlocal,
                ilist,
                numneigh,
                firstneigh,
                nghost,
                umax_num_neighs,
                dei_drij);
        }
    }

    ctx->save_for_backward({bforce_sr_der_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor});
    return {bforce_sr_val_tensor};
}

torch::autograd::variable_list ForceSrFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs)
{
    at::Tensor bgrad_output_tensor = grad_outputs[0];
    if (!bgrad_output_tensor.is_contiguous())
        bgrad_output_tensor = bgrad_output_tensor.contiguous();
    at::Tensor bforce_sr_der_tensor = ctx->get_saved_variables()[0];
    at::Tensor bilist_tensor = ctx->get_saved_variables()[1];
    at::Tensor bnumneigh_tensor = ctx->get_saved_variables()[2];
    at::Tensor bfirstneigh_tensor = ctx->get_saved_variables()[3];
    
    int nbatches = (int)bforce_sr_der_tensor.size(0);
    int nlocal = (int)bforce_sr_der_tensor.size(3);
    int nghost = int(bforce_sr_der_tensor.size(1) - nlocal);
    int umax_num_neighs = (int)bforce_sr_der_tensor.size(4);
    c10::TensorOptions float_options;
    torch::Tensor bout_der_tensor;

    if (bforce_sr_der_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
            .dtype(torch::kFloat32)
            .device(bforce_sr_der_tensor.device());
        bout_der_tensor = at::zeros({nbatches, nlocal, umax_num_neighs, 3}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            float *out_der = bout_der_tensor[bb].data_ptr<float>();
            float *grad_output = bgrad_output_tensor[bb].data_ptr<float>();
            float *force_sr_der = bforce_sr_der_tensor[bb].data_ptr<float>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();

            for (int ii=0; ii<nlocal; ii++) {
                int center_idx = ilist[ii];
                for (int jj=0; jj<numneigh[ii]; jj++) {
                    int neigh_idx = firstneigh[ii*umax_num_neighs + jj];
                    for (int kk=0; kk<3; kk++) {
                        int tmp_idx = ii*umax_num_neighs*3 + jj*3 + kk;
                        out_der[tmp_idx] += grad_output[center_idx*3 + kk] 
                                            * force_sr_der[(center_idx*3 + kk)*nlocal*umax_num_neighs*3 + tmp_idx];
                        out_der[tmp_idx] += grad_output[neigh_idx*3 + kk]
                                            * force_sr_der[(neigh_idx*3 + kk)*nlocal*umax_num_neighs*3 + tmp_idx];
                    }
                }
            }

        }
    } else {
        float_options = c10::TensorOptions()
            .dtype(bforce_sr_der_tensor.scalar_type())
            .device(bforce_sr_der_tensor.device());
        bout_der_tensor = at::zeros({nbatches, nlocal, umax_num_neighs, 3}, float_options);

        for (int bb=0; bb<nbatches; bb++) {
            double *out_der = bout_der_tensor[bb].data_ptr<double>();
            double *grad_output = bgrad_output_tensor[bb].data_ptr<double>();
            double *force_sr_der = bforce_sr_der_tensor[bb].data_ptr<double>();
            int *ilist = bilist_tensor[bb].data_ptr<int>();
            int *numneigh = bnumneigh_tensor[bb].data_ptr<int>();
            int *firstneigh = bfirstneigh_tensor[bb].data_ptr<int>();

            for (int ii=0; ii<nlocal; ii++) {
                int center_idx = ilist[ii];
                for (int jj=0; jj<numneigh[ii]; jj++) {
                    int neigh_idx = firstneigh[ii*umax_num_neighs + jj];
                    for (int kk=0; kk<3; kk++) {
                        int tmp_idx = ii*umax_num_neighs*3 + jj*3 + kk;
                        out_der[tmp_idx] += grad_output[center_idx*3 + kk]
                                            * force_sr_der[(center_idx*3 + kk)*nlocal*umax_num_neighs*3 + tmp_idx];
                        out_der[tmp_idx] += grad_output[neigh_idx*3 + kk]
                                            * force_sr_der[(neigh_idx*3 + kk)*nlocal*umax_num_neighs*3 + tmp_idx];
                    }
                }
            }
        }
    }
    return {at::Tensor(), at::Tensor(), at::Tensor(), 
            at::Tensor(), at::Tensor(), bout_der_tensor};
}


torch::autograd::variable_list ForceSrOp(
    const at::Tensor& bilist_tensor,
    const at::Tensor& bnumneigh_tensor,
    const at::Tensor& bfirstneigh_tensor,
    int nghost,
    int umax_num_neighs,
    const at::Tensor& bdei_drij_tensor)
{
    return ForceSrFunction::apply(
        bilist_tensor,
        bnumneigh_tensor,
        bfirstneigh_tensor,
        nghost,
        umax_num_neighs,
        bdei_drij_tensor);
}

};  // class : fvt
};  // class : ai2pot