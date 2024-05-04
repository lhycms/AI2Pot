#include <stdexcept>
#include "../include/envMatrixOp.h"


namespace matersdk {
namespace deepPotSE {

/**
 * @brief Process structure batch by batch. 
 *        Note: `batch_size` maybe larger than 1.
 * 
 * @param ctx torch::autograd::AutogradContext* 
 * @param ilist_tensor .size() = (batch_size, num_atoms)
 * @param numneigh_tensor .size() = (batch_size, num_atoms)
 * @param firstneigh_tensor .size() = (batch_size, num_atoms, umax_num_neigh_atoms)
 * @param relative_coords_tensor .size() = (batch_size, num_atoms, umax_num_neigh_atoms, 3)
 * @param types_tensor .size() = (batch_size, num_atoms + num_ghost)
 * @param umax_num_neigh_atoms_lst_tensor .size() = (ntypes)
 * @param rcut 
 * @param rcut_smooth 
 * @return torch::autograd::variable_list 
 */
torch::autograd::variable_list EnvMatrixFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& ilist_tensor,
    const at::Tensor& numneigh_tensor,
    const at::Tensor& firstneigh_tensor,
    const at::Tensor& relative_coords_tensor,  // float
    const at::Tensor& types_tensor,
    const at::Tensor& umax_num_neigh_atoms_lst_tensor,
    double rcut,
    double rcut_smooth)
{   
    // Step 1. 
    // Step 1.1. Some assert
    assert(ilist_tensor.scalar_type() == torch::kInt32);
    assert(numneigh_tensor.scalar_type() == torch::kInt32);
    assert(firstneigh_tensor.scalar_type() == torch::kInt32);
    assert(types_tensor.scalar_type() == torch::kInt32);
    assert(umax_num_neigh_atoms_lst_tensor.scalar_type() == torch::kInt32);
    assert(
        (relative_coords_tensor.scalar_type() == torch::kFloat32) 
        | (relative_coords_tensor.scalar_type() == torch::kFloat64));
    // Step 1.2. ctx->save_for_backward({...})
    // Setp 1.3. Get `batch_size`, ...
    int batch_size = (int)ilist_tensor.size(0);     // long -> int
    int inum = (int)ilist_tensor.size(1);           // long -> int
    int umax_num_neigh_atoms = umax_num_neigh_atoms_lst_tensor.sum().item<int>();
    int ntypes = (int)umax_num_neigh_atoms_lst_tensor.size(0);  // long -> int
    
    // Step 2. Init return variables
    c10::Device device = relative_coords_tensor.device();
    c10::TensorOptions float_options;
    at::Tensor tilde_r_tensor;
    at::Tensor tilde_r_deriv_tensor;
    if (relative_coords_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions().dtype(torch::kFloat32).device(device);
        tilde_r_tensor = at::zeros(
            {batch_size, inum, umax_num_neigh_atoms, 4}, 
            float_options);
        tilde_r_deriv_tensor = at::zeros(
            {batch_size, inum, umax_num_neigh_atoms, 4, 3},
            float_options);
    } else {
        float_options = c10::TensorOptions().dtype(torch::kFloat64).device(device);
        tilde_r_tensor = at::zeros(
            {batch_size, inum, umax_num_neigh_atoms, 4},
            float_options);
        tilde_r_deriv_tensor = at::zeros(
            {batch_size, inum, umax_num_neigh_atoms, 4, 3},
            float_options);
    }

    // Step 3. find tilde_r_tensor, tilde_r_deriv_tensor
    for (int ii=0; ii<batch_size; ii++) {
        int* ilist = ilist_tensor[ii].data_ptr<int>();
        int* numneigh = numneigh_tensor[ii].data_ptr<int>();
        int* firstneigh = firstneigh_tensor[ii].data_ptr<int>();
        int* types = types_tensor[ii].data_ptr<int>();
        int* umax_num_neigh_atoms_lst = umax_num_neigh_atoms_lst_tensor.data_ptr<int>();
        
        if (relative_coords_tensor.scalar_type() == torch::kFloat32) {
            float* tilde_r = tilde_r_tensor[ii].data_ptr<float>();
            float* tilde_r_deriv = tilde_r_deriv_tensor[ii].data_ptr<float>();
            float* relative_coords = relative_coords_tensor[ii].data_ptr<float>();
            EnvMatrix<float>::find_value_deriv(
                tilde_r,
                tilde_r_deriv,
                inum,
                ilist,
                numneigh,
                firstneigh,
                relative_coords,
                types,
                ntypes,
                umax_num_neigh_atoms_lst,
                rcut,
                rcut_smooth);            
        } else {
            double* tilde_r = tilde_r_tensor[ii].data_ptr<double>();
            double* tilde_r_deriv = tilde_r_deriv_tensor[ii].data_ptr<double>();
            double* relative_coords = relative_coords_tensor[ii].data_ptr<double>();
            EnvMatrix<double>::find_value_deriv(
                tilde_r,
                tilde_r_deriv,
                inum,
                ilist,
                numneigh,
                firstneigh,
                relative_coords,
                types,
                ntypes,
                umax_num_neigh_atoms_lst,
                rcut,
                rcut_smooth);
        }
    }

    ctx->save_for_backward({tilde_r_deriv_tensor});
    
    return {tilde_r_tensor};
}


torch::autograd::variable_list EnvMatrixFunction::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs)
{
    at::Tensor grad_output_tensor = grad_outputs[0];
    if (!grad_output_tensor.is_contiguous()) 
        grad_output_tensor = grad_output_tensor.contiguous();
    at::Tensor tilde_r_deriv_tensor = ctx->get_saved_variables()[0];
    int nbatches = (int)tilde_r_deriv_tensor.size(0);
    int natoms = (int)tilde_r_deriv_tensor.size(1);
    int umax_neigh_atoms = (int)tilde_r_deriv_tensor.size(2);
    
    c10::Device device = tilde_r_deriv_tensor.device();
    c10::TensorOptions float_options;
    at::Tensor grad_output_deriv_tensor;
    if (tilde_r_deriv_tensor.scalar_type() == torch::kFloat32) {
        float_options = c10::TensorOptions()
            .device(device)
            .dtype(torch::kFloat32);
        grad_output_deriv_tensor = at::zeros({nbatches, natoms, umax_neigh_atoms, 3}, float_options);
        float *grad_output_deriv = grad_output_deriv_tensor.data_ptr<float>();
        float *grad_output = grad_output_tensor.data_ptr<float>();
        float *tilde_r_deriv = tilde_r_deriv_tensor.data_ptr<float>();
        for (int ii=0; ii<nbatches; ii++) 
            for (int jj=0; jj<natoms; jj++) 
                for (int kk=0; kk<umax_neigh_atoms; kk++) 
                    for (int ll=0; ll<4; ll++) 
                        for (int mm=0; mm<3; mm++) 
                            grad_output_deriv[ii*natoms*umax_neigh_atoms*3 + jj*umax_neigh_atoms*3 + kk*3 + mm] = 
                                grad_output[ii*natoms*umax_neigh_atoms*4 + jj*umax_neigh_atoms*4 + kk*4 + ll]
                                * tilde_r_deriv[ii*natoms*umax_neigh_atoms*4*3 + jj*umax_neigh_atoms*4*3 + kk*4*3 + ll*3 + mm];
    } else {
        float_options = c10::TensorOptions()
            .device(device)
            .dtype(torch::kFloat64);
        grad_output_deriv_tensor = at::zeros({nbatches, natoms, umax_neigh_atoms, 3}, float_options);
        double *grad_output_deriv = grad_output_deriv_tensor.data_ptr<double>();
        double *grad_output = grad_output_tensor.data_ptr<double>();
        double *tilde_r_deriv = tilde_r_deriv_tensor.data_ptr<double>();
        for (int ii=0; ii<nbatches; ii++)
            for (int jj=0; jj<natoms; jj++)
                for (int kk=0; kk<umax_neigh_atoms; kk++)
                    for (int ll=0; ll<4; ll++)
                        for (int mm=0; mm<3; mm++)
                            grad_output_deriv[ii*natoms*umax_neigh_atoms*3 + jj*umax_neigh_atoms*3 + kk*3 + mm] = 
                                grad_output[ii*natoms*umax_neigh_atoms*4 + jj*umax_neigh_atoms*4 + kk*4 + ll]
                                * tilde_r_deriv[ii*natoms*umax_neigh_atoms*4*3 + jj*umax_neigh_atoms*4*3 + kk*4*3 + ll*3 + mm];
    }

    return {at::Tensor(), at::Tensor(), at::Tensor(),
            grad_output_deriv_tensor, at::Tensor(), at::Tensor(),
            at::Tensor(), at::Tensor()};
}


torch::autograd::variable_list EnvMatrixOp(
    const at::Tensor& ilist_tensor,
    const at::Tensor& numneigh_tensor,
    const at::Tensor& firstneigh_tensor,
    const at::Tensor& relative_coords_tensor,
    const at::Tensor& types_tensor,
    const at::Tensor& umax_num_neigh_atoms_lst_tensor,
    double rcut,
    double rcut_smooth)
{
    return EnvMatrixFunction::apply(
        ilist_tensor,
        numneigh_tensor,
        firstneigh_tensor,
        relative_coords_tensor,
        types_tensor,
        umax_num_neigh_atoms_lst_tensor,
        rcut,
        rcut_smooth);
}

};  // namespace : deepPotSE
};  // namespace : matersdk
