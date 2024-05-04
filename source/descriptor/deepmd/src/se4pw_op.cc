#include <torch/torch.h>
#include <cstdlib>
#include <stdio.h>
#include "../include/se4pw_op.h"


namespace matersdk {
namespace deepPotSE {


/**
 * @brief 
 * 
 * @note It doesn't matter whether input tensor is flatten or not.
 *  1. int: torch::kInt32
 *  2. float: torch::kFloat32, torch::kFloat64
 * 
 * @param inum 
 * @param ilist 
 * @param numneigh 
 * @param firstneigh 
 * @param x 
 * @param types 
 * @param ntypes 
 * @param num_neigh_atoms_lst 
 * @param rcut 
 * @param rcut_smooth 
 * @return torch::autograd::variable_list 
 */
torch::autograd::variable_list Se4pwOp::forward(
        int batch_size,
        int inum,
        at::Tensor& ilist,
        at::Tensor& numneigh,
        at::Tensor& firstneigh,
        at::Tensor& x,
        at::Tensor& types,
        int ntypes,
        at::Tensor& num_neigh_atoms_lst,
        double rcut,
        double rcut_smooth)
{
    c10::Device device = x.device();
    c10::ScalarType dtype = x.scalar_type();
    c10::TensorOptions tensor_options = c10::TensorOptions().device(device).dtype(dtype);
    c10::ScalarType int_dtype = ilist.scalar_type();
    
    ilist.to(int_dtype);
    numneigh.to(int_dtype);
    firstneigh.to(int_dtype);
    types.to(int_dtype);
    num_neigh_atoms_lst.to(int_dtype);
    
    int tot_num_neigh_atoms = num_neigh_atoms_lst.sum().item<int>();
    at::Tensor tilde_r = at::zeros({batch_size, inum, tot_num_neigh_atoms, 4}, tensor_options);
    at::Tensor tilde_r_deriv = at::zeros({batch_size, inum, tot_num_neigh_atoms, 4, 3}, tensor_options);
    at::Tensor relative_coords = at::zeros({batch_size, inum, tot_num_neigh_atoms, 3}, tensor_options);

    if (dtype == torch::kFloat32) {
        for (int ii=0; ii<batch_size; ii++) {
            float* tilde_r_ptr = tilde_r[ii].data_ptr<float>();
            float* tilde_r_deriv_ptr = tilde_r_deriv[ii].data_ptr<float>();
            float* relative_coords_ptr = relative_coords[ii].data_ptr<float>();

            Se4pw<float>::generate(
                    tilde_r_ptr,
                    tilde_r_deriv_ptr,
                    relative_coords_ptr,
                    inum,
                    ilist[ii].data_ptr<int>(),
                    numneigh[ii].data_ptr<int>(),
                    firstneigh[ii].data_ptr<int>(),
                    x[ii].data_ptr<float>(),
                    types[ii].data_ptr<int>(),
                    ntypes,
                    num_neigh_atoms_lst.data_ptr<int>(),
                    (float)rcut,
                    (float)rcut_smooth);
        }
    } else {
        for (int ii=0; ii<batch_size; ii++) {   // loop over all images in this batch
            double* tilde_r_ptr = tilde_r[ii].data_ptr<double>();
            double* tilde_r_deriv_ptr = tilde_r_deriv[ii].data_ptr<double>();
            double* relative_coords_ptr = relative_coords[ii].data_ptr<double>();

            Se4pw<double>::generate(
                    tilde_r_ptr,
                    tilde_r_deriv_ptr,
                    relative_coords_ptr,
                    inum,
                    ilist[ii].data_ptr<int>(),
                    numneigh[ii].data_ptr<int>(),
                    firstneigh[ii].data_ptr<int>(),
                    x[ii].data_ptr<double>(),
                    types[ii].data_ptr<int>(),
                    ntypes,
                    num_neigh_atoms_lst.data_ptr<int>(),
                    rcut,
                    rcut_smooth);
        }
    }


    return {tilde_r, tilde_r_deriv, relative_coords};
}



at::Tensor Se4pwOp::get_prim_indices_from_matersdk(
        int batch_size,
        int inum,
        at::Tensor& ilist,
        at::Tensor& numneigh,
        at::Tensor& firstneigh,
        at::Tensor& types,
        int ntypes,
        at::Tensor& num_neigh_atoms_lst)
{
    c10::Device device = ilist.device();
    c10::ScalarType int_dtype = ilist.scalar_type();

    ilist.to(int_dtype);
    numneigh.to(int_dtype);
    firstneigh.to(int_dtype);
    types.to(int_dtype);
    num_neigh_atoms_lst.to(int_dtype);

    int tot_num_neigh_atoms = num_neigh_atoms_lst.sum().item<int>();
    at::Tensor prim_indices = at::zeros({batch_size, inum, tot_num_neigh_atoms}).to(int_dtype);

    for (int ii=0; ii<batch_size; ii++) {
        int* prim_indices_ptr = prim_indices[ii].data_ptr<int>();
        Se4pw<double>::get_prim_indices_from_matersdk(
            prim_indices_ptr,
            inum,
            ilist[ii].data_ptr<int>(),
            numneigh[ii].data_ptr<int>(),
            firstneigh[ii].data_ptr<int>(),
            types[ii].data_ptr<int>(),
            ntypes,
            num_neigh_atoms_lst.data_ptr<int>());
    }

    return prim_indices;
}


};  // namespace : deepPotSE 
};  // namespace : matersdk