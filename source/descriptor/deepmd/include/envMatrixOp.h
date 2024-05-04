#include <torch/torch.h>
#include "./envMatrix.h"


namespace matersdk {
namespace deepPotSE {

class EnvMatrixFunction : public torch::autograd::Function<EnvMatrixFunction> {
public: 
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& ilist_tensor,
        const at::Tensor& numneigh_tensor,
        const at::Tensor& firstneigh_tensor,
        const at::Tensor& relative_coords_tensor,
        const at::Tensor& types_tensor,
        const at::Tensor& umax_num_neigh_atoms_lst_tensor,
        double rcut,
        double rcut_smooth);

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs);
};  // class : EnvMatrixFunction


torch::autograd::variable_list EnvMatrixOp(
    const at::Tensor& ilist_tensor,
    const at::Tensor& numneigh_tensor,
    const at::Tensor& firstneigh_tensor,
    const at::Tensor& relative_coords_tensor,
    const at::Tensor& types_tensor,
    const at::Tensor& umax_num_neigh_atoms_lst_tensor,
    double rcut,
    double rcut_smooth);

};  // namespace : deepPotSE
};  // namespace : matersdk