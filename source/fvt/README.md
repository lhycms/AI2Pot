# 1. Keys for `ForceSrFunction`
<font color="FF0066" size="5">

1. Parameters:
   1. `const at::Tensor &binum_tensor`
   2. `const at::Tensor &bilist_tensor`
   3. `const at::Tensor &bnumneigh_tensor`
   4. `const at::Tensor &bfirstneigh_tensor`
   5. `int nghost` (`torch.int32`)
      1. 训练时，`同一个 batch` 中的 frames 均满足 `nghost=0`
      2. Lammps 运行 MD 时，`nghost`不必等于`0`
   6. `int umax_num_neighs` (`torch.int32`)
   7. `const at::Tensor &bdei_drij_tensor`

</font>


# 2. Keys for `VirialSrFunction`
<font color="FF0066" size="5">

1. 

</font>
