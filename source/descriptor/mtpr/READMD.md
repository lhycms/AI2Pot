# 1. Input tensor size for `MtpBasisOp`
1. `binum_tensor`: [batch_size,]
2. `bilist_tensor`: [batch_size, natoms]
3. `bnumneigh_tensor`: [batch_size, natoms]
4. `bfirstneigh_tensor`: [batch_size, natoms, umax_num_neigh_atoms]
5. `brcs_tensor`: [batch_size, natoms, umax_num_neigh_atoms, 3]
6. `btypes_tensor`: [batch_size, natoms+nghost]


## 1.1. `torch.utils.data.DataLoader`
1. `inum_tensor`: []
2. `ilist_tensor`: [natoms]
3. `numneigh_tensor`: [natoms]
4. `firstneigh_tensor`: [natoms, umax_num_neigh_atoms]
5. `rcs_tensor`: [natoms, umax_num_neigh_atoms, 3]
6. `types_tensor`: [natoms+nghost]


## 1.2. `aitpot.utils.MlffInput`
1. `binum_tensor`: [1]
2. `bilist_tensor`: [1, natoms]
3. `bnumneigh_tensor`: [1, natoms]
4. `bfirstneugh_tensor`: [1, natoms, umax_num_neigh_atoms]
5. `brcs_tensor`: [1, natoms, umax_num_neigh_atoms, 3]
6. `types_tensor`: [1, natoms+nghost]
