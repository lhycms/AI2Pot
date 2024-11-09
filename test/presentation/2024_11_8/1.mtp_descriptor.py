import torch
from pymatgen.core import Structure
from ai2pot.core import Nblist
from ai2pot.mtp.nn_mtp import DescriptorMtp

# 1. 读取结构
structure = Structure.from_file("./POSCAR")
# 2. 获取近邻信息
nblist = Nblist(structure=structure,
                rcut=3.5,
                umax_num_neigh_atoms=100,
                pbc_xyz=[True, True, True],
                sort=True)
# 3. 初始化 MTP 模块
mtp = DescriptorMtp(mtp_level=12,
                    ntypes=4,
                    chebyshev_size=8,
                    rmax=3.5,
                    rmin=0.5,
                    umax_num_neighs=100)

# 1. 获取相对坐标的tensor
rcs = torch.tensor(nblist.rcs, 
                   dtype=torch.float32, 
                   requires_grad=True).unsqueeze(dim=0)
rcs = rcs.clone().detach().requires_grad_(True)
# 2. 计算MTP描述符
mtp_descriptor = mtp(
    torch.tensor(nblist.ilist).unsqueeze(dim=0),
    torch.tensor(nblist.numneigh).unsqueeze(dim=0),
    torch.tensor(nblist.firstneigh).unsqueeze(dim=0),
    rcs,
    torch.tensor(nblist.types).unsqueeze(dim=0))
# 3. 计算导数
mtp_descriptor.sum().backward()
# 4. 输出导数
print("MTP对原子坐标的导数: ", rcs.grad.size())
print("MTP对参数的导数: ", mtp.coeffs_list[0].grad.size())


