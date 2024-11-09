from dpdata import LabeledSystem
from ai2pot.gst import Altbc


# 1. 初始化 LabeledSystem
ls = LabeledSystem("./OUTCAR")
# 2. 初始化 ALTBC 对象
altbc = Altbc(rcut=3.5, 
              umax_num_neigh_atoms=100, 
              angle_threshold=120.0)
# 3. 计算体系的 altbc
result = altbc.analyse_labeled_system(
              labeled_system=ls)


