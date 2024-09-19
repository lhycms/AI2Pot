from dpdata import LabeledSystem
from ai2pot.gst import Altbc


labeled_system = LabeledSystem("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/test_data/OUTCARs/OUTCAR")
altbc = Altbc(rcut=3.5, umax_num_neigh_atoms=100, angle_threshold=120.0)
result = altbc.analyse_labeled_system(labeled_system)
print(result.shape)
