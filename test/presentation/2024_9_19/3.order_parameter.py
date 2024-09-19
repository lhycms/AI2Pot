from pymatgen.core import Structure
from ai2pot.gst import OrderParameter


structure = Structure.from_file("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/test_data/POSCARs/POSCAR")
order_parameter = OrderParameter(rcut=3.5, umax_num_neigh_atoms=100)
result = order_parameter.analyse_structure(structure=structure)
print(result)
