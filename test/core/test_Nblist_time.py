from pymatgen.core import Structure
from ai2pot.core import Nblist
import time

t1 = time.time()
structure = Structure.from_file("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/test_data/POSCARs/POSCAR")
structure = structure * 4 * 4
t2 = time.time()
print(len(structure))
nblist = Nblist(structure, 3.2, umax_num_neigh_atoms=6)
t3 = time.time()
print(t2 - t1)
print(t3 - t2)
