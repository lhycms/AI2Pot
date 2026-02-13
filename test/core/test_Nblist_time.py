from pymatgen.core import Structure
from ai2pot.core import Nblist
import time

t1 = time.time()
structure = Structure.from_file("/data/home/liuhanyu/mycode/AI2Pot/test/test_data/POSCARs/POSCAR")
structure = structure.make_supercell([10, 10, 1])
t2 = time.time()
print(len(structure))
nblist = Nblist.from_pymatgen(structure, 3.2, umax_num_neigh_atoms=200)
t3 = time.time()
print(t2 - t1)
print(t3 - t2)
