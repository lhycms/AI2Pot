from pymatgen.core import Structure
from ai2pot.core import Nblist


structure = Structure.from_file("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/test_data/POSCARs/POSCAR")
nblist = Nblist(structure=structure,
                rcut=3.5,
                umax_num_neigh_atoms=20,
                pbc_xyz=[True, True, True],
                sort=True)
print(nblist.firstneigh)
