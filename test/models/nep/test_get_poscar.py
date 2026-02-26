from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write


atoms_list = ase_read("/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz", index=":")
ase_write("/data/home/liuhanyu/mycode/AI2Pot/test/test_data/POSCARs/C/POSCAR", atoms_list[4])