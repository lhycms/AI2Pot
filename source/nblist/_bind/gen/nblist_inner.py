from pymatgen.core import Structure
import numpy as np
import time
import copy


# lattice, frac_coords still lose accuracy due to unclear reasons.
poscar_path = "/data/home/liuhanyu/hyliu/pwmat_demo/MoS2/scf_/POSCAR"
structure = Structure.from_file(poscar_path)    #.make_supercell([3, 4, 1])
lattice = structure.lattice.matrix
lattice = copy.deepcopy(lattice)
#for ii in range(3):
#    for jj in range(3):
#        print(lattice[ii][jj], end=", ")
#    print("\n")
species = np.array([tmp_specie.Z for tmp_specie in structure.species])
species = np.where(species==42, 0, 1)
frac_coords = structure.frac_coords
rcut = 3.3
rcut_smooth = 3.0
pbc_xyz = [True, True, False]
umax_num_neigh_atoms = 19
umax_num_neigh_atoms_lst = np.array([10, 9]).astype(np.int32)
sort = True
is_cart_coord = False


import nblist
print(nblist)
start = time.time()    
info = nblist.find_info4mlff(
    lattice.astype(np.float64),
    species.astype(np.int32),
    frac_coords.astype(np.float64),
    rcut,
    pbc_xyz,
    umax_num_neigh_atoms,
    sort,
    is_cart_coord)
end = time.time()
print("Consuming time in seconds: ", end-start)

inum:int = info[0]
ilist:np.array = info[1]
numneigh:np.array = info[2]
firstneigh:np.array = info[3]
relative_coords:np.array = info[4]
types:np.array = info[5]
nghost:int = info[6]
#print(inum)
#print(ilist)
#print(numneigh)
#print(firstneigh)
#print(relative_coords)
#print(types)
#print(nghost)

