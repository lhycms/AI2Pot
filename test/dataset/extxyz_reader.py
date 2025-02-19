import os
from ase.io import read as ase_read

TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")

# 1. get energy -> Atoms.get_potential_energy()
extxyz_path = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")
obj_e = ase_read(extxyz_path, index=":")
print("1. Energy of obj_e[0] = ", obj_e[0].get_potential_energy())

# 2. get forces -> Atoms.get_forces()   `sed "s/force/forces/g" train.xyz > train_m.xyz`
extxyz_path = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")
obj_f = ase_read(extxyz_path, index=":")
print("2. Force of obj_f[0] = ", obj_f[0].get_forces())

# 3. get dipole -> Atoms.calc.results
extxyz_path = os.path.join(TEST_FILES_DIR, "XYZ", "12_NEP_dipole_QM7B", "train_m.xyz")
obj_d = ase_read(extxyz_path, index=":")
print("3. Dipole of obj_d[0] = ", obj_d[0].calc.results)

# 4. get pol -> Atoms.info
extxyz_path = os.path.join(TEST_FILES_DIR, "XYZ", "12_NEP_dipole_QM7B", "train_m.xyz")
obj_p = ase_read(extxyz_path, index=":")
print("4. Polarizalibity of obj_p[0] = ", obj_p[0].info)
