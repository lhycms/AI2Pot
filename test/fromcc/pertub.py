from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import PerturbStructureTransformation

structure = Structure.from_file("/data/home/liuhanyu/mycode/AI2Pot/test/test_data/POSCARs/MoS2/POSCAR")

# 幅度 0.02 Å
transform = PerturbStructureTransformation(0.1)
new_structure = transform.apply_transformation(structure)

new_structure.to(fmt="poscar", filename="/data/home/liuhanyu/mycode/AI2Pot/test/test_data/POSCARs/MoS2/POSCAR_perturbed0.1")