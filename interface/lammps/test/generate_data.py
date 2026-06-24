from ase.io import read
from ase.data import atomic_masses, atomic_numbers


def write_lammps_atomic_data_from_extxyz(
    extxyz_path: str,
    data_path: str,
    index: int = 0,
    type_order=None,
):
    """
    Convert one structure from extxyz to LAMMPS data file with atom_style atomic.

    Parameters
    ----------
    extxyz_path
        Input extxyz file.
    data_path
        Output LAMMPS data file.
    index
        Structure index in extxyz. 0 means first structure.
    type_order
        Element order for LAMMPS atom types, e.g. ["Ge", "Sb", "Te"].
        If None, use sorted unique symbols by atomic number.
    """

    atoms = read(extxyz_path, index=index)

    symbols = atoms.get_chemical_symbols()

    if type_order is None:
        unique_symbols = sorted(set(symbols), key=lambda s: atomic_numbers[s])
    else:
        unique_symbols = list(type_order)

    symbol_to_type = {
        symbol: i + 1 for i, symbol in enumerate(unique_symbols)
    }

    positions = atoms.get_positions()
    cell = atoms.get_cell()

    # Only handle orthogonal cells here
    lx = cell[0, 0]
    ly = cell[1, 1]
    lz = cell[2, 2]

    if abs(cell[0, 1]) > 1e-10 or abs(cell[0, 2]) > 1e-10 or \
       abs(cell[1, 0]) > 1e-10 or abs(cell[1, 2]) > 1e-10 or \
       abs(cell[2, 0]) > 1e-10 or abs(cell[2, 1]) > 1e-10:
        raise ValueError("This simple writer only supports orthogonal cells.")

    with open(data_path, "w") as f:
        f.write("LAMMPS data file generated from extxyz by ASE\n\n")

        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(unique_symbols)} atom types\n\n")

        f.write(f"0.0 {lx:.16f} xlo xhi\n")
        f.write(f"0.0 {ly:.16f} ylo yhi\n")
        f.write(f"0.0 {lz:.16f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for i, symbol in enumerate(unique_symbols, start=1):
            mass = atomic_masses[atomic_numbers[symbol]]
            f.write(f"{i} {mass:.6f} # {symbol}\n")

        f.write("\nAtoms # atomic\n\n")
        for atom_id, (symbol, pos) in enumerate(zip(symbols, positions), start=1):
            atom_type = symbol_to_type[symbol]
            x, y, z = pos
            f.write(f"{atom_id} {atom_type} {x:.16f} {y:.16f} {z:.16f}\n")


if __name__ == "__main__":
    write_lammps_atomic_data_from_extxyz(
        extxyz_path="/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/train.xyz",
        data_path="data.gst",
        index=2000,
        type_order=["Ge", "Sb", "Te"],
    )