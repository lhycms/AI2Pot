from typing import List
import numpy as np
from dpdata import LabeledSystem, MultiSystems


def svd_solve(A, b):
    U, S, V = np.linalg.svd(A)
    B = np.matmul(U.T, b)
    X = B[:len(S), :] / S.reshape(len(S), -1)
    x = np.matmul(V.T, X)
    return x


class LsShifter(object):
    def __init__(self, ls: LabeledSystem):
        self.nframes: int = len(ls)
        self.ntypes: List[int] = ls["atom_types"]
        self.type_names: List[str] = ls["atom_names"]
        self.types_energy_shifts, self.systems_shifted_energy = LsShifter.get_energy_shifts(ls)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        print("LsShifter Summary:")
        print("------------------")
        print("1. nframes = ", self.nframes)
        print("2. type_name = ", self.type_names)
        print("3. types energy shifts:\n", self.types_energy_shifts)
        print("4. systems shifted energy:\n", self.systems_shifted_energy)
        return ""
    
    @staticmethod
    def get_energy_shifts(ls: LabeledSystem):
        unique_atom_types: np.ndarray = np.unique(ls["atom_types"])
        nframes: int = len(ls)
        ntypes: int = unique_atom_types.shape[0]
        coeff_mtx: np.ndarray = np.zeros((nframes, ntypes))
        energy_mtx: np.ndarray = np.zeros((nframes, 1))

        for ii in range(nframes):
            for jj in range(ntypes):
                coeff_mtx[ii][jj] = ls["atom_numbs"][jj]
            energy_mtx[ii][0] = ls["energies"][ii]
        
        if (np.linalg.matrix_rank(coeff_mtx) < ntypes):
            import itertools
            for type_pair in itertools.combinations(unique_atom_types, 2):
                add_mtx: np.ndarray = np.zeros(ntypes)
                add_mtx[type_pair[0]] = 1
                add_mtx[type_pair[1]] = -1
                add_energy = np.zeros(1)
                coeff_mtx = np.r_[coeff_mtx, [add_mtx]]
                energy_mtx = np.r_[energy_mtx, [add_energy]]
        types_energy_shifts: np.ndarray = svd_solve(coeff_mtx, energy_mtx)
        systems_shifted_energy: np.ndarray = (energy_mtx - np.matmul(coeff_mtx, types_energy_shifts)).flatten()
        return types_energy_shifts.flatten(), systems_shifted_energy
        

class MsShifter(object):
    def __init__(self, ms: MultiSystems):
        self.nsys: int = len(ms)
        self.nframes: int = ms.get_nframes()
        self.ntypes: int = np.unique([ls["atom_types"] for ls in ms]).shape[0]
        self.type_names: List[str] = []
        for ls in ms:
            for ele in ls["atom_names"]:
                if ele not in self.type_names:
                    self.type_names.append(ele)
        self.types_energy_shifts, self.systems_shifted_energy = MsShifter.get_energy_shifts(ms=ms)
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        print("LsShifter Summary:")
        print("------------------")
        print("1. nframes = ", self.nframes)
        print("2. type_name = ", self.type_names)
        print("3. types energy shifts:\n", self.types_energy_shifts)
        print("4. systems shifted energy:\n", self.systems_shifted_energy)
        return ""
            
    @staticmethod
    def get_energy_shifts(ms: MultiSystems):
        ntypes: int = np.unique([ls["atom_types"] for ls in ms]).shape[0]
        unique_atom_types: np.ndarray = np.arange(ntypes)
        nframes: int = ms.get_nframes()
        
        coeff_mtx: np.ndarray = np.zeros((nframes, ntypes))
        energy_mtx: np.ndarray = np.zeros((nframes, 1))
        accum_nframes: int = 0
        for ls in ms:
            ls_nframes: int = len(ls)
            for ii in range(ls_nframes):
                for tmp_type in np.unique(ls["atom_types"]):
                    coeff_mtx[accum_nframes+ii][tmp_type] = ls["atom_numbs"][tmp_type]
                energy_mtx[accum_nframes+ii] = ls["energies"][ii]
            accum_nframes += ls_nframes

        if (np.linalg.matrix_rank(coeff_mtx) < ntypes):
            import itertools
            for type_pair in itertools.combinations(unique_atom_types, 2):
                add_mtx: np.ndarray = np.zeros(ntypes)
                add_mtx[type_pair[0]] = 1
                add_mtx[type_pair[1]] = -1
                add_energy: np.ndarray = np.zeros(1)
                coeff_mtx = np.r_[coeff_mtx, [add_mtx]]
                energy_mtx = np.r_[energy_mtx, [add_energy]]
        types_energy_shifts: np.ndarray = svd_solve(coeff_mtx, energy_mtx)
        systems_shifted_energy: np.ndarray = (energy_mtx - np.matmul(coeff_mtx, types_energy_shifts)).flatten()
        return types_energy_shifts.flatten(), systems_shifted_energy
        
        

class PCA(object):
    pass
