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
        self.nframes = len(ls)
        self.elements_energy_shifts, self.systems_energy_shifts = LsShifter.get_energy_shifts(ls)
    
    
    @staticmethod
    def get_energy_shifts(ls: LabeledSystem):
        unique_atom_types: np.ndarray =  np.unique(ls["atom_types"])
        nframes: int = len(ls)
        ntypes: int = unique_atom_types.shape[0]
        coeff_mtx: np.ndarray = np.zeros((nframes, ntypes))
        energy_mtx: np.ndarray = np.zeros((nframes, 1))

        for ii in range(nframes):
            for jj in range(ntypes):
                coeff_mtx[ii][jj] = np.sum(ls["atom_types"] == jj)
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
        elements_energy_shifts: np.ndarray = svd_solve(coeff_mtx, energy_mtx)
        systems_energy_shifts: np.ndarray = (energy_mtx - np.matmul(coeff_mtx, elements_energy_shifts)).flatten()
        return elements_energy_shifts, systems_energy_shifts
        

class MsShifter(object):
    def __init__(self, ms: MultiSystems):
        self.nframes: int = len(MultiSystems)
    
    @staticmethod
    def get_energy_shifts(ms: MultiSystems):
        pass
        

class PCA(object):
    pass
