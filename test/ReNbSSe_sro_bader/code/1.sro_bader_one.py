import numpy as np
from typing import List
import copy
import matplotlib.pyplot as plt
import pandas as pd

from pymatgen.core import Structure
from ai2pot.core import Nblist


class Bader(object):
    def __init__(
                self,
                acf_dat_path:str,
                elements_lst:List[str],
                num_atoms_lst:List[int],
                charges_origin_lst:List[float],
                ):
        '''
        Description
        -----------
            1. 由 `ACF.dat` 文件得到 Bader 电荷情况
        
        Parameters
        ----------
            1. acf_dat_path: str
                - `ACF.dat` 文件的路径
            2. elements_lst:
                - 体系内的元素种类，按照出现顺序排列
            3. num_atoms_lst:
                - 各个元素的原子数目
            4. charges_origin_lst:
                - 各个元素的原价电子数目
        '''
        self.num_atoms = sum(num_atoms_lst)
        self.charges_current_arary = np.loadtxt(
                                fname=acf_dat_path,
                                skiprows=2,
                                max_rows=self.num_atoms,)[:, 4] 
        
        self.elements_lst = elements_lst
        self.charges_origin_lst = charges_origin_lst
        self.num_atoms_lst = num_atoms_lst
    

    def get_charges_current_array(self):
        '''
        Description
        -----------
            1. 得到现在体系内的电荷情况
        '''
        return self.charges_current_arary
    

    def _get_element_split_marks(self):
        '''
        E.g.
        ----
            1. [  0  18  36  72 108]
        '''
        element_split_marks = copy.deepcopy(self.num_atoms_lst)
        element_split_marks.insert(0, 0)
        element_split_marks = np.cumsum(element_split_marks)
        return element_split_marks


    def get_charges_diff_array(self):
        '''
        Description
        -----------
            1. 得到体系内电荷的电子得失情况
                - 负的为失去电子
                - 正的为得到电子
        '''
        element_split_marks = self._get_element_split_marks()
        charges_diff_array = copy.deepcopy(self.charges_current_arary)
        
        for tmp_idx in range(1, len(element_split_marks)):
            # e.g. [0, 18]
            tmp_range = [element_split_marks[tmp_idx-1], element_split_marks[tmp_idx]]
            charges_diff_array[tmp_range[0]: tmp_range[1]] = \
                        self.charges_current_arary[tmp_range[0]: tmp_range[1]] - self.charges_origin_lst[tmp_idx-1]

        return charges_diff_array        
        
    
    def diff_mean_lst(self):
        diff_means_lst = []
        element_split_marks = self._get_element_split_marks()
        charges_diff_array = self.get_charges_diff_array()

        for tmp_idx in range(1, len(element_split_marks)):
            tmp_range = [element_split_marks[tmp_idx-1], element_split_marks[tmp_idx]]
            tmp_diff_mean = round(np.mean(charges_diff_array[tmp_range[0]: tmp_range[1]]), 4)
            diff_means_lst.append(tmp_diff_mean)
        
        return diff_means_lst


    def diff_std_lst(self):
        diff_stds_lst = []
        element_split_marks = self._get_element_split_marks()
        charges_diff_array = self.get_charges_diff_array()

        for tmp_idx in range(1, len(element_split_marks)):
            tmp_range = [element_split_marks[tmp_idx-1], element_split_marks[tmp_idx]]
            tmp_diff_mean = round(np.std(charges_diff_array[tmp_range[0]: tmp_range[1]]), 4)
            diff_stds_lst.append(tmp_diff_mean)
        
        return diff_stds_lst


    def diff_max_lst(self):
        diff_maxs_lst = []
        element_split_marks = self._get_element_split_marks()
        charges_diff_array = self.get_charges_diff_array()

        for tmp_idx in range(1, len(element_split_marks)):
            tmp_range = [element_split_marks[tmp_idx-1], element_split_marks[tmp_idx]]
            tmp_diff_mean = round(np.max(charges_diff_array[tmp_range[0]: tmp_range[1]]), 4)
            diff_maxs_lst.append(tmp_diff_mean)
        
        return diff_maxs_lst


    def diff_min_lst(self):
        diff_mins_lst = []
        element_split_marks = self._get_element_split_marks()
        charges_diff_array = self.get_charges_diff_array()

        for tmp_idx in range(1, len(element_split_marks)):
            tmp_range = [element_split_marks[tmp_idx-1], element_split_marks[tmp_idx]]
            tmp_diff_mean = round(np.min(charges_diff_array[tmp_range[0]: tmp_range[1]]), 4)
            diff_mins_lst.append(tmp_diff_mean)
        
        return diff_mins_lst
    


def get_sro_1st(poscar_path: str,
                cutoff: float,
                center_type: int,
                neigh_type: int,
                neighbor_types: List[int]):
    structure = Structure.from_file(poscar_path)
    nblist = Nblist(structure=structure,
                    rcut=cutoff,
                    umax_num_neigh_atoms=umax_num_neigh_atoms,
                    pbc_xyz=pbc_xyz,
                    sort=sort)
    
    # Get new firstneigh
    flatten_index: np.ndarray = nblist.firstneigh.flatten()
    new_firstneigh = np.empty_like(flatten_index)
    for ii in range(len(flatten_index)):
        if flatten_index[ii] == -1:
            new_firstneigh[ii] = -1
        else:
            new_firstneigh[ii] = nblist.type_map_inv[nblist.types[flatten_index[ii]]]
    new_firstneigh = new_firstneigh.reshape(-1, umax_num_neigh_atoms)
    
    # Get itype_mask
    new_ilist = np.array([nblist.type_map_inv[nblist.types[tmp_idx]] for tmp_idx in nblist.ilist])
    itype_mask = np.where(new_ilist == center_type, True, False)
   
    # Get new_firstneigh we need
    selected_firstneigh = new_firstneigh[itype_mask]
    num_neigh0 = np.where(selected_firstneigh == neigh_type, 
                        True, 
                        False)
    num_neigh0 = np.sum(num_neigh0, axis=1)
    num_neigh1 = np.where(selected_firstneigh == neighbor_types[0], 
                        True, 
                        False)
    num_neigh1 = np.sum(num_neigh1, axis=1)
    num_neigh2 = np.where(selected_firstneigh == neighbor_types[1], 
                        True, 
                        False)
    num_neigh2 = np.sum(num_neigh2, axis=1)
    tot_num_neighs = num_neigh1 + num_neigh2
    
    # Calculate SRO in first shell
    sro = num_neigh0 / tot_num_neighs
    
    return sro, itype_mask


if __name__ == "__main__":
    # Part 1.
    system = "ReNbSSe"
    mc_x = "mc_3_28"
    poscar_path: str = "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSSe/input_data/{0}/CONTCARs/{1}/CONTCAR".format(system, mc_x)
    acf_dat_path = "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSSe/input_data/{0}/bader/{1}/ACF.dat".format(system, mc_x)
    # atomic numbers : [75, 41, 16, 34]
    elements_lst = ["Re", "Nb", "S", "Se"]
    charges_origin_lst = [7, 13, 6, 6]
    num_atoms_lst = [18, 18, 36, 36]
    
    # Part 2.
    cutoff: float = 3.0 # 4.8 for metal. 3.0 for S.
    umax_num_neigh_atoms: int = 20
    pbc_xyz: List[bool] = [True, True, True]
    sort: bool = True
    
    center_type = 75
    neigh_type = 16
    neighbor_types = [16, 34]
    
    
    # Part 3.得到对应中心原子的SRO
    sro, itype_mask = get_sro_1st(poscar_path=poscar_path,
                                  cutoff=cutoff,
                                  center_type=center_type,
                                  neigh_type=neigh_type,
                                  neighbor_types=neighbor_types)
    xs_lst = sro

    # Part 4. 得到对应中心原子的bader
    bader_object = Bader(
                acf_dat_path=acf_dat_path,
                elements_lst=elements_lst,
                num_atoms_lst=num_atoms_lst,
                charges_origin_lst=charges_origin_lst)
    ys_lst = bader_object.get_charges_diff_array()[itype_mask]
    
    # Part 5. Save DataFrame
    df = pd.DataFrame()
    df["{0}_{1}_sro".format(center_type, neigh_type)] = xs_lst
    df["{0}_charge".format(center_type)] = ys_lst
    
    print("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSSe/output/{0}/{1}_{2}.csv".format(mc_x, center_type, neigh_type))
    df.to_csv("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/ReNbSSe/output/{0}/{1}_{2}.csv".format(mc_x, center_type, neigh_type))
    