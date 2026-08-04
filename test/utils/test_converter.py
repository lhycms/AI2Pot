from typing import List
import unittest
import os

import torch
from torch.utils.data import DataLoader
from pymatgen.core import Structure
from ai2pot.utils.usepot import MlffInput
from ai2pot.data.mlffdataset import ExtxyzDataset

from ai2pot.utils.converter import (
    GraphData,
    GraphDataConverter)


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
MoS2_POSCAR_PATH = os.path.join(TEST_FILES_DIR, "POSCARs", "MoS2", "POSCAR_perturbed0.2")
GeSbTe_EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/train.xyz"


class GraphDataConverterTest(unittest.TestCase):
    def setUp(self):
        print("GraphDataConverterTest (TestCase) is setting up...\n")
        self.type_map: List[int] = [42, 16]
        self.rcut: float = 6.0
        self.umax_num_neigh_atoms: int = 200
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.device: torch._C.device = torch.device("cuda")
        
        self.mlff_input: MlffInput = MlffInput(
            type_map=self.type_map,
            rcut=self.rcut,
            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
            pbc_xyz=[True, True, True],
            sort=False,
            dtype=self.torch_float_dtype,
            device=self.device)
        
        self.structure: Structure = Structure.from_file(filename=MoS2_POSCAR_PATH)
        
        self.extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=GeSbTe_EXTXYZ_PATH,
                                                           rcut=self.rcut,
                                                           umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                           pbc_xyz=[True, True, True],
                                                           sort=False,
                                                           torch_float_dtype=self.torch_float_dtype,
                                                           has_virial=True)
        self.extxyz_dataloader: DataLoader = DataLoader(dataset=self.extxyz_dataset,
                                                        batch_size=200,
                                                        shuffle=True)
    
    
    def tearDown(self):
        print("GraphDataConverterTest (TestCase) is tearing down...\n")
    
    
    def est_convert_nblist_to_graph(self):
        binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, brcs_tensor, btypes_tensor, bnghost_tensor = self.mlff_input.analyse_pymatgen(structure=self.structure)
        
        graph_data: GraphData = GraphDataConverter.convert_nblist_to_graph(
            binum_tensor=binum_tensor,
            bilist_tensor=bilist_tensor,
            bnumneigh_tensor=bnumneigh_tensor,
            bfirstneigh_tensor=bfirstneigh_tensor,
            brcs_tensor=brcs_tensor,
            btypes_tensor=btypes_tensor,
            bnghost_tensor=bnghost_tensor)

        print("\t1.1. graph_data.node_types_tensor.shape = ", graph_data.node_types_tensor.shape)
        print("\t1.2. graph_data.node_frame_idx_tensor.shape = ", graph_data.node_frame_idx_tensor.shape)
        print("\t1.3. graph_data.edge_index_tensor.shape = ", graph_data.edge_index_tensor.shape)
        print("\t1.4. graph_data.edge_vec_tensor.shape = ", graph_data.edge_vec_tensor.shape)
        print("\t1.5. graph_data.binum_tensor.shape = ", graph_data.binum_tensor.shape)
        print("\t1.6. graph_data.bnghost_tensor.shape = ", graph_data.bnghost_tensor.shape)
        
    
    def test_dataloader_converter(self):
        num_atoms: int = 0
        for batch_idx, batch_data in enumerate(self.extxyz_dataloader):
            binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, brcs_tensor, btypes_tensor, bnghost_tensor, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            batch_graph_data: GraphData = GraphDataConverter.convert_nblist_to_graph(
                binum_tensor=binum_tensor,
                bilist_tensor=bilist_tensor,
                bnumneigh_tensor=bnumneigh_tensor,
                bfirstneigh_tensor=bfirstneigh_tensor,
                brcs_tensor=brcs_tensor,
                btypes_tensor=btypes_tensor,
                bnghost_tensor=bnghost_tensor
            )
            num_atoms += batch_graph_data.node_types_tensor.shape[0]
            
            print("Batch #{0:4d}: ".format(batch_idx))
            print("\t1. batch_graph_data.node_types_tensor.shape = ", batch_graph_data.node_types_tensor.shape)
            print("\t2. batch_graph_data.node_frame_idx_tensor.shape = ", batch_graph_data.node_frame_idx_tensor.shape)
            print("\t3. batch_graph_data.edge_index_tensor.shape = ", batch_graph_data.edge_index_tensor.shape)
            print("\t4. batch_graph_data.edge_vec_tensor.shape = ", batch_graph_data.edge_vec_tensor.shape)
            print("\t5. batch_graph_data.binum_tensor.shape = ", batch_graph_data.binum_tensor.shape)
            print("\t6. batch_graph_data.bnghost_tensor.shape = ", batch_graph_data.bnghost_tensor.shape)
            print()
            
        print("Total number of atoms = ", num_atoms)
        

if __name__ == "__main__":
    unittest.main()
    