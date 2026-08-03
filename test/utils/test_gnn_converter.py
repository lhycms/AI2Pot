from typing import List
import unittest
import os

import torch
from pymatgen.core import Structure
from ai2pot.utils.usepot import MlffInput

from ai2pot.utils.gnn_converter import (
    GraphData,
    GraphDataConverter)


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
MoS2_POSCAR_PATH = os.path.join(TEST_FILES_DIR, "POSCARs", "MoS2", "POSCAR_perturbed0.2")


class GraphDataConverterTest(unittest.TestCase):
    def setUp(self):
        print("GraphDataConverterTest (TestCase) is setting up...\n")
        self.type_map: List[int] = [42, 16]
        self.rcut: float = 5.0
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
    
    
    def tearDown(self):
        print("GraphDataConverterTest (TestCase) is tearing down...\n")
    
    
    def test_convert_nblist_to_graph(self):
        binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, brcs_tensor, btypes_tensor, bnghost_tensor = self.mlff_input.analyse_pymatgen(structure=self.structure)
        
        graph_data: GraphData = GraphDataConverter.convert_nblist_to_graph(
            binum_tensor=binum_tensor,
            bilist_tensor=bilist_tensor,
            bnumneigh_tensor=bnumneigh_tensor,
            bfirstneigh_tensor=bfirstneigh_tensor,
            brcs_tensor=brcs_tensor,
            btypes_tensor=btypes_tensor,
            bnghost_tensor=bnghost_tensor)

        print(graph_data.node_types_tensor.shape)
        print(graph_data.node_frame_idx_tensor.shape)
        print(graph_data.edge_index_tensor.shape)
        print(graph_data.edge_vec_tensor.shape)
        print(graph_data.binum_tensor.shape)
        


if __name__ == "__main__":
    unittest.main()
    