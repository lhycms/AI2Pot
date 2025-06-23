from typing import List, Union, Optional
import torch
from lightning import LightningDataModule

from ai2pot.data.mlffdataset import ExtxyzDataset


class ExtxyzDataModule(LightningDataModule):
    def __init__(self,
                 trainset_path: Union[str, None] = None,
                 validset_path: Union[str, None] = None,
                 testset_path: Union[str, None] = None,
                 predict_path: Union[str, None] = None,
                 rcut: float = 6.0,
                 umax_num_neigh_atoms: int = 100,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = True,
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 has_virial: bool = False):
        super().__init__()
        self.trainset_path: str = trainset_path
        self.validset_path: str = validset_path
        self.testset_path: str = testset_path
        self.predict_path: str = predict_path
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype
        self.has_virial: bool = has_virial


    def setup(self, stage: Optional[str] = None):
        if (stage == "fit") or (stage is None):
            self.trainset_dataset = ExtxyzDataset(filename=self.trainset_path,
                                                  rcut=self.rcut,
                                                  umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                  pbc_xyz=self.pbc_xyz,
                                                  sort=self.sort,
                                                  torch_float_dtype=self.torch_float_dtype,
                                                  has_virial=self.has_virial)
            self.validset_dataset = ExtxyzDataset(filename=self.validset_path,
                                                  rcut=self.rcut,
                                                  umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                  pbc_xyz=self.pbc_xyz,
                                                  sort=self.sort,
                                                  torch_float_dtype=self.torch_float_dtype,
                                                  has_virial=self.has_virial)
        if (stage == "test") or (stage is None):
            self.testset_dataset = ExtxyzDataset(filename=self.testset_path,
                                                 rcut=self.rcut,
                                                 umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                 pbc_xyz=self.pbc_xyz,
                                                 sort=self.sort,
                                                 torch_float_dtype=self.torch_float_dtype,
                                                 has_virial=self.has_virial)
        if (stage == "predict") or (stage is None):
            self.predictset_dataset = ExtxyzDataset(filename=self.predict_path,
                                                    rcut=self.rcut,
                                                    umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                    pbc_xyz=self.pbc_xyz,
                                                    sort=self.sort,
                                                    torch_float_dtype=self.torch_float_dtype,
                                                    has_virial=self.has_virial)


    def train_dataloader(self):
        # trainer.fit(model, datamodule=dm)
        trainset_dataset: ExtxyzDataset = ExtxyzDataset(filename=self.trainset_path,
                                                        rcut=self.rcut,
                                                        umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                        pbc_xyz=self.pbc_xyz,
                                                        sort=self.sort,
                                                        torch_float_dtype=self.torch_float_dtype,
                                                        has_virial=self.has_virial)
        return trainset_dataset
    

    def val_dataloader(self):
        # trainer.fit(model, datamodule=dm)
        validset_dataset: ExtxyzDataset = ExtxyzDataset(filename=self.validset_path,
                                                        rcut=self.rcut,
                                                        umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                        pbc_xyz=self.pbc_xyz,
                                                        sort=self.sort,
                                                        torch_float_dtype=self.torch_float_dtype,
                                                        has_virial=self.has_virial)
        return validset_dataset
    

    def test_dataloader(self):
        # trainer.test(model, datamodule=dm)
        testset_dataset: ExtxyzDataset = ExtxyzDataset(filename=self.testset_path,
                                                       rcut=self.rcut,
                                                       umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                       pbc_xyz=self.pbc_xyz,
                                                       sort=self.sort,
                                                       torch_float_dtype=self.torch_float_dtype,
                                                       has_virial=self.has_virial)
        return testset_dataset

    
    def predict_dataloader(self):
        # trainer.predict(model, datamodule=dm)
        predictset_dataset: ExtxyzDataset = ExtxyzDataset(filename=self.predict_path,
                                                          rcut=self.rcut,
                                                          umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                          pbc_xyz=self.pbc_xyz,
                                                          sort=self.sort,
                                                          torch_float_dtype=self.torch_float_dtype,
                                                          has_virial=self.has_virial)
        return predictset_dataset
    