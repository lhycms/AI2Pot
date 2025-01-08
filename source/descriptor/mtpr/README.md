# 1. Input tensor size for `MtpBasisOp`
<font color="FF0066" size="6">

1. `binum_tensor`: [`batch_size`,]
2. `bilist_tensor`: [`batch_size`, natoms]
3. `bnumneigh_tensor`: [`batch_size`, natoms]
4. `bfirstneigh_tensor`: [`batch_size`, natoms, umax_num_neigh_atoms]
5. `brcs_tensor`: [`batch_size`, natoms, umax_num_neigh_atoms, 3]
6. `btypes_tensor`: [`batch_size`, natoms+nghost]

</font>


# 2. `torch.utils.data.Dataset`
<font color="FF0066" size="6">

1. `inum_tensor`: []
2. `ilist_tensor`: [natoms]
3. `numneigh_tensor`: [natoms]
4. `firstneigh_tensor`: [natoms, umax_num_neigh_atoms]
5. `rcs_tensor`: [natoms, umax_num_neigh_atoms, 3]
6. `types_tensor`: [natoms+nghost]

</font>

```python
mlff_dataset: ScDataset = ScDataset(labeled_system=labeled_system,
                                    rcut=3.2,
                                    umax_num_neigh_atoms=20)
mlff_dataloader: DataLoader = DataLoader(
    self.mlff_dataset,
    batch_size=5,
    shuffle=True)

for ii in self.mlff_dataset[0]:
    if (type(ii) is torch.Tensor):
        print(ii.size())
    else:
        print(type(ii))
```
Output:
```shell
$ python3 main.py
torch.Size([])                      # inum
torch.Size([108])                   # ilist
torch.Size([108])                   # numneigh
torch.Size([108, 20])               # firstneigh
torch.Size([108, 20, 3])            # rcs
torch.Size([108])                   # types
torch.Size([])                      # nghost
torch.Size([])                      # energy
torch.Size([108, 3])                # force
torch.Size([3, 3])                  # virial
```


# 3. `torch.utils.data.DataLoader`
## Case 1 : `batch_size = 1`
<font color="FF0066" size="6">

1. `binum_tensor`: [`1`,]
2. `bilist_tensor`: [`1`, natoms]
3. `bnumneigh_tensor`: [`1`, natoms]
4. `bfirstneigh_tensor`: [`1`, natoms, umax_num_neigh_atoms]
5. `brcs_tensor`: [`1`, natoms, umax_num_neigh_atoms, 3]
6. `btypes_tensor`: [`1`, natoms+nghost]

</font>

## Case 2 : `batch_size = 5`
<font color="FF0066" size="6">

1. `binum_tensor`: [`5`,]
2. `bilist_tensor`: [`5`, natoms]
3. `bnumneigh_tensor`: [`5`, natoms]
4. `bfirstneigh_tensor`: [`5`, natoms, umax_num_neigh_atoms]
5. `brcs_tensor`: [`5`, natoms, umax_num_neigh_atoms, 3]
6. `btypes_tensor`: [`5`, natoms+nghost]

</font>

```python
mlff_dataset: ScDataset = ScDataset(labeled_system=labeled_system,
                                    rcut=3.2,
                                    umax_num_neigh_atoms=20)
mlff_dataloader: DataLoader = DataLoader(
    self.mlff_dataset,
    batch_size=5,
    shuffle=True)

for batch_idx, batch_data in enumerate(self.mlff_dataloader):
    print("\tbinum.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[0].size()))
    print("\tbilist.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[1].size()))
    print("\tbnumneigh.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[2].size()))
    print("\tbfirstneigh.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[3].size()))
    print("\tbrcs.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[4].size()))
    print("\tbtypes.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[5].size()))
    print("\tbnghost.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[6].size()))
    print("\tbenergies.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[7].size()))
    print("\tbforces.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[8].size()))
    print("\tbvirials.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[9].size()))
    print("\n")
```
Output:
```output
$ python3 main.py
ScDatasetTest is setting up...

        binum.size() inner Batch#0 = torch.Size([1])
        bilist.size() inner Batch#0 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#0 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#0 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#0 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#0 = torch.Size([1, 108])
        bnghost.size() inner Batch#0 = torch.Size([1])
        benergies.size() inner Batch#0 = torch.Size([1])
        bforces.size() inner Batch#0 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#0 = torch.Size([1, 3, 3])


        binum.size() inner Batch#1 = torch.Size([1])
        bilist.size() inner Batch#1 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#1 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#1 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#1 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#1 = torch.Size([1, 108])
        bnghost.size() inner Batch#1 = torch.Size([1])
        benergies.size() inner Batch#1 = torch.Size([1])
        bforces.size() inner Batch#1 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#1 = torch.Size([1, 3, 3])


        binum.size() inner Batch#2 = torch.Size([1])
        bilist.size() inner Batch#2 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#2 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#2 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#2 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#2 = torch.Size([1, 108])
        bnghost.size() inner Batch#2 = torch.Size([1])
        benergies.size() inner Batch#2 = torch.Size([1])
        bforces.size() inner Batch#2 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#2 = torch.Size([1, 3, 3])


        binum.size() inner Batch#3 = torch.Size([1])
        bilist.size() inner Batch#3 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#3 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#3 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#3 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#3 = torch.Size([1, 108])
        bnghost.size() inner Batch#3 = torch.Size([1])
        benergies.size() inner Batch#3 = torch.Size([1])
        bforces.size() inner Batch#3 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#3 = torch.Size([1, 3, 3])


        binum.size() inner Batch#4 = torch.Size([1])
        bilist.size() inner Batch#4 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#4 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#4 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#4 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#4 = torch.Size([1, 108])
        bnghost.size() inner Batch#4 = torch.Size([1])
        benergies.size() inner Batch#4 = torch.Size([1])
        bforces.size() inner Batch#4 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#4 = torch.Size([1, 3, 3])


        binum.size() inner Batch#5 = torch.Size([1])
        bilist.size() inner Batch#5 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#5 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#5 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#5 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#5 = torch.Size([1, 108])
        bnghost.size() inner Batch#5 = torch.Size([1])
        benergies.size() inner Batch#5 = torch.Size([1])
        bforces.size() inner Batch#5 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#5 = torch.Size([1, 3, 3])


        binum.size() inner Batch#6 = torch.Size([1])
        bilist.size() inner Batch#6 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#6 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#6 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#6 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#6 = torch.Size([1, 108])
        bnghost.size() inner Batch#6 = torch.Size([1])
        benergies.size() inner Batch#6 = torch.Size([1])
        bforces.size() inner Batch#6 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#6 = torch.Size([1, 3, 3])


        binum.size() inner Batch#7 = torch.Size([1])
        bilist.size() inner Batch#7 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#7 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#7 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#7 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#7 = torch.Size([1, 108])
        bnghost.size() inner Batch#7 = torch.Size([1])
        benergies.size() inner Batch#7 = torch.Size([1])
        bforces.size() inner Batch#7 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#7 = torch.Size([1, 3, 3])


        binum.size() inner Batch#8 = torch.Size([1])
        bilist.size() inner Batch#8 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#8 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#8 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#8 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#8 = torch.Size([1, 108])
        bnghost.size() inner Batch#8 = torch.Size([1])
        benergies.size() inner Batch#8 = torch.Size([1])
        bforces.size() inner Batch#8 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#8 = torch.Size([1, 3, 3])


        binum.size() inner Batch#9 = torch.Size([1])
        bilist.size() inner Batch#9 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#9 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#9 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#9 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#9 = torch.Size([1, 108])
        bnghost.size() inner Batch#9 = torch.Size([1])
        benergies.size() inner Batch#9 = torch.Size([1])
        bforces.size() inner Batch#9 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#9 = torch.Size([1, 3, 3])


        binum.size() inner Batch#10 = torch.Size([1])
        bilist.size() inner Batch#10 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#10 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#10 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#10 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#10 = torch.Size([1, 108])
        bnghost.size() inner Batch#10 = torch.Size([1])
        benergies.size() inner Batch#10 = torch.Size([1])
        bforces.size() inner Batch#10 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#10 = torch.Size([1, 3, 3])


        binum.size() inner Batch#11 = torch.Size([1])
        bilist.size() inner Batch#11 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#11 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#11 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#11 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#11 = torch.Size([1, 108])
        bnghost.size() inner Batch#11 = torch.Size([1])
        benergies.size() inner Batch#11 = torch.Size([1])
        bforces.size() inner Batch#11 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#11 = torch.Size([1, 3, 3])


        binum.size() inner Batch#12 = torch.Size([1])
        bilist.size() inner Batch#12 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#12 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#12 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#12 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#12 = torch.Size([1, 108])
        bnghost.size() inner Batch#12 = torch.Size([1])
        benergies.size() inner Batch#12 = torch.Size([1])
        bforces.size() inner Batch#12 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#12 = torch.Size([1, 3, 3])


        binum.size() inner Batch#13 = torch.Size([1])
        bilist.size() inner Batch#13 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#13 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#13 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#13 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#13 = torch.Size([1, 108])
        bnghost.size() inner Batch#13 = torch.Size([1])
        benergies.size() inner Batch#13 = torch.Size([1])
        bforces.size() inner Batch#13 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#13 = torch.Size([1, 3, 3])


        binum.size() inner Batch#14 = torch.Size([1])
        bilist.size() inner Batch#14 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#14 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#14 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#14 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#14 = torch.Size([1, 108])
        bnghost.size() inner Batch#14 = torch.Size([1])
        benergies.size() inner Batch#14 = torch.Size([1])
        bforces.size() inner Batch#14 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#14 = torch.Size([1, 3, 3])


        binum.size() inner Batch#15 = torch.Size([1])
        bilist.size() inner Batch#15 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#15 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#15 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#15 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#15 = torch.Size([1, 108])
        bnghost.size() inner Batch#15 = torch.Size([1])
        benergies.size() inner Batch#15 = torch.Size([1])
        bforces.size() inner Batch#15 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#15 = torch.Size([1, 3, 3])


        binum.size() inner Batch#16 = torch.Size([1])
        bilist.size() inner Batch#16 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#16 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#16 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#16 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#16 = torch.Size([1, 108])
        bnghost.size() inner Batch#16 = torch.Size([1])
        benergies.size() inner Batch#16 = torch.Size([1])
        bforces.size() inner Batch#16 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#16 = torch.Size([1, 3, 3])


        binum.size() inner Batch#17 = torch.Size([1])
        bilist.size() inner Batch#17 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#17 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#17 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#17 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#17 = torch.Size([1, 108])
        bnghost.size() inner Batch#17 = torch.Size([1])
        benergies.size() inner Batch#17 = torch.Size([1])
        bforces.size() inner Batch#17 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#17 = torch.Size([1, 3, 3])


        binum.size() inner Batch#18 = torch.Size([1])
        bilist.size() inner Batch#18 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#18 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#18 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#18 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#18 = torch.Size([1, 108])
        bnghost.size() inner Batch#18 = torch.Size([1])
        benergies.size() inner Batch#18 = torch.Size([1])
        bforces.size() inner Batch#18 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#18 = torch.Size([1, 3, 3])


        binum.size() inner Batch#19 = torch.Size([1])
        bilist.size() inner Batch#19 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#19 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#19 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#19 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#19 = torch.Size([1, 108])
        bnghost.size() inner Batch#19 = torch.Size([1])
        benergies.size() inner Batch#19 = torch.Size([1])
        bforces.size() inner Batch#19 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#19 = torch.Size([1, 3, 3])


        binum.size() inner Batch#20 = torch.Size([1])
        bilist.size() inner Batch#20 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#20 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#20 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#20 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#20 = torch.Size([1, 108])
        bnghost.size() inner Batch#20 = torch.Size([1])
        benergies.size() inner Batch#20 = torch.Size([1])
        bforces.size() inner Batch#20 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#20 = torch.Size([1, 3, 3])


        binum.size() inner Batch#21 = torch.Size([1])
        bilist.size() inner Batch#21 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#21 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#21 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#21 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#21 = torch.Size([1, 108])
        bnghost.size() inner Batch#21 = torch.Size([1])
        benergies.size() inner Batch#21 = torch.Size([1])
        bforces.size() inner Batch#21 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#21 = torch.Size([1, 3, 3])


        binum.size() inner Batch#22 = torch.Size([1])
        bilist.size() inner Batch#22 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#22 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#22 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#22 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#22 = torch.Size([1, 108])
        bnghost.size() inner Batch#22 = torch.Size([1])
        benergies.size() inner Batch#22 = torch.Size([1])
        bforces.size() inner Batch#22 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#22 = torch.Size([1, 3, 3])


        binum.size() inner Batch#23 = torch.Size([1])
        bilist.size() inner Batch#23 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#23 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#23 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#23 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#23 = torch.Size([1, 108])
        bnghost.size() inner Batch#23 = torch.Size([1])
        benergies.size() inner Batch#23 = torch.Size([1])
        bforces.size() inner Batch#23 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#23 = torch.Size([1, 3, 3])


        binum.size() inner Batch#24 = torch.Size([1])
        bilist.size() inner Batch#24 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#24 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#24 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#24 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#24 = torch.Size([1, 108])
        bnghost.size() inner Batch#24 = torch.Size([1])
        benergies.size() inner Batch#24 = torch.Size([1])
        bforces.size() inner Batch#24 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#24 = torch.Size([1, 3, 3])


        binum.size() inner Batch#25 = torch.Size([1])
        bilist.size() inner Batch#25 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#25 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#25 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#25 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#25 = torch.Size([1, 108])
        bnghost.size() inner Batch#25 = torch.Size([1])
        benergies.size() inner Batch#25 = torch.Size([1])
        bforces.size() inner Batch#25 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#25 = torch.Size([1, 3, 3])


        binum.size() inner Batch#26 = torch.Size([1])
        bilist.size() inner Batch#26 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#26 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#26 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#26 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#26 = torch.Size([1, 108])
        bnghost.size() inner Batch#26 = torch.Size([1])
        benergies.size() inner Batch#26 = torch.Size([1])
        bforces.size() inner Batch#26 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#26 = torch.Size([1, 3, 3])


        binum.size() inner Batch#27 = torch.Size([1])
        bilist.size() inner Batch#27 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#27 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#27 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#27 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#27 = torch.Size([1, 108])
        bnghost.size() inner Batch#27 = torch.Size([1])
        benergies.size() inner Batch#27 = torch.Size([1])
        bforces.size() inner Batch#27 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#27 = torch.Size([1, 3, 3])


        binum.size() inner Batch#28 = torch.Size([1])
        bilist.size() inner Batch#28 = torch.Size([1, 108])
        bnumneigh.size() inner Batch#28 = torch.Size([1, 108])
        bfirstneigh.size() inner Batch#28 = torch.Size([1, 108, 20])
        brcs.size() inner Batch#28 = torch.Size([1, 108, 20, 3])
        btypes.size() inner Batch#28 = torch.Size([1, 108])
        bnghost.size() inner Batch#28 = torch.Size([1])
        benergies.size() inner Batch#28 = torch.Size([1])
        bforces.size() inner Batch#28 = torch.Size([1, 108, 3])
        bvirials.size() inner Batch#28 = torch.Size([1, 3, 3])


ScDatasetTest is tearing down...
```


# 4. `aitpot.utils.MlffInput`
<font color="FF0066" size="6">

1. `binum_tensor`: [`1`]
2. `bilist_tensor`: [`1`, natoms]
3. `bnumneigh_tensor`: [`1`, natoms]
4. `bfirstneugh_tensor`: [`1`, natoms, umax_num_neigh_atoms]
5. `brcs_tensor`: [`1`, natoms, umax_num_neigh_atoms, 3]
6. `types_tensor`: [`1`, natoms+nghost]

$$\frac{\partial {\frac{\partial D_{ik}}{\partial r_{ij}}}}{\partial c_{Z_i Z_j n \xi}}$$
.shape = $[N_{atom}, N_{descriptors}, N_{neighbors}, 3, N_{coeffs}]$

</font>
