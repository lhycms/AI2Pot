# 1. `has_forces` && `has_virial`

## 1.1. `ai2pot.data.dataset.ScDataset` && `ai2pot.data.dataset.McDataset`
<font color="#73DB90" size="5">

1. `has_virial`: 
   1. `has_virial=False`, return `9` tensors:
      1. `binum_tensor`
      2. `bilist_tensor`
      3. `bnumneigh_tensor`
      4. `bfirstneigh_tensor`
      5. `brcs_tensor`
      6. `btypes_tensor`
      7. `bnghost_tensor`
      8. `benergies`
      9. `bforces`
   2. `has_virial=True`, return `10` tensors:
      1. `binum_tensor`
      2. `bilist_tensor`
      3. `bnumneigh_tensor`
      4. `bfirstneigh_tensor`
      5. `brcs_tensor`
      6. `btypes_tensor`
      7. `bnghost_tensor`
      8. `benergies`
      9. `bforces`
      10. `bvirials`

</font>

## 1.2. `ai2pot.models.mtp.nn_mtp.NNMtp`
<font color="FF0077" size="5">

1. `has_forces`: 
   1. Whether True or False, return `9` tensors
2. `has_virial`: 
   1. `has_virial = False`, return `9` tensors
   2. `has_virial = True`, return `10` tensors
3. `ai2pot.data.dataset.ScDataset.has_virial` = `True` -> `ai2pot.models.mtp.nn_mtp.NNMtp.has_virial` = `True`

</font>
