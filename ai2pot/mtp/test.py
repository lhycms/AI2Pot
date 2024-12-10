import torch

x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
for sub in torch.split(x, [3, 5, 2]):
    print(sub)
    