import torch


model = torch.jit.load("/data/home/liuhanyu/hyliu/code/mlff/PWmatMLFF_dev/test/CH4_torch_script/torch_script_module.pt", map_location=torch.device("cuda:0"))
print(model.davg[:, :4])
print(model.dstd[:, :4])
