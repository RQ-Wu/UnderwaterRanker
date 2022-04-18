from baselines.all_model.MyCoaT.model import MyCoaT
import torch

model = MyCoaT(patch_size=4, embed_dims=[152, 320, 320, 320], serial_depths=[2, 2, 2, 2], parallel_depth=6, num_heads=8, mlp_ratios=[4, 4, 4, 4])
model = model.cuda()

img = torch.rand((1, 3, 736, 1120)).cuda()
p = model(img)
print(p['final_result'][0][0][0])