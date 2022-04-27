import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

from einops import rearrange
from functools import partial
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        out_H, out_W = H // self.patch_size[0], W // self.patch_size[1]

        x = self.proj(x).flatten(2).transpose(1, 2)
        out = self.norm(x)
        
        return out, (out_H, out_W)

class SerialBlock(nn.Module):
    def __init__(self):
        pass 
    
    def forward(self):
        pass

class Ranker(nn.Module):
    def __init__(self, patch_size=4, in_channel=3, num_classes=1, embed_dims=[152, 320, 320, 320],
                 serial_depths=[2, 2, 2, 2], parallel_depth=6, mlp_ratio=[4,4,4,4],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), serial_name=None, serial_args=None,
                 parallel_name=None, parallel_args=None):
        # Patch embeddings
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Class tokens
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        seriablock = eval(serial_name)
        parallelblock = eval(parallel_name)
        # Serial blocks 1.
        self.serial_blocks1 = nn.ModuleList([
            seriablock(dim=embed_dims[0], mlp_ratio=mlp_ratio[0], **serial_args) for _ in range(serial_depths[0])]
        )
        # Serial blocks 2.
        self.serial_blocks2 = nn.ModuleList([
            seriablock(dim=embed_dims[1], mlp_ratio=mlp_ratio[1], **serial_args) for _ in range(serial_depths[1])]
        )
        # Serial blocks 1.
        self.serial_blocks3 = nn.ModuleList([
            seriablock(dim=embed_dims[2], mlp_ratio=mlp_ratio[2], **serial_args) for _ in range(serial_depths[2])]
        )
        # Serial blocks 1.
        self.serial_blocks4 = nn.ModuleList([
            seriablock(dim=embed_dims[3], mlp_ratio=mlp_ratio[3], **serial_args) for _ in range(serial_depths[3])]
        )

        # parallel_blocks
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList([
                parallelblock(**parallel_args) for _ in range(self.parallel_depth)
            ])