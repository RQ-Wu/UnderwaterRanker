##########
# paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8917818
# github: https://github1s.com/Li-Chongyi/Water-Net_Code/blob/master/model.py
##########

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class FTU(nn.Module):
    def __init__(self, *, init_weights=True):
        super().__init__()
        self.nonlinearity = nn.ReLU(True)

        self.conv7x7 = nn.Conv2d(6, 32, kernel_size=7, stride=1, padding=3)
        self.conv5x5 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.conv3x3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y):
        assert x.shape[1] == 3 and y.shape[1] == 3
        xx = torch.cat([x, y], dim=1)
        xx = self.nonlinearity(self.conv7x7(xx))
        xx = self.nonlinearity(self.conv5x5(xx))
        xx = self.nonlinearity(self.conv3x3(xx))
        return xx

class MainBranch(nn.Module):
    def __init__(self, *, init_weights=True):
        super().__init__()
        self.nonlinearity = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

        self.conv_1 = nn.Conv2d(3 * 4, 128, kernel_size=7, stride=1, padding=3)
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

        self.conv_5 = nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3)
        self.conv_6 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.conv_7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_8 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, images, images_wb, images_ce, images_gc):
        xx = torch.cat([images, images_wb, images_ce, images_gc], dim=1)
        xx = self.nonlinearity(self.conv_1(xx))
        xx = self.nonlinearity(self.conv_2(xx))
        xx = self.nonlinearity(self.conv_3(xx))
        xx = self.nonlinearity(self.conv_4(xx))
        xx = self.nonlinearity(self.conv_5(xx))
        xx = self.nonlinearity(self.conv_6(xx))
        xx = self.nonlinearity(self.conv_7(xx))
        xx = self.sigmoid(self.conv_8(xx))
        return torch.chunk(xx, chunks=3, dim=1)

class WaterNet(nn.Module):
    def __init__(self, *, init_weights=True):
        super().__init__()
        self.main_branch = MainBranch(init_weights=init_weights)
        self.ftu_wb = FTU(init_weights=init_weights)
        self.ftu_ce = FTU(init_weights=init_weights)
        self.ftu_gc = FTU(init_weights=init_weights)

    def forward(self, images, images_wb, images_ce, images_gc):
        weight_wb ,weight_ce, weight_gc = self.main_branch(images, images_wb, images_ce, images_gc)
        wb = self.ftu_wb(images, images_wb)
        ce = self.ftu_ce(images, images_ce)
        gc = self.ftu_gc(images, images_gc)

        return wb * weight_wb + ce * weight_ce + gc * weight_gc

if __name__ == '__main__':
    water_net = WaterNet()
    img = torch.rand((4, 3, 128, 128))
    img_wb = torch.rand((4, 3, 128, 128))
    img_ce = torch.rand((4, 3, 128, 128))
    img_gc = torch.rand((4, 3, 128, 128))
    y = water_net(img, img_wb, img_ce, img_gc)
    print(y.shape)