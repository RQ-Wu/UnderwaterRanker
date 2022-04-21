import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

class Encoder_block_lite(nn.Module):
    def __init__(self, in_channel, out_channel, is_downsample=True):
        super().__init__()
        self.is_downsample = is_downsample
        if self.is_downsample:
            self.downsample = nn.MaxPool2d(2, 2)
            
        # HSV encoder
        self.conv2_1_HSV = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_1_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_2_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_3_HSV = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        
        self.conv2_2_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_4_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_5_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_6_HSV = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        
        # LAB encoder
        self.conv2_1_LAB = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_1_LAB = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_2_LAB = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_3_LAB = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        
        self.conv2_2_LAB = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_4_LAB = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_5_LAB = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_6_LAB = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        
        # RGB encoder
        self.conv2_1_RGB = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_1_RGB = nn.Sequential(nn.Conv2d(out_channel*3, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_2_RGB = nn.Sequential(nn.Conv2d(out_channel*3, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_3_RGB = nn.Conv2d(out_channel*3, out_channel, 3, 1, 1)
        
        self.conv2_2_RGB = nn.Sequential(nn.Conv2d(out_channel*3, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_4_RGB = nn.Sequential(nn.Conv2d(out_channel*3, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_5_RGB = nn.Sequential(nn.Conv2d(out_channel*3, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_6_RGB = nn.Conv2d(out_channel*3, out_channel, 3, 1, 1)
        
    def forward(self, x_hsv, x_lab, x_rgb):
        if self.is_downsample:
            x_hsv = self.downsample(x_hsv)
            x_lab = self.downsample(x_lab)
            x_rgb = self.downsample(x_rgb)
            
        conv2_1_HSV = self.conv2_1_HSV(x_hsv)
        conv2_1_LAB = self.conv2_1_LAB(x_lab)
        conv2_1_RGB = self.conv2_1_RGB(x_lab)
        # print(1)
        # print(conv2_1_RGB)
        
        conv2_cb1_1_HSV = self.conv2_cb1_1_HSV(conv2_1_HSV)
        conv2_cb1_1_LAB = self.conv2_cb1_1_LAB(conv2_1_LAB)
        conv2_cb1_1_RGB = self.conv2_cb1_1_RGB(torch.cat((conv2_1_RGB, conv2_1_HSV, conv2_1_LAB), dim=1))
        # print(2)
        # print(conv2_cb1_1_RGB)
        
        conv2_cb1_2_HSV = self.conv2_cb1_2_HSV(conv2_cb1_1_HSV)
        conv2_cb1_2_LAB = self.conv2_cb1_2_LAB(conv2_cb1_1_LAB)
        conv2_cb1_2_RGB = self.conv2_cb1_2_RGB(torch.cat((conv2_cb1_1_RGB, conv2_cb1_1_HSV, conv2_cb1_1_LAB), dim=1))
        del conv2_cb1_1_HSV, conv2_cb1_1_LAB, conv2_cb1_1_RGB
        # print(3)
        # print(conv2_cb1_2_RGB)

        conv2_cb1_3_HSV = self.conv2_cb1_3_HSV(conv2_cb1_2_HSV)
        conv2_cb1_3_LAB = self.conv2_cb1_3_LAB(conv2_cb1_2_LAB)
        conv2_cb1_3_RGB = self.conv2_cb1_3_RGB(torch.cat((conv2_cb1_2_RGB, conv2_cb1_2_HSV, conv2_cb1_2_LAB), dim=1))
        del conv2_cb1_2_HSV, conv2_cb1_2_LAB, conv2_cb1_2_RGB
        # print(4)
        # print(conv2_cb1_3_RGB)

        add_1_HSV = conv2_cb1_3_HSV + conv2_1_HSV
        add_1_LAB = conv2_cb1_3_LAB + conv2_1_LAB
        add_1_RGB = conv2_cb1_3_RGB + conv2_1_RGB
        del conv2_cb1_3_HSV, conv2_cb1_3_LAB, conv2_cb1_3_RGB, conv2_1_HSV, conv2_1_LAB, conv2_1_RGB
        # print(5)
        # print(add_1_RGB)

        conv2_2_HSV = self.conv2_2_HSV(add_1_HSV)
        conv2_2_LAB = self.conv2_2_LAB(add_1_LAB)
        conv2_2_RGB = self.conv2_2_RGB(torch.cat((add_1_RGB, add_1_HSV, add_1_LAB), dim=1))
        del add_1_HSV, add_1_LAB, add_1_RGB
        
        conv2_cb1_4_HSV = self.conv2_cb1_4_HSV(conv2_2_HSV)
        conv2_cb1_4_LAB = self.conv2_cb1_4_LAB(conv2_2_LAB)
        conv2_cb1_4_RGB = self.conv2_cb1_4_RGB(torch.cat((conv2_2_RGB, conv2_2_HSV, conv2_2_LAB), dim=1))
        
        conv2_cb1_5_HSV = self.conv2_cb1_5_HSV(conv2_cb1_4_HSV)
        conv2_cb1_5_LAB = self.conv2_cb1_5_LAB(conv2_cb1_4_LAB)
        conv2_cb1_5_RGB = self.conv2_cb1_5_RGB(torch.cat((conv2_cb1_4_RGB, conv2_cb1_4_HSV, conv2_cb1_4_LAB), dim=1))
        del conv2_cb1_4_HSV, conv2_cb1_4_LAB, conv2_cb1_4_RGB
        
        conv2_cb1_6_HSV = self.conv2_cb1_6_HSV(conv2_cb1_5_HSV)
        conv2_cb1_6_LAB = self.conv2_cb1_6_LAB(conv2_cb1_5_LAB)
        conv2_cb1_6_RGB = self.conv2_cb1_6_RGB(torch.cat((conv2_cb1_5_RGB, conv2_cb1_5_HSV, conv2_cb1_5_LAB), dim=1))
        del conv2_cb1_5_HSV, conv2_cb1_5_LAB, conv2_cb1_5_RGB
        
        encoder_HSV = conv2_cb1_6_HSV + conv2_2_HSV
        encoder_LAB = conv2_cb1_6_LAB + conv2_2_LAB
        encoder_RGB = conv2_cb1_6_RGB + conv2_2_RGB
        del conv2_cb1_6_HSV, conv2_cb1_6_LAB, conv2_cb1_6_RGB, conv2_2_HSV, conv2_2_LAB, conv2_2_RGB

        # print(6)
        # print(encoder_RGB)
        
        return encoder_HSV, encoder_LAB, encoder_RGB 
        
class ChannelAttention(nn.Module):
    def __init__(self, out_channel, ratio):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channel, out_channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // ratio, out_channel),
            nn.Sigmoid()
        )
        self.tail = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 3, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.excitation(self.squeeze(x).view(b, c)).view(b, c, 1, 1)
        y = x * y
        y = self.tail(y)
        
        return y
 
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.MaxPool2d(2, 2)
        self.encoder1 = Encoder_block_lite(3, 128, is_downsample=False)
        self.encoder2 = Encoder_block_lite(128, 256)
        self.encoder3 = Encoder_block_lite(256, 512)
        
        self.se1 = ChannelAttention(128*3, 16)
        self.se2 = ChannelAttention(256*3, 16)
        self.se3 = ChannelAttention(512*3, 16)
        
    def forward(self, x_hsv, x_lab, x_rgb, depth):
        x_hsv1, x_lab1, x_rgb1 = self.encoder1(x_hsv, x_lab, x_rgb)
        
        x_hsv2, x_lab2, x_rgb2 = self.encoder2(x_hsv1, x_lab1, x_rgb1)
        
        x_hsv3, x_lab3, x_rgb3 = self.encoder3(x_hsv2, x_lab2, x_rgb2)
        
        se1_input = torch.cat((x_rgb1, x_hsv1, x_lab1), dim=1)
        del x_rgb1, x_hsv1, x_lab1
        first_con = self.se1(se1_input)
        depth1 = 1 - depth
        feature_first = first_con * depth1 + first_con
        
        se2_input = torch.cat((x_rgb2, x_hsv2, x_lab2), dim=1)
        del x_rgb2, x_hsv2, x_lab2
        second_con = self.se2(se2_input)
        depth2 = self.downsample(depth1)
        feature_second = second_con * depth2 + second_con
        
        se3_input = torch.cat((x_rgb3, x_hsv3, x_lab3), dim=1)
        del x_rgb3, x_hsv3, x_lab3
        third_con = self.se3(se3_input)
        depth3 = self.downsample(depth2)
        feature_third = third_con * depth3 + third_con

        return feature_first, feature_second, feature_third

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv2_1_HSV = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_1_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_2_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_3_HSV = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        
        self.conv2_2_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_4_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_5_HSV = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv2_cb1_6_HSV = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
    
    def forward(self, x):
        conv2_1_HSV = self.conv2_1_HSV(x)
        conv2_cb1_1_HSV = self.conv2_cb1_1_HSV(conv2_1_HSV)
        conv2_cb1_1_HSV = self.conv2_cb1_2_HSV(conv2_cb1_1_HSV)
        conv2_cb1_1_HSV = self.conv2_cb1_3_HSV(conv2_cb1_1_HSV)
        add_1_HSV = conv2_1_HSV + conv2_cb1_1_HSV
        del conv2_1_HSV, conv2_cb1_1_HSV
        
        conv2_2_HSV = self.conv2_2_HSV(add_1_HSV)
        conv2_cb1_4_HSV = self.conv2_cb1_4_HSV(conv2_2_HSV)
        conv2_cb1_4_HSV = self.conv2_cb1_5_HSV(conv2_cb1_4_HSV)
        conv2_cb1_4_HSV = self.conv2_cb1_6_HSV(conv2_cb1_4_HSV)
        encoder_HSV = conv2_2_HSV + conv2_cb1_4_HSV
        del conv2_2_HSV, conv2_cb1_4_HSV
        
        return encoder_HSV
           
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder1 = ResidualBlock(512, 512)
        self.decoder2 = ResidualBlock(512+256, 256)
        self.decoder3 = ResidualBlock(256+128, 128)
        self.refine = nn.Conv2d(128, 3, 3, 1, 1)
        
        
    def forward(self, feature_first, feature_second, feature_third):
        d1 = self.decoder1(feature_third)
        d1 = self.upsampler(d1)
        
        d2 = self.decoder2(torch.cat((d1, feature_second), dim=1))
        d2 = self.upsampler(d2)
        
        d3 = self.decoder3(torch.cat((d2, feature_first), dim=1))
        d3 = self.refine(d3)

        return d3    
    
class UColor(nn.Module):
    def __init__(self, init_way=None):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        if init_way:
        # weight initialization
            init_class = getattr(nn.init, init_way)
            for m in self.modules():
                init_class(m.weight) if isinstance(m, nn.Conv2d) else None
    
    def forward(self, hsv_img, lab_img, rgb_img, depth, **kwargs):
        feature_first, feature_second, feature_third = self.encoder(hsv_img, lab_img, rgb_img, depth)
        result = self.decoder(feature_first, feature_second, feature_third)
        
        return result