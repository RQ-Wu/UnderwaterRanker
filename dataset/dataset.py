import imghdr
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as transFunc


import os
import yaml
import numpy as np
from PIL import Image
from random import Random
import matplotlib.pyplot as plt

class Dataset_Ucolor(data.Dataset):
    def __init__(self, opt, type="train", **kwargs):
        self.opt = opt
        self.type = type
        self.filenames = open(opt[self.type + '_list_path']).readlines()

    def __getitem__(self, index):
        filepath = os.path.join(self.opt['root'], self.filenames[index].rstrip())
        all_data = np.load(filepath[:-3] + 'npy')

        rgb_img = transforms.ToTensor()(all_data[:, :, :3]).type(torch.float32)
        lab_img = transforms.ToTensor()(all_data[:, :, 3:6]).type(torch.float32)
        hsv_img = transforms.ToTensor()(all_data[:, :, 6:9]).type(torch.float32)
        gt_img = transforms.ToTensor()(all_data[:, :, 9:12]).type(torch.float32)
        depth = transforms.ToTensor()(all_data[:, :, 12:]).type(torch.float32)

        # data aug for training
        if self.type == "train":
            # random crop
            i, j, h, w = transforms.RandomCrop(self.opt['crop_size']).get_params(rgb_img, (self.opt['crop_size'], self.opt['crop_size']))
            rgb_img = transFunc.crop(rgb_img, i, j, h, w)        
            lab_img = transFunc.crop(lab_img, i, j, h, w)        
            hsv_img = transFunc.crop(hsv_img, i, j, h, w)        
            gt_img = transFunc.crop(gt_img, i, j, h, w)        
            depth = transFunc.crop(depth, i, j, h, w)
            output = {
                'rgb_img': rgb_img / 255.0,
                'lab_img': lab_img / torch.tensor([100.0, 255.0, 255.0]).unsqueeze(1).unsqueeze(2),
                'hsv_img': hsv_img,
                'depth': depth / 255.0,
                'gt_img': gt_img / 255.0
            }
            return output
                    
        # make sure resolution will not be changed during encoder-decoder stage
        elif self.type == "test":
            img_h, img_w = rgb_img.shape[1], rgb_img.shape[2]
            # print(rgb_img.shape)
            rgb_img = transforms.Resize((img_h // 4 * 4, img_w // 4 * 4))(rgb_img)
            lab_img = transforms.Resize((img_h // 4 * 4, img_w // 4 * 4))(lab_img)
            hsv_img = transforms.Resize((img_h // 4 * 4, img_w // 4 * 4))(hsv_img)
            gt_img = transforms.Resize((img_h // 4 * 4, img_w // 4 * 4))(gt_img)
            depth = transforms.Resize((img_h // 4 * 4, img_w // 4 * 4))(depth)
            output = {
                'rgb_img': rgb_img / 255.0,
                'lab_img': lab_img / torch.tensor([100.0, 255.0, 255.0]).unsqueeze(1).unsqueeze(2),
                'hsv_img': hsv_img,
                'depth': depth / 255.0,
                'gt_img': gt_img / 255.0,
            }
            return output, self.filenames[index].rstrip()
    
    def __len__(self):
        return len(self.filenames)
        