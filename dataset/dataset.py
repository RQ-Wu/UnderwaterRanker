import imghdr
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as transFunc
import torchvision.transforms.functional as F

import os
import yaml
import numpy as np
from PIL import Image
from random import Random
import matplotlib.pyplot as plt
from utils import build_historgram

myrandom = Random(567)
class Dataset_UIEB(data.Dataset):
    def __init__(self, opt, type="train"):
        self.raw_path = opt['root'] + 'raw-890/'
        self.gt_path = opt['root'] + 'reference-890/reference-890/'
        self.crop_size = opt['crop_size']
        self.type = type
        if self.type == "train":
            f = open(opt['train_list_path'])
        elif self.type == "test" or self.type == 'valid':
            f = open(opt['test_list_path'])
        self.filenames = f.readlines()

    def __getitem__(self, item):
        raw_item_path = self.raw_path + self.filenames[item].rstrip()
        gt_item_path = self.gt_path + self.filenames[item].rstrip()

        raw_img = Image.open(raw_item_path)
        gt_img = Image.open(gt_item_path)
        img_w = raw_img.size[0]
        img_h = raw_img.size[1]

        if self.type == "train":
            # random resize and crop to 256 x 256
            i, j, h, w = transforms.RandomResizedCrop(self.crop_size).get_params(raw_img, (0.08, 1.0),
                                                                                  (3. / 4., 4. / 3.))
            raw_cropped = F.resized_crop(raw_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            gt_cropped = F.resized_crop(gt_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            raw_cropped = transforms.ToTensor()(raw_cropped)
            gt_cropped = transforms.ToTensor()(gt_cropped)
            

            if np.random.rand(1) < 0.5:  # flip horizonly
                raw_cropped = torch.flip(raw_cropped, [2])
                gt_cropped = torch.flip(gt_cropped, [2])
            if np.random.rand(1) < 0.5:  # flip vertically
                raw_cropped = torch.flip(raw_cropped, [1])
                gt_cropped = torch.flip(gt_cropped, [1])

            return {'raw_img':raw_cropped, 'gt_img':gt_cropped}
        
        elif self.type == "test":
            raw_img = transforms.Resize((img_h // 16 * 16, img_w // 16 * 16))(raw_img)
            raw_img = transforms.ToTensor()(raw_img)
            gt_img = transforms.ToTensor()(gt_img)

            return {'raw_img':raw_img, 'gt_img':gt_img}
            
    def __len__(self):
        return len(self.filenames)

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
            # gt_img = transforms.Resize((img_h // 4 * 4, img_w // 4 * 4))(gt_img)
            depth = transforms.Resize((img_h // 4 * 4, img_w // 4 * 4))(depth)
            output = {
                'rgb_img': rgb_img / 255.0,
                'lab_img': lab_img / torch.tensor([100.0, 255.0, 255.0]).unsqueeze(1).unsqueeze(2),
                'hsv_img': hsv_img,
                'depth': depth / 255.0,
                'gt_img': gt_img / 255.0,
            }
            return output
    
    def __len__(self):
        return len(self.filenames)

class Dataset_Ranker(data.Dataset):
    def __init__(self, opt, type="train"):
        self.opt = opt
        self.type = type
        self.filenames = open(opt[self.type+'_list_path']).readlines()
        
    def __getitem__(self, index):
        if self.type == "train":
            # random choose the ranks
            random_rank = np.random.randint(0, 10, 2)
            while random_rank[0] == random_rank[1]:
                random_rank = np.random.randint(0, 10, 2)
            high_rank = np.min(random_rank)
            low_rank = np.max(random_rank)
            
            # read image pair
            high_path = os.path.join(self.opt['root'], self.filenames[index * 10 + high_rank][:-1])
            low_path = os.path.join(self.opt['root'], self.filenames[index * 10 + low_rank][:-1])
            
            high_img = Image.open(high_path)
            low_img = Image.open(low_path)
            img_w, img_h = high_img.size[0], high_img.size[1]
            
            # data augumentations
            high_img = transforms.Resize((img_h//2, img_w//2))(high_img)
            low_img = transforms.Resize((img_h//2, img_w//2))(low_img)
            
            high_img = transforms.ToTensor()(high_img)
            low_img = transforms.ToTensor()(low_img)
            if np.random.rand(1) < 0.5:  # flip horizonly
                high_img = torch.flip(high_img, [2])
                low_img = torch.flip(low_img, [2])
            if np.random.rand(1) < 0.5:  # flip vertically
                high_img = torch.flip(high_img, [1])
                low_img = torch.flip(low_img, [1])

            output = {
                'high_img': high_img,
                'low_img': low_img,
                'high_rank': high_rank,
                'low_rank': low_rank
            }

        elif self.type == "test":
            img_path = os.path.join(self.opt['root'], self.filenames[index][:-1])
            img = Image.open(img_path)
            img_w, img_h = img.size[0], img.size[1]
            
            img = transforms.Resize((img_h//2, img_w//2))(img)
            img = transforms.ToTensor()(img)

            output = {
                'img': img
            }

        return output
        
    def __len__(self):
        return len(self.filenames) // 10 if self.type == "train" else len(self.filenames)