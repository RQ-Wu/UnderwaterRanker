from tkinter import E
import utils
import torch
import argparse
import os
import skimage.io as io
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()

options = utils.get_option(args.opt_path)
options['model']['resume_ckpt_path'] = args.checkpoint_path
model = utils.build_model(options['model'])
filenames = os.listdir(args.input_path)
utils.make_dir(args.save_path)
for filename in filenames:
    filepath = os.path.join(args.input_path, filename)
    img = Image.open(filepath)
    img_w, img_h = img.size[0], img.size[1]
    
    upsample = nn.UpsamplingBilinear2d((img_h, img_w))
    img = transforms.Resize((img_h//16*16, img_w//16*16))(img)
    img = transforms.ToTensor()(img).cuda().unsqueeze(0)
    
    pred = upsample(utils.normalize_img(model(img)))[0].permute(1, 2, 0).detach().cpu().numpy()
    cv2.imwrite(os.path.join(args.save_path, filename), pred[:, :, ::-1] * 255.0)
    print(filename, 'is done!')

